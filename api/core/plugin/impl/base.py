import inspect
import json
import logging
from collections.abc import Callable, Generator
from typing import TypeVar

import requests
from pydantic import BaseModel
from requests.exceptions import HTTPError
from yarl import URL

from configs import dify_config
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.plugin.endpoint.exc import EndpointSetupFailedError
from core.plugin.entities.plugin_daemon import PluginDaemonBasicResponse, PluginDaemonError, PluginDaemonInnerError
from core.plugin.impl.exc import (
    PluginDaemonBadRequestError,
    PluginDaemonInternalServerError,
    PluginDaemonNotFoundError,
    PluginDaemonUnauthorizedError,
    PluginInvokeError,
    PluginNotFoundError,
    PluginPermissionDeniedError,
    PluginUniqueIdentifierError,
)

plugin_daemon_inner_api_baseurl = URL(str(dify_config.PLUGIN_DAEMON_URL))

T = TypeVar("T", bound=(BaseModel | dict | list | bool | str))

logger = logging.getLogger(__name__)


class BasePluginClient:
    def _request(
        self,
        method: str,
        path: str,
        headers: dict | None = None,
        data: bytes | dict | str | None = None,
        params: dict | None = None,
        files: dict | None = None,
        stream: bool = False,
    ) -> requests.Response:
        """
        向插件守护进程内部API发起请求
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param headers: 请求头字典
        :param data: 请求数据，可以是字节、字典或字符串
        :param params: 查询参数字典
        :param files: 文件参数字典
        :param stream: 是否以流式方式处理响应
        :return: requests.Response对象
        """
        url = plugin_daemon_inner_api_baseurl / path
        headers = headers or {}
        headers["X-Api-Key"] = dify_config.PLUGIN_DAEMON_KEY
        headers["Accept-Encoding"] = "gzip, deflate, br"

        if headers.get("Content-Type") == "application/json" and isinstance(data, dict):
            data = json.dumps(data)

        try:
            response = requests.request(
                method=method, url=str(url), headers=headers, data=data, params=params, stream=stream, files=files
            )
        except requests.ConnectionError:
            logger.exception("Request to Plugin Daemon Service failed")
            raise PluginDaemonInnerError(code=-500, message="Request to Plugin Daemon Service failed")

        return response

    def _stream_request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        data: bytes | dict | None = None,
        files: dict | None = None,
    ) -> Generator[bytes, None, None]:
        """
        向插件守护进程内部API发起流式请求
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param params: 查询参数字典
        :param headers: 请求头字典
        :param data: 请求数据，可以是字节或字典
        :param files: 文件参数字典
        :return: 返回字节数据的生成器
        """
        response = self._request(method, path, headers, data, params, files, stream=True)
        for line in response.iter_lines(chunk_size=1024 * 8):
            line = line.decode("utf-8").strip()
            # 处理SSE格式的数据行
            if line.startswith("data:"):
                line = line[5:].strip()
            if line:
                yield line

    def _stream_request_with_model(
        self,
        method: str,
        path: str,
        type: type[T],
        headers: dict | None = None,
        data: bytes | dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
    ) -> Generator[T, None, None]:
        """
        向插件守护进程内部API发起流式请求，并将响应作为模型对象返回
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param type: 期望返回的数据类型
        :param headers: 请求头字典
        :param data: 请求数据，可以是字节或字典
        :param params: 查询参数字典
        :param files: 文件参数字典
        :return: 返回指定类型数据的生成器
        """
        for line in self._stream_request(method, path, params, headers, data, files):
            yield type(**json.loads(line))  # type: ignore

    def _request_with_model(
        self,
        method: str,
        path: str,
        type: type[T],
        headers: dict | None = None,
        data: bytes | None = None,
        params: dict | None = None,
        files: dict | None = None,
    ) -> T:
        """
        向插件守护进程内部API发起请求，并将响应作为模型对象返回
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param type: 期望返回的数据类型
        :param headers: 请求头字典
        :param data: 请求数据（字节类型）
        :param params: 查询参数字典
        :param files: 文件参数字典
        :return: 指定类型的响应数据
        """
        response = self._request(method, path, headers, data, params, files)
        return type(**response.json())  # type: ignore

    def _request_with_plugin_daemon_response(
        self,
        method: str,
        path: str,
        type: type[T],
        headers: dict | None = None,
        data: bytes | dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        transformer: Callable[[dict], dict] | None = None,
    ) -> T:
        """
        向插件守护进程内部API发起请求，并将响应作为模型对象返回
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param type: 期望返回的数据类型
        :param headers: 请求头字典
        :param data: 请求数据，可以是字节或字典
        :param params: 查询参数字典
        :param files: 文件参数字典
        :param transformer: 数据转换函数
        :return: 指定类型的响应数据
        """
        try:
            response = self._request(method, path, headers, data, params, files)
            response.raise_for_status()
        except HTTPError as e:
            msg = f"Failed to request plugin daemon, status: {e.response.status_code}, url: {path}"
            logger.exception(msg)
            raise e
        except Exception as e:
            msg = f"Failed to request plugin daemon, url: {path}"
            logger.exception(msg)
            raise ValueError(msg) from e

        try:
            json_response = response.json()
            if transformer:
                json_response = transformer(json_response)
            rep = PluginDaemonBasicResponse[type](**json_response)  # type: ignore
        except Exception:
            msg = (
                f"Failed to parse response from plugin daemon to PluginDaemonBasicResponse [{str(type.__name__)}],"
                f" url: {path}"
            )
            logger.exception(msg)
            raise ValueError(msg)

        if rep.code != 0:
            try:
                error = PluginDaemonError(**json.loads(rep.message))
            except Exception:
                raise ValueError(f"{rep.message}, code: {rep.code}")

            self._handle_plugin_daemon_error(error.error_type, error.message)
        if rep.data is None:
            frame = inspect.currentframe()
            raise ValueError(f"got empty data from plugin daemon: {frame.f_lineno if frame else 'unknown'}")

        return rep.data

    def _request_with_plugin_daemon_response_stream(
        self,
        method: str,
        path: str,
        type: type[T],
        headers: dict | None = None,
        data: bytes | dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
    ) -> Generator[T, None, None]:
        """
        向插件守护进程内部API发起流式请求，并将响应作为模型对象返回
        
        :param method: HTTP请求方法
        :param path: 请求路径
        :param type: 期望返回的数据类型
        :param headers: 请求头字典
        :param data: 请求数据，可以是字节或字典
        :param params: 查询参数字典
        :param files: 文件参数字典
        :return: 返回指定类型数据的生成器
        """
        for line in self._stream_request(method, path, params, headers, data, files):
            try:
                rep = PluginDaemonBasicResponse[type].model_validate_json(line)  # type: ignore
            except (ValueError, TypeError):
                # TODO modify this when line_data has code and message
                try:
                    line_data = json.loads(line)
                except (ValueError, TypeError):
                    raise ValueError(line)
                # If the dictionary contains the `error` key, use its value as the argument
                # for `ValueError`.
                # Otherwise, use the `line` to provide better contextual information about the error.
                raise ValueError(line_data.get("error", line))

            if rep.code != 0:
                if rep.code == -500:
                    try:
                        error = PluginDaemonError(**json.loads(rep.message))
                    except Exception:
                        raise PluginDaemonInnerError(code=rep.code, message=rep.message)

                    logger.error("Error in stream reponse for plugin %s", rep.__dict__)
                    self._handle_plugin_daemon_error(error.error_type, error.message)
                raise ValueError(f"plugin daemon: {rep.message}, code: {rep.code}")
            if rep.data is None:
                frame = inspect.currentframe()
                raise ValueError(f"got empty data from plugin daemon: {frame.f_lineno if frame else 'unknown'}")
            yield rep.data

    def _handle_plugin_daemon_error(self, error_type: str, message: str):
        """
        处理来自插件守护进程的错误
        
        :param error_type: 错误类型
        :param message: 错误消息
        """
        match error_type:
            case PluginDaemonInnerError.__name__:
                raise PluginDaemonInnerError(code=-500, message=message)
            case PluginInvokeError.__name__:
                error_object = json.loads(message)
                invoke_error_type = error_object.get("error_type")
                args = error_object.get("args")
                match invoke_error_type:
                    case InvokeRateLimitError.__name__:
                        raise InvokeRateLimitError(description=args.get("description"))
                    case InvokeAuthorizationError.__name__:
                        raise InvokeAuthorizationError(description=args.get("description"))
                    case InvokeBadRequestError.__name__:
                        raise InvokeBadRequestError(description=args.get("description"))
                    case InvokeConnectionError.__name__:
                        raise InvokeConnectionError(description=args.get("description"))
                    case InvokeServerUnavailableError.__name__:
                        raise InvokeServerUnavailableError(description=args.get("description"))
                    case CredentialsValidateFailedError.__name__:
                        raise CredentialsValidateFailedError(error_object.get("message"))
                    case EndpointSetupFailedError.__name__:
                        raise EndpointSetupFailedError(error_object.get("message"))
                    case _:
                        raise PluginInvokeError(description=message)
            case PluginDaemonInternalServerError.__name__:
                raise PluginDaemonInternalServerError(description=message)
            case PluginDaemonBadRequestError.__name__:
                raise PluginDaemonBadRequestError(description=message)
            case PluginDaemonNotFoundError.__name__:
                raise PluginDaemonNotFoundError(description=message)
            case PluginUniqueIdentifierError.__name__:
                raise PluginUniqueIdentifierError(description=message)
            case PluginNotFoundError.__name__:
                raise PluginNotFoundError(description=message)
            case PluginDaemonUnauthorizedError.__name__:
                raise PluginDaemonUnauthorizedError(description=message)
            case PluginPermissionDeniedError.__name__:
                raise PluginPermissionDeniedError(description=message)
            case _:
                raise Exception(f"got unknown error from plugin daemon: {error_type}, message: {message}")