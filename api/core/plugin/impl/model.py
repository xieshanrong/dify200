
import binascii
from collections.abc import Generator, Sequence
from typing import IO, Optional

from core.model_runtime.entities.llm_entities import LLMResultChunk
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.entities.model_entities import AIModelEntity
from core.model_runtime.entities.rerank_entities import RerankResult
from core.model_runtime.entities.text_embedding_entities import TextEmbeddingResult
from core.model_runtime.utils.encoders import jsonable_encoder
from core.plugin.entities.plugin_daemon import (
    PluginBasicBooleanResponse,
    PluginDaemonInnerError,
    PluginLLMNumTokensResponse,
    PluginModelProviderEntity,
    PluginModelSchemaEntity,
    PluginStringResultResponse,
    PluginTextEmbeddingNumTokensResponse,
    PluginVoicesResponse,
)
from core.plugin.impl.base import BasePluginClient


class PluginModelClient(BasePluginClient):
    def fetch_model_providers(self, tenant_id: str) -> Sequence[PluginModelProviderEntity]:
        """
        获取指定租户的模型提供商列表。

        :param tenant_id: 租户ID。
        :return: 模型提供商实体列表。
        """
        response = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/models",
            list[PluginModelProviderEntity],
            params={"page": 1, "page_size": 256},
        )
        return response

    def get_model_schema(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model_type: str,
        model: str,
        credentials: dict,
    ) -> AIModelEntity | None:
        """
        获取模型的元数据结构定义（schema）。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model_type: 模型类型。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :return: AI模型实体或None。
        """
        response = self._request_with_plugin_daemon_response_stream(
            "POST",
            f"plugin/{tenant_id}/dispatch/model/schema",
            PluginModelSchemaEntity,
            data={
                "user_id": user_id,
                "data": {
                    "provider": provider,
                    "model_type": model_type,
                    "model": model,
                    "credentials": credentials,
                },
            },
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp.model_schema

        return None

    def validate_provider_credentials(
        self, tenant_id: str, user_id: str, plugin_id: str, provider: str, credentials: dict
    ) -> bool:
        """
        验证提供商的认证凭据是否有效。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param credentials: 认证信息字典。
        :return: 凭据验证结果布尔值。
        """
        response = self._request_with_plugin_daemon_response_stream(
            "POST",
            f"plugin/{tenant_id}/dispatch/model/validate_provider_credentials",
            PluginBasicBooleanResponse,
            data={
                "user_id": user_id,
                "data": {
                    "provider": provider,
                    "credentials": credentials,
                },
            },
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            if resp.credentials and isinstance(resp.credentials, dict):
                credentials.update(resp.credentials)

            return resp.result

        return False

    def validate_model_credentials(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model_type: str,
        model: str,
        credentials: dict,
    ) -> bool:
        """
        验证特定模型的认证凭据是否有效。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model_type: 模型类型。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :return: 凭据验证结果布尔值。
        """
        response = self._request_with_plugin_daemon_response_stream(
            "POST",
            f"plugin/{tenant_id}/dispatch/model/validate_model_credentials",
            PluginBasicBooleanResponse,
            data={
                "user_id": user_id,
                "data": {
                    "provider": provider,
                    "model_type": model_type,
                    "model": model,
                    "credentials": credentials,
                },
            },
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            if resp.credentials and isinstance(resp.credentials, dict):
                credentials.update(resp.credentials)

            return resp.result

        return False

    def invoke_llm(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: Optional[dict] = None,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
    ) -> Generator[LLMResultChunk, None, None]:
        """
        调用大语言模型进行推理。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param prompt_messages: 输入提示消息列表。
        :param model_parameters: 可选的模型参数。
        :param tools: 可选的工具调用列表。
        :param stop: 停止词列表。
        :param stream: 是否以流式方式返回响应。
        :yield: 大语言模型输出的结果块。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/llm/invoke",
            type=LLMResultChunk,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "llm",
                        "model": model,
                        "credentials": credentials,
                        "prompt_messages": prompt_messages,
                        "model_parameters": model_parameters,
                        "tools": tools,
                        "stop": stop,
                        "stream": stream,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        try:
            yield from response
        except PluginDaemonInnerError as e:
            raise ValueError(e.message + str(e.code))

    def get_llm_num_tokens(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model_type: str,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        获取大语言模型输入文本的token数量。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model_type: 模型类型。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param prompt_messages: 输入提示消息列表。
        :param tools: 工具调用列表。
        :return: token数量整数。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/llm/num_tokens",
            type=PluginLLMNumTokensResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": model_type,
                        "model": model,
                        "credentials": credentials,
                        "prompt_messages": prompt_messages,
                        "tools": tools,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp.num_tokens

        return 0

    def invoke_text_embedding(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        texts: list[str],
        input_type: str,
    ) -> TextEmbeddingResult:
        """
        调用文本嵌入模型生成向量表示。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param texts: 待处理文本列表。
        :param input_type: 输入类型标识符。
        :return: 文本嵌入结果对象。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/text_embedding/invoke",
            type=TextEmbeddingResult,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "text-embedding",
                        "model": model,
                        "credentials": credentials,
                        "texts": texts,
                        "input_type": input_type,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp

        raise ValueError("Failed to invoke text embedding")

    def get_text_embedding_num_tokens(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        texts: list[str],
    ) -> list[int]:
        """
        获取文本嵌入模型中各段文本对应的token数量。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param texts: 待分析文本列表。
        :return: 各段文本的token数量组成的列表。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/text_embedding/num_tokens",
            type=PluginTextEmbeddingNumTokensResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "text-embedding",
                        "model": model,
                        "credentials": credentials,
                        "texts": texts,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp.num_tokens

        return []

    def invoke_rerank(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> RerankResult:
        """
        对文档进行重排序操作。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param query: 查询语句。
        :param docs: 待排序文档列表。
        :param score_threshold: 最低得分阈值。
        :param top_n: 返回前N个结果。
        :return: 重新排序后的结果对象。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/rerank/invoke",
            type=RerankResult,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "rerank",
                        "model": model,
                        "credentials": credentials,
                        "query": query,
                        "docs": docs,
                        "score_threshold": score_threshold,
                        "top_n": top_n,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp

        raise ValueError("Failed to invoke rerank")

    def invoke_tts(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        content_text: str,
        voice: str,
    ) -> Generator[bytes, None, None]:
        """
        将文本转换为语音音频数据。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param content_text: 待合成的文本内容。
        :param voice: 使用的声音风格。
        :yield: 音频二进制数据流。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/tts/invoke",
            type=PluginStringResultResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "tts",
                        "model": model,
                        "credentials": credentials,
                        "tenant_id": tenant_id,
                        "content_text": content_text,
                        "voice": voice,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        try:
            for result in response:
                hex_str = result.result
                yield binascii.unhexlify(hex_str)
        except PluginDaemonInnerError as e:
            raise ValueError(e.message + str(e.code))

    def get_tts_model_voices(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        language: Optional[str] = None,
    ):
        """
        获取TTS模型支持的所有声音选项。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param language: 可选的语言筛选条件。
        :return: 支持的声音选项列表。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/tts/model/voices",
            type=PluginVoicesResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "tts",
                        "model": model,
                        "credentials": credentials,
                        "language": language,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            voices = []
            for voice in resp.voices:
                voices.append({"name": voice.name, "value": voice.value})

            return voices

        return []

    def invoke_speech_to_text(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        file: IO[bytes],
    ) -> str:
        """
        将语音文件转录为文字内容。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param file: 包含语音数据的IO对象。
        :return: 转录出的文字字符串。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/speech2text/invoke",
            type=PluginStringResultResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "speech2text",
                        "model": model,
                        "credentials": credentials,
                        "file": binascii.hexlify(file.read()).decode(),
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp.result

        raise ValueError("Failed to invoke speech to text")

    def invoke_moderation(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        model: str,
        credentials: dict,
        text: str,
    ) -> bool:
        """
        判断给定文本是否违反了平台的内容审核规则。

        :param tenant_id: 租户ID。
        :param user_id: 用户ID。
        :param plugin_id: 插件ID。
        :param provider: 提供商名称。
        :param model: 模型名称。
        :param credentials: 认证信息字典。
        :param text: 待检测的文本内容。
        :return: 审核通过与否的布尔值。
        """
        response = self._request_with_plugin_daemon_response_stream(
            method="POST",
            path=f"plugin/{tenant_id}/dispatch/moderation/invoke",
            type=PluginBasicBooleanResponse,
            data=jsonable_encoder(
                {
                    "user_id": user_id,
                    "data": {
                        "provider": provider,
                        "model_type": "moderation",
                        "model": model,
                        "credentials": credentials,
                        "text": text,
                    },
                }
            ),
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        for resp in response:
            return resp.result

        raise ValueError("Failed to invoke moderation")
