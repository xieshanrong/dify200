import logging
import os
from collections.abc import Generator
from pathlib import Path

import opendal  # type: ignore[import]
from dotenv import dotenv_values

from extensions.storage.base_storage import BaseStorage

logger = logging.getLogger(__name__)


def _get_opendal_kwargs(*, scheme: str, env_file_path: str = ".env", prefix: str = "OPENDAL_"):
    """
    从环境变量和.env文件中获取OpenDAL的配置参数。

    该函数根据给定的scheme和前缀，从环境变量和指定的.env文件中提取配置参数。

    Args:
        scheme: 存储方案名称，如"S3"、"LOCAL"等
        env_file_path: 包含环境变量的文件路径，默认为".env"
        prefix: 环境变量的前缀，默认为"OPENDAL_"

    Returns:
        dict: 包含OpenDAL配置参数的字典
    """
    kwargs = {}
    config_prefix = prefix + scheme.upper() + "_"

    # 从环境变量中提取配置
    for key, value in os.environ.items():
        if key.startswith(config_prefix):
            kwargs[key[len(config_prefix):].lower()] = value

    # 从.env文件中提取配置
    file_env_vars: dict = dotenv_values(env_file_path) or {}
    for key, value in file_env_vars.items():
        if key.startswith(config_prefix) and key[len(config_prefix):].lower() not in kwargs and value:
            kwargs[key[len(config_prefix):].lower()] = value

    return kwargs


class OpenDALStorage(BaseStorage):
    def __init__(self, scheme: str, **kwargs):
        """
        初始化OpenDAL存储操作符。

        该方法根据给定的scheme和配置参数初始化OpenDAL操作符，并添加重试层以增强可靠性。

        Args:
            scheme: 存储方案名称，如"S3"、"LOCAL"等
            **kwargs: OpenDAL的额外配置参数
        """
        kwargs = kwargs or _get_opendal_kwargs(scheme=scheme)

        if scheme == "fs":
            # 对于本地文件系统，确保根目录存在
            root = kwargs.get("root", "storage")
            Path(root).mkdir(parents=True, exist_ok=True)

        # 初始化OpenDAL操作符
        self.op = opendal.Operator(scheme=scheme, **kwargs)  # type: ignore
        logger.debug("opendal operator created with scheme %s", scheme)

        # 添加重试层以处理临时性错误
        retry_layer = opendal.layers.RetryLayer(max_times=3, factor=2.0, jitter=True)
        self.op = self.op.layer(retry_layer)
        logger.debug("added retry layer to opendal operator")

    def save(self, filename: str, data: bytes):
        """
        保存文件数据到指定路径。

        Args:
            filename: 文件名
            data: 文件数据（字节）
        """
        self.op.write(path=filename, bs=data)
        logger.debug("file %s saved", filename)

    def load_once(self, filename: str) -> bytes:
        """
        一次性加载指定文件的内容。

        Args:
            filename: 文件名

        Returns:
            bytes: 文件内容

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        content: bytes = self.op.read(path=filename)
        logger.debug("file %s loaded", filename)
        return content

    def load_stream(self, filename: str) -> Generator:
        """
        流式加载指定文件的内容，返回生成器。

        Args:
            filename: 文件名

        Returns:
            Generator: 文件内容生成器

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        batch_size = 4096
        file = self.op.open(path=filename, mode="rb")
        while chunk := file.read(batch_size):
            yield chunk
        logger.debug("file %s loaded as stream", filename)

    def download(self, filename: str, target_filepath: str):
        """
        下载指定文件到本地指定路径。

        Args:
            filename: 文件名
            target_filepath: 目标文件路径
        """
        if not self.exists(filename):
            raise FileNotFoundError("File not found")

        with Path(target_filepath).open("wb") as f:
            f.write(self.op.read(path=filename))
        logger.debug("file %s downloaded to %s", filename, target_filepath)

    def exists(self, filename: str) -> bool:
        """
        检查指定文件是否存在。

        Args:
            filename: 文件名

        Returns:
            bool: 文件是否存在
        """
        res: bool = self.op.exists(path=filename)
        return res

    def delete(self, filename: str):
        """
        删除指定文件。

        Args:
            filename: 文件名
        """
        if self.exists(filename):
            self.op.delete(path=filename)
            logger.debug("file %s deleted", filename)
            return
        logger.debug("file %s not found, skip delete", filename)

    def scan(self, path: str, files: bool = True, directories: bool = False) -> list[str]:
        """
        扫描指定路径下的文件和目录。

        Args:
            path: 扫描路径
            files: 是否包含文件，默认为True
            directories: 是否包含目录，默认为False

        Returns:
            list[str]: 文件或目录的路径列表

        Raises:
            FileNotFoundError: 如果指定的路径不存在
            ValueError: 如果files和directories都为False
        """
        if not self.exists(path):
            raise FileNotFoundError("Path not found")

        all_files = self.op.scan(path=path)

        if files and directories:
            logger.debug("files and directories on %s scanned", path)
            return [f.path for f in all_files]
        if files:
            logger.debug("files on %s scanned", path)
            return [f.path for f in all_files if not f.path.endswith("/")]
        elif directories:
            logger.debug("directories on %s scanned", path)
            return [f.path for f in all_files if f.path.endswith("/")]
        else:
            raise ValueError("At least one of files or directories must be True")
