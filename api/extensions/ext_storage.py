import logging
from collections.abc import Callable, Generator
from typing import Literal, Union, overload

from flask import Flask

from configs import dify_config
from dify_app import DifyApp
from extensions.storage.base_storage import BaseStorage
from extensions.storage.storage_type import StorageType

logger = logging.getLogger(__name__)


class Storage:
    def init_app(self, app: Flask):
        """
        初始化应用程序的存储功能。

        该方法根据配置的存储类型初始化存储功能，并创建对应的存储运行器。

        Args:
            app: Flask应用程序实例
        """
        storage_factory = self.get_storage_factory(dify_config.STORAGE_TYPE)
        with app.app_context():
            self.storage_runner = storage_factory()

    @staticmethod
    def get_storage_factory(storage_type: str) -> Callable[[], BaseStorage]:
        """
        根据存储类型获取对应的存储工厂函数。

        该方法使用匹配语句（`match`）来根据不同的存储类型加载相应的存储实现。

        Args:
            storage_type: 存储类型字符串，支持的类型包括：
                - S3
                - OPENDAL
                - LOCAL
                - AZURE_BLOB
                - ALIYUN_OSS
                - GOOGLE_STORAGE
                - TENCENT_COS
                - OCI_STORAGE
                - HUAWEI_OBS
                - BAIDU_OBS
                - VOLCENGINE_TOS
                - SUPABASE
                - CLICKZETTA_VOLUME

        Returns:
            Callable[[], BaseStorage]: 返回一个工厂函数，用于创建对应的存储实例。

        Raises:
            ValueError: 当存储类型不支持时抛出异常。
        """
        match storage_type:
            case StorageType.S3:
                from extensions.storage.aws_s3_storage import AwsS3Storage
                return AwsS3Storage
            case StorageType.OPENDAL:
                from extensions.storage.opendal_storage import OpenDALStorage
                return lambda: OpenDALStorage(dify_config.OPENDAL_SCHEME)
            case StorageType.LOCAL:
                from extensions.storage.opendal_storage import OpenDALStorage
                return lambda: OpenDALStorage(scheme="fs", root=dify_config.STORAGE_LOCAL_PATH)
            case StorageType.AZURE_BLOB:
                from extensions.storage.azure_blob_storage import AzureBlobStorage
                return AzureBlobStorage
            case StorageType.ALIYUN_OSS:
                from extensions.storage.aliyun_oss_storage import AliyunOssStorage
                return AliyunOssStorage
            case StorageType.GOOGLE_STORAGE:
                from extensions.storage.google_cloud_storage import GoogleCloudStorage
                return GoogleCloudStorage
            case StorageType.TENCENT_COS:
                from extensions.storage.tencent_cos_storage import TencentCosStorage
                return TencentCosStorage
            case StorageType.OCI_STORAGE:
                from extensions.storage.oracle_oci_storage import OracleOCIStorage
                return OracleOCIStorage
            case StorageType.HUAWEI_OBS:
                from extensions.storage.huawei_obs_storage import HuaweiObsStorage
                return HuaweiObsStorage
            case StorageType.BAIDU_OBS:
                from extensions.storage.baidu_obs_storage import BaiduObsStorage
                return BaiduObsStorage
            case StorageType.VOLCENGINE_TOS:
                from extensions.storage.volcengine_tos_storage import VolcengineTosStorage
                return VolcengineTosStorage
            case StorageType.SUPABASE:
                from extensions.storage.supabase_storage import SupabaseStorage
                return SupabaseStorage
            case StorageType.CLICKZETTA_VOLUME:
                from extensions.storage.clickzetta_volume.clickzetta_volume_storage import (
                    ClickZettaVolumeConfig,
                    ClickZettaVolumeStorage,
                )

                def create_clickzetta_volume_storage():
                    # ClickZettaVolumeConfig将自动从环境变量中读取配置
                    # 如果未设置CLICKZETTA_VOLUME_*，则回退到CLICKZETTA_*配置
                    volume_config = ClickZettaVolumeConfig()
                    return ClickZettaVolumeStorage(volume_config)

                return create_clickzetta_volume_storage
            case _:
                raise ValueError(f"unsupported storage type {storage_type}")

    def save(self, filename: str, data: bytes) -> None:
        """
        保存文件数据到存储。

        Args:
            filename: 文件名
            data: 文件数据（字节）
        """
        self.storage_runner.save(filename, data)

    @overload
    def load(self, filename: str, /, *, stream: Literal[False] = False) -> bytes:
        """
        加载文件内容，以一次性方式返回文件内容。

        Args:
            filename: 文件名
            stream: 是否以流式方式加载，默认为False

        Returns:
            bytes: 文件内容
        """
        pass

    @overload
    def load(self, filename: str, /, *, stream: Literal[True]) -> Generator:
        """
        加载文件内容，以流式方式返回文件内容生成器。

        Args:
            filename: 文件名
            stream: 是否以流式方式加载，默认为True

        Returns:
            Generator: 文件内容生成器
        """
        pass

    def load(self, filename: str, /, *, stream: bool = False) -> Union[bytes, Generator]:
        """
        加载文件内容，根据stream参数决定是以一次性方式还是流式方式加载。

        Args:
            filename: 文件名
            stream: 是否以流式方式加载，默认为False

        Returns:
            Union[bytes, Generator]: 文件内容（字节或生成器）
        """
        if stream:
            return self.load_stream(filename)
        else:
            return self.load_once(filename)

    def load_once(self, filename: str) -> bytes:
        """
        一次性加载文件内容。

        Args:
            filename: 文件名

        Returns:
            bytes: 文件内容
        """
        return self.storage_runner.load_once(filename)

    def load_stream(self, filename: str) -> Generator:
        """
        流式加载文件内容，返回生成器。

        Args:
            filename: 文件名

        Returns:
            Generator: 文件内容生成器
        """
        return self.storage_runner.load_stream(filename)

    def download(self, filename: str, target_filepath: str) -> None:
        """
        下载文件到指定的本地路径。

        Args:
            filename: 文件名
            target_filepath: 目标文件路径
        """
        self.storage_runner.download(filename, target_filepath)

    def exists(self, filename: str) -> bool:
        """
        检查指定文件是否存在。

        Args:
            filename: 文件名

        Returns:
            bool: 文件是否存在
        """
        return self.storage_runner.exists(filename)

    def delete(self, filename: str) -> bool:
        """
        删除指定文件。

        Args:
            filename: 文件名

        Returns:
            bool: 删除是否成功
        """
        return self.storage_runner.delete(filename)

    def scan(self, path: str, files: bool = True, directories: bool = False) -> list[str]:
        """
        扫描指定路径下的文件和目录。

        Args:
            path: 扫描路径
            files: 是否包含文件，默认为True
            directories: 是否包含目录，默认为False

        Returns:
            list[str]: 文件或目录的列表
        """
        return self.storage_runner.scan(path, files=files, directories=directories)


storage = Storage()


def init_app(app: DifyApp) -> None:
    """
    初始化应用程序的存储功能。

    Args:
        app: DifyApp应用程序实例
    """
    storage.init_app(app)
