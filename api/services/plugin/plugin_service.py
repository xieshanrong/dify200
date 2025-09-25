import logging
from collections.abc import Mapping, Sequence
from mimetypes import guess_type
from typing import Optional

from pydantic import BaseModel

from configs import dify_config
from core.helper import marketplace
from core.helper.download import download_with_size_limit
from core.helper.marketplace import download_plugin_pkg
from core.plugin.entities.bundle import PluginBundleDependency
from core.plugin.entities.plugin import (
    PluginDeclaration,
    PluginEntity,
    PluginInstallation,
    PluginInstallationSource,
)
from core.plugin.entities.plugin_daemon import (
    PluginDecodeResponse,
    PluginInstallTask,
    PluginListResponse,
    PluginVerification,
)
from core.plugin.impl.asset import PluginAssetManager
from core.plugin.impl.debugging import PluginDebuggingClient
from core.plugin.impl.plugin import PluginInstaller
from extensions.ext_redis import redis_client
from models.provider_ids import GenericProviderID
from services.errors.plugin import PluginInstallationForbiddenError
from services.feature_service import FeatureService, PluginInstallationScope

logger = logging.getLogger(__name__)


class PluginService:
    """
    插件服务类，提供插件管理相关功能
    """
    class LatestPluginCache(BaseModel):
        """
        最新插件缓存模型类
        """
        plugin_id: str
        version: str
        unique_identifier: str
        status: str
        deprecated_reason: str
        alternative_plugin_id: str

    REDIS_KEY_PREFIX = "plugin_service:latest_plugin:"
    REDIS_TTL = 60 * 5  # 5 minutes

    @staticmethod
    def fetch_latest_plugin_version(plugin_ids: Sequence[str]) -> Mapping[str, Optional[LatestPluginCache]]:
        """
        获取插件的最新版本信息

        Args:
            plugin_ids (Sequence[str]): 插件ID列表

        Returns:
            Mapping[str, Optional[LatestPluginCache]]: 插件ID到最新版本信息的映射
        """
        result: dict[str, Optional[PluginService.LatestPluginCache]] = {}

        try:
            cache_not_exists = []

            # 首先尝试从Redis缓存中获取
            for plugin_id in plugin_ids:
                cached_data = redis_client.get(f"{PluginService.REDIS_KEY_PREFIX}{plugin_id}")
                if cached_data:
                    result[plugin_id] = PluginService.LatestPluginCache.model_validate_json(cached_data)
                else:
                    cache_not_exists.append(plugin_id)

            if cache_not_exists:
                manifests = {
                    manifest.plugin_id: manifest
                    for manifest in marketplace.batch_fetch_plugin_manifests(cache_not_exists)
                }

                for plugin_id, manifest in manifests.items():
                    latest_plugin = PluginService.LatestPluginCache(
                        plugin_id=plugin_id,
                        version=manifest.latest_version,
                        unique_identifier=manifest.latest_package_identifier,
                        status=manifest.status,
                        deprecated_reason=manifest.deprecated_reason,
                        alternative_plugin_id=manifest.alternative_plugin_id,
                    )

                    # 存储到Redis缓存中
                    redis_client.setex(
                        f"{PluginService.REDIS_KEY_PREFIX}{plugin_id}",
                        PluginService.REDIS_TTL,
                        latest_plugin.model_dump_json(),
                    )

                    result[plugin_id] = latest_plugin

                    # 从cache_not_exists列表中移除已处理的插件ID
                    cache_not_exists.remove(plugin_id)

                for plugin_id in cache_not_exists:
                    result[plugin_id] = None

            return result
        except Exception:
            logger.exception("failed to fetch latest plugin version")
            return result

    @staticmethod
    def _check_marketplace_only_permission():
        """
        检查是否启用了仅市场权限限制
        如果启用了该限制，则抛出插件安装被禁止的异常
        """
        features = FeatureService.get_system_features()
        if features.plugin_installation_permission.restrict_to_marketplace_only:
            raise PluginInstallationForbiddenError("Plugin installation is restricted to marketplace only")

    @staticmethod
    def _check_plugin_installation_scope(plugin_verification: Optional[PluginVerification]):
        """
        检查插件安装范围权限

        Args:
            plugin_verification (Optional[PluginVerification]): 插件验证信息
        """
        features = FeatureService.get_system_features()

        match features.plugin_installation_permission.plugin_installation_scope:
            case PluginInstallationScope.OFFICIAL_ONLY:
                if (
                    plugin_verification is None
                    or plugin_verification.authorized_category != PluginVerification.AuthorizedCategory.Langgenius
                ):
                    raise PluginInstallationForbiddenError("Plugin installation is restricted to official only")
            case PluginInstallationScope.OFFICIAL_AND_SPECIFIC_PARTNERS:
                if plugin_verification is None or plugin_verification.authorized_category not in [
                    PluginVerification.AuthorizedCategory.Langgenius,
                    PluginVerification.AuthorizedCategory.Partner,
                ]:
                    raise PluginInstallationForbiddenError(
                        "Plugin installation is restricted to official and specific partners"
                    )
            case PluginInstallationScope.NONE:
                raise PluginInstallationForbiddenError("Installing plugins is not allowed")
            case PluginInstallationScope.ALL:
                pass

    @staticmethod
    def get_debugging_key(tenant_id: str) -> str:
        """
        获取租户的调试密钥

        Args:
            tenant_id (str): 租户ID

        Returns:
            str: 调试密钥
        """
        manager = PluginDebuggingClient()
        return manager.get_debugging_key(tenant_id)

    @staticmethod
    def list_latest_versions(plugin_ids: Sequence[str]) -> Mapping[str, Optional[LatestPluginCache]]:
        """
        列出插件的最新版本信息

        Args:
            plugin_ids (Sequence[str]): 插件ID列表

        Returns:
            Mapping[str, Optional[LatestPluginCache]]: 插件ID到最新版本信息的映射
        """
        return PluginService.fetch_latest_plugin_version(plugin_ids)

    @staticmethod
    def list(tenant_id: str) -> list[PluginEntity]:
        """
        列出租户的所有插件

        Args:
            tenant_id (str): 租户ID

        Returns:
            list[PluginEntity]: 插件实体列表
        """
        manager = PluginInstaller()
        plugins = manager.list_plugins(tenant_id)
        return plugins

    @staticmethod
    def list_with_total(tenant_id: str, page: int, page_size: int) -> PluginListResponse:
        """
        列出租户的所有插件（带总数信息）

        Args:
            tenant_id (str): 租户ID
            page (int): 页码
            page_size (int): 每页大小

        Returns:
            PluginListResponse: 包含插件列表和总数的响应对象
        """
        manager = PluginInstaller()
        plugins = manager.list_plugins_with_total(tenant_id, page, page_size)
        return plugins

    @staticmethod
    def list_installations_from_ids(tenant_id: str, ids: Sequence[str]) -> Sequence[PluginInstallation]:
        """
        根据ID列表获取插件安装信息

        Args:
            tenant_id (str): 租户ID
            ids (Sequence[str]): 插件安装ID列表

        Returns:
            Sequence[PluginInstallation]: 插件安装信息列表
        """
        manager = PluginInstaller()
        return manager.fetch_plugin_installation_by_ids(tenant_id, ids)

    @staticmethod
    def get_asset(tenant_id: str, asset_file: str) -> tuple[bytes, str]:
        """
        获取插件的资源文件

        Args:
            tenant_id (str): 租户ID
            asset_file (str): 资源文件路径

        Returns:
            tuple[bytes, str]: 资源文件内容和MIME类型
        """
        manager = PluginAssetManager()
        # 推测MIME类型
        mime_type, _ = guess_type(asset_file)
        return manager.fetch_asset(tenant_id, asset_file), mime_type or "application/octet-stream"

    @staticmethod
    def check_plugin_unique_identifier(tenant_id: str, plugin_unique_identifier: str) -> bool:
        """
        检查插件唯一标识符是否已被其他租户安装

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifier (str): 插件唯一标识符

        Returns:
            bool: 如果插件唯一标识符已被其他租户安装则返回True，否则返回False
        """
        manager = PluginInstaller()
        return manager.fetch_plugin_by_identifier(tenant_id, plugin_unique_identifier)

    @staticmethod
    def fetch_plugin_manifest(tenant_id: str, plugin_unique_identifier: str) -> PluginDeclaration:
        """
        获取插件声明信息

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifier (str): 插件唯一标识符

        Returns:
            PluginDeclaration: 插件声明信息
        """
        manager = PluginInstaller()
        return manager.fetch_plugin_manifest(tenant_id, plugin_unique_identifier)

    @staticmethod
    def is_plugin_verified(tenant_id: str, plugin_unique_identifier: str) -> bool:
        """
        检查插件是否已验证

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifier (str): 插件唯一标识符

        Returns:
            bool: 如果插件已验证则返回True，否则返回False
        """
        manager = PluginInstaller()
        try:
            return manager.fetch_plugin_manifest(tenant_id, plugin_unique_identifier).verified
        except Exception:
            return False

    @staticmethod
    def fetch_install_tasks(tenant_id: str, page: int, page_size: int) -> Sequence[PluginInstallTask]:
        """
        获取插件安装任务列表

        Args:
            tenant_id (str): 租户ID
            page (int): 页码
            page_size (int): 每页大小

        Returns:
            Sequence[PluginInstallTask]: 插件安装任务列表
        """
        manager = PluginInstaller()
        return manager.fetch_plugin_installation_tasks(tenant_id, page, page_size)

    @staticmethod
    def fetch_install_task(tenant_id: str, task_id: str) -> PluginInstallTask:
        """
        获取插件安装任务

        Args:
            tenant_id (str): 租户ID
            task_id (str): 任务ID

        Returns:
            PluginInstallTask: 插件安装任务
        """
        manager = PluginInstaller()
        return manager.fetch_plugin_installation_task(tenant_id, task_id)

    @staticmethod
    def delete_install_task(tenant_id: str, task_id: str) -> bool:
        """
        删除插件安装任务

        Args:
            tenant_id (str): 租户ID
            task_id (str): 任务ID

        Returns:
            bool: 删除成功返回True，否则返回False
        """
        manager = PluginInstaller()
        return manager.delete_plugin_installation_task(tenant_id, task_id)

    @staticmethod
    def delete_all_install_task_items(
        tenant_id: str,
    ) -> bool:
        """
        删除所有插件安装任务项

        Args:
            tenant_id (str): 租户ID

        Returns:
            bool: 删除成功返回True，否则返回False
        """
        manager = PluginInstaller()
        return manager.delete_all_plugin_installation_task_items(tenant_id)

    @staticmethod
    def delete_install_task_item(tenant_id: str, task_id: str, identifier: str) -> bool:
        """
        删除插件安装任务项

        Args:
            tenant_id (str): 租户ID
            task_id (str): 任务ID
            identifier (str): 任务项标识符

        Returns:
            bool: 删除成功返回True，否则返回False
        """
        manager = PluginInstaller()
        return manager.delete_plugin_installation_task_item(tenant_id, task_id, identifier)

    @staticmethod
    def upgrade_plugin_with_marketplace(
        tenant_id: str, original_plugin_unique_identifier: str, new_plugin_unique_identifier: str
    ):
        """
        通过市场升级插件

        Args:
            tenant_id (str): 租户ID
            original_plugin_unique_identifier (str): 原始插件唯一标识符
            new_plugin_unique_identifier (str): 新插件唯一标识符
        """
        if not dify_config.MARKETPLACE_ENABLED:
            raise ValueError("marketplace is not enabled")

        if original_plugin_unique_identifier == new_plugin_unique_identifier:
            raise ValueError("you should not upgrade plugin with the same plugin")

        # 检查插件包是否已下载
        manager = PluginInstaller()

        features = FeatureService.get_system_features()

        try:
            manager.fetch_plugin_manifest(tenant_id, new_plugin_unique_identifier)
            # 已下载，跳过并记录安装事件
            marketplace.record_install_plugin_event(new_plugin_unique_identifier)
        except Exception:
            # 插件未安装，下载并上传包
            pkg = download_plugin_pkg(new_plugin_unique_identifier)
            response = manager.upload_pkg(
                tenant_id,
                pkg,
                verify_signature=features.plugin_installation_permission.restrict_to_marketplace_only,
            )

            # 检查插件是否可安装
            PluginService._check_plugin_installation_scope(response.verification)

        return manager.upgrade_plugin(
            tenant_id,
            original_plugin_unique_identifier,
            new_plugin_unique_identifier,
            PluginInstallationSource.Marketplace,
            {
                "plugin_unique_identifier": new_plugin_unique_identifier,
            },
        )

    @staticmethod
    def upgrade_plugin_with_github(
        tenant_id: str,
        original_plugin_unique_identifier: str,
        new_plugin_unique_identifier: str,
        repo: str,
        version: str,
        package: str,
    ):
        """
        通过GitHub升级插件

        Args:
            tenant_id (str): 租户ID
            original_plugin_unique_identifier (str): 原始插件唯一标识符
            new_plugin_unique_identifier (str): 新插件唯一标识符
            repo (str): GitHub仓库名
            version (str): 版本号
            package (str): 包名

        Returns:
            升级结果
        """
        PluginService._check_marketplace_only_permission()
        manager = PluginInstaller()
        return manager.upgrade_plugin(
            tenant_id,
            original_plugin_unique_identifier,
            new_plugin_unique_identifier,
            PluginInstallationSource.Github,
            {
                "repo": repo,
                "version": version,
                "package": package,
            },
        )

    @staticmethod
    def upload_pkg(tenant_id: str, pkg: bytes, verify_signature: bool = False) -> PluginDecodeResponse:
        """
        上传插件包文件

        Args:
            tenant_id (str): 租户ID
            pkg (bytes): 插件包数据
            verify_signature (bool): 是否验证签名，默认为False

        Returns:
            PluginDecodeResponse: 插件解码响应，包含插件唯一标识符
        """
        PluginService._check_marketplace_only_permission()
        manager = PluginInstaller()
        features = FeatureService.get_system_features()
        response = manager.upload_pkg(
            tenant_id,
            pkg,
            verify_signature=features.plugin_installation_permission.restrict_to_marketplace_only,
        )
        return response

    @staticmethod
    def upload_pkg_from_github(
        tenant_id: str, repo: str, version: str, package: str, verify_signature: bool = False
    ) -> PluginDecodeResponse:
        """
        从GitHub发布包安装插件

        Args:
            tenant_id (str): 租户ID
            repo (str): GitHub仓库名
            version (str): 版本号
            package (str): 包名
            verify_signature (bool): 是否验证签名，默认为False

        Returns:
            PluginDecodeResponse: 插件解码响应，包含插件唯一标识符
        """
        PluginService._check_marketplace_only_permission()
        pkg = download_with_size_limit(
            f"https://github.com/{repo}/releases/download/{version}/{package}", dify_config.PLUGIN_MAX_PACKAGE_SIZE
        )
        features = FeatureService.get_system_features()

        manager = PluginInstaller()
        response = manager.upload_pkg(
            tenant_id,
            pkg,
            verify_signature=features.plugin_installation_permission.restrict_to_marketplace_only,
        )
        return response

    @staticmethod
    def upload_bundle(
        tenant_id: str, bundle: bytes, verify_signature: bool = False
    ) -> Sequence[PluginBundleDependency]:
        """
        上传插件包并返回依赖项

        Args:
            tenant_id (str): 租户ID
            bundle (bytes): 插件包数据
            verify_signature (bool): 是否验证签名，默认为False

        Returns:
            Sequence[PluginBundleDependency]: 插件包依赖项列表
        """
        manager = PluginInstaller()
        PluginService._check_marketplace_only_permission()
        return manager.upload_bundle(tenant_id, bundle, verify_signature)

    @staticmethod
    def install_from_local_pkg(tenant_id: str, plugin_unique_identifiers: Sequence[str]):
        """
        从本地包安装插件

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifiers (Sequence[str]): 插件唯一标识符列表
        """
        PluginService._check_marketplace_only_permission()

        manager = PluginInstaller()

        return manager.install_from_identifiers(
            tenant_id,
            plugin_unique_identifiers,
            PluginInstallationSource.Package,
            [{}],
        )

    @staticmethod
    def install_from_github(tenant_id: str, plugin_unique_identifier: str, repo: str, version: str, package: str):
        """
        从GitHub发布包安装插件

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifier (str): 插件唯一标识符
            repo (str): GitHub仓库名
            version (str): 版本号
            package (str): 包名

        Returns:
            安装结果，包含插件唯一标识符
        """
        PluginService._check_marketplace_only_permission()

        manager = PluginInstaller()
        return manager.install_from_identifiers(
            tenant_id,
            [plugin_unique_identifier],
            PluginInstallationSource.Github,
            [
                {
                    "repo": repo,
                    "version": version,
                    "package": package,
                }
            ],
        )

    @staticmethod
    def fetch_marketplace_pkg(tenant_id: str, plugin_unique_identifier: str) -> PluginDeclaration:
        """
        获取市场包

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifier (str): 插件唯一标识符

        Returns:
            PluginDeclaration: 插件声明信息
        """
        if not dify_config.MARKETPLACE_ENABLED:
            raise ValueError("marketplace is not enabled")

        features = FeatureService.get_system_features()

        manager = PluginInstaller()
        try:
            declaration = manager.fetch_plugin_manifest(tenant_id, plugin_unique_identifier)
        except Exception:
            pkg = download_plugin_pkg(plugin_unique_identifier)
            response = manager.upload_pkg(
                tenant_id,
                pkg,
                verify_signature=features.plugin_installation_permission.restrict_to_marketplace_only,
            )
            # 检查插件是否可安装
            PluginService._check_plugin_installation_scope(response.verification)
            declaration = response.manifest

        return declaration

    @staticmethod
    def install_from_marketplace_pkg(tenant_id: str, plugin_unique_identifiers: Sequence[str]):
        """
        从市场包文件安装插件

        Args:
            tenant_id (str): 租户ID
            plugin_unique_identifiers (Sequence[str]): 插件唯一标识符列表

        Returns:
            安装任务ID
        """
        if not dify_config.MARKETPLACE_ENABLED:
            raise ValueError("marketplace is not enabled")

        manager = PluginInstaller()

        # 收集实际的插件唯一标识符
        actual_plugin_unique_identifiers = []
        metas = []
        features = FeatureService.get_system_features()

        # 检查是否已下载
        for plugin_unique_identifier in plugin_unique_identifiers:
            try:
                manager.fetch_plugin_manifest(tenant_id, plugin_unique_identifier)
                plugin_decode_response = manager.decode_plugin_from_identifier(tenant_id, plugin_unique_identifier)
                # 检查插件是否可安装
                PluginService._check_plugin_installation_scope(plugin_decode_response.verification)
                # 已下载，跳过
                actual_plugin_unique_identifiers.append(plugin_unique_identifier)
                metas.append({"plugin_unique_identifier": plugin_unique_identifier})
            except Exception:
                # 插件未安装，下载并上传包
                pkg = download_plugin_pkg(plugin_unique_identifier)
                response = manager.upload_pkg(
                    tenant_id,
                    pkg,
                    verify_signature=features.plugin_installation_permission.restrict_to_marketplace_only,
                )
                # 检查插件是否可安装
                PluginService._check_plugin_installation_scope(response.verification)
                # 使用响应中的插件唯一标识符
                actual_plugin_unique_identifiers.append(response.unique_identifier)
                metas.append({"plugin_unique_identifier": response.unique_identifier})

        return manager.install_from_identifiers(
            tenant_id,
            actual_plugin_unique_identifiers,
            PluginInstallationSource.Marketplace,
            metas,
        )

    @staticmethod
    def uninstall(tenant_id: str, plugin_installation_id: str) -> bool:
        """
        卸载插件

        Args:
            tenant_id (str): 租户ID
            plugin_installation_id (str): 插件安装ID

        Returns:
            bool: 卸载成功返回True，否则返回False
        """
        manager = PluginInstaller()
        return manager.uninstall(tenant_id, plugin_installation_id)

    @staticmethod
    def check_tools_existence(tenant_id: str, provider_ids: Sequence[GenericProviderID]) -> Sequence[bool]:
        """
        检查工具是否存在

        Args:
            tenant_id (str): 租户ID
            provider_ids (Sequence[GenericProviderID]): 提供者ID列表

        Returns:
            Sequence[bool]: 工具存在性检查结果列表
        """
        manager = PluginInstaller()
        return manager.check_tools_existence(tenant_id, provider_ids)
