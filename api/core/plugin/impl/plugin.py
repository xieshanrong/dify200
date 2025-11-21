from collections.abc import Sequence

from core.plugin.entities.bundle import PluginBundleDependency
from core.plugin.entities.plugin import (
    MissingPluginDependency,
    PluginDeclaration,
    PluginEntity,
    PluginInstallation,
    PluginInstallationSource,
)
from core.plugin.entities.plugin_daemon import (
    PluginDecodeResponse,
    PluginInstallTask,
    PluginInstallTaskStartResponse,
    PluginListResponse,
)
from core.plugin.impl.base import BasePluginClient
from models.provider_ids import GenericProviderID

# 定义一个插件安装器类，继承自BasePluginClient
class PluginInstaller(BasePluginClient):
    # 根据标识符查找插件
    def fetch_plugin_by_identifier(self, tenant_id: str, identifier: str) -> bool:
        """
        根据插件标识符查找插件，返回是否存在该插件。

        参数:
        tenant_id: 租户ID
        identifier: 插件标识符

        返回:
        布尔值，表示插件是否存在
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/fetch/identifier",
            bool,
            params={"plugin_unique_identifier": identifier},
        )

    # 列出插件
    def list_plugins(self, tenant_id: str) -> list[PluginEntity]:
        """
        返回指定租户的所有插件列表。

        参数:
        tenant_id: 租户ID

        返回:
        包含PluginEntity对象的列表，表示所有插件
        """
        result = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/list",
            PluginListResponse,
            params={"page": 1, "page_size": 256, "response_type": "paged"},
        )
        return result.list

    # 带总数的插件列表
    def list_plugins_with_total(self, tenant_id: str, page: int, page_size: int) -> PluginListResponse:
        """
        返回指定租户的分页插件列表，并包含总数。

        参数:
        tenant_id: 租户ID
        page: 当前页码
        page_size: 每页大小

        返回:
        PluginListResponse对象，包含分页插件列表和总数
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/list",
            PluginListResponse,
            params={"page": page, "page_size": page_size, "response_type": "paged"},
        )

    # 上传插件包
    def upload_pkg(self, tenant_id: str, pkg: bytes, verify_signature: bool = False) -> PluginDecodeResponse:
        """
        上传插件包并返回插件唯一标识符。

        参数:
        tenant_id: 租户ID
        pkg: 插件包的二进制数据
        verify_signature: 是否验证签名，默认为False

        返回:
        PluginDecodeResponse对象，包含插件唯一标识符
        """
        body = {
            "dify_pkg": ("dify_pkg", pkg, "application/octet-stream"),
        }
        data = {
            "verify_signature": "true" if verify_signature else "false",
        }
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/managementinstall/upload/package",
            PluginDecodeResponse,
            files=body,
            data=data,
        )

    # 上传插件捆绑包
    def upload_bundle(self, tenant_id: str, bundle: bytes, verify_signature: bool = False) -> Sequence[PluginBundleDependency]:
        """
        上传插件捆绑包并返回依赖项列表。

        参数:
        tenant_id: 租户ID
        bundle: 插件捆绑包的二进制数据
        verify_signature: 是否验证签名，默认为False

        返回:
        包含PluginBundleDependency对象的序列，表示依赖项
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/upload/bundle",
            list[PluginBundleDependency],
            files={"dify_bundle": ("dify_bundle", bundle, "application/octet-stream")},
            data={"verify_signature": "true" if verify_signature else "false"},
        )

    # 根据标识符安装插件
    def install_from_identifiers(
        self,
        tenant_id: str,
        identifiers: Sequence[str],
        source: PluginInstallationSource,
        metas: list[dict],
    ) -> PluginInstallTaskStartResponse:
        """
        根据标识符列表安装插件，返回任务启动响应。

        参数:
        tenant_id: 租户ID
        identifiers: 插件标识符列表
        source: 插件安装源
        metas: 元数据列表

        返回:
        PluginInstallTaskStartResponse对象，表示任务已启动
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/identifiers",
            PluginInstallTaskStartResponse,
            data={
                "plugin_unique_identifiers": identifiers,
                "source": source,
                "metas": metas,
            },
            headers={"Content-Type": "application/json"},
        )

    # 获取插件安装任务列表
    def fetch_plugin_installation_tasks(self, tenant_id: str, page: int, page_size: int) -> Sequence[PluginInstallTask]:
        """
        获取指定租户的插件安装任务列表。

        参数:
        tenant_id: 租户ID
        page: 当前页码
        page_size: 每页大小

        返回:
        包含PluginInstallTask对象的序列，表示所有安装任务
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/install/tasks",
            list[PluginInstallTask],
            params={"page": page, "page_size": page_size},
        )

    # 获取插件安装任务详情
    def fetch_plugin_installation_task(self, tenant_id: str, task_id: str) -> PluginInstallTask:
        """
        获取指定插件安装任务的详细信息。

        参数:
        tenant_id: 租户ID
        task_id: 任务ID

        返回:
        PluginInstallTask对象，表示任务详情
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/install/tasks/{task_id}",
            PluginInstallTask,
        )

    # 删除插件安装任务
    def delete_plugin_installation_task(self, tenant_id: str, task_id: str) -> bool:
        """
        删除指定的插件安装任务。

        参数:
        tenant_id: 租户ID
        task_id: 任务ID

        返回:
        布尔值，表示删除是否成功
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/tasks/{task_id}/delete",
            bool,
        )

    # 删除所有插件安装任务项
    def delete_all_plugin_installation_task_items(self, tenant_id: str) -> bool:
        """
        删除指定租户的所有插件安装任务项。

        参数:
        tenant_id: 租户ID

        返回:
        布尔值，表示删除是否成功
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/tasks/delete_all",
            bool,
        )

    # 删除插件安装任务项
    def delete_plugin_installation_task_item(self, tenant_id: str, task_id: str, identifier: str) -> bool:
        """
        删除指定插件安装任务项。

        参数:
        tenant_id: 租户ID
        task_id: 任务ID
        identifier: 插件标识符

        返回:
        布尔值，表示删除是否成功
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/tasks/{task_id}/delete/{identifier}",
            bool,
        )

    # 获取插件清单
    def fetch_plugin_manifest(self, tenant_id: str, plugin_unique_identifier: str) -> PluginDeclaration:
        """
        获取指定插件的清单信息。

        参数:
        tenant_id: 租户ID
        plugin_unique_identifier: 插件唯一标识符

        返回:
        PluginDeclaration对象，表示插件清单
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/fetch/manifest",
            PluginDeclaration,
            params={"plugin_unique_identifier": plugin_unique_identifier},
        )

    # 解码插件标识符
    def decode_plugin_from_identifier(self, tenant_id: str, plugin_unique_identifier: str) -> PluginDecodeResponse:
        """
        根据标识符解码插件。

        参数:
        tenant_id: 租户ID
        plugin_unique_identifier: 插件唯一标识符

        返回:
        PluginDecodeResponse对象，包含解码后的插件信息
        """
        return self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/decode/from_identifier",
            PluginDecodeResponse,
            data={"plugin_unique_identifier": plugin_unique_identifier},
            headers={"Content-Type": "application/json"},
        )

    # 根据ID获取插件安装信息
    def fetch_plugin_installation_by_ids(self, tenant_id: str, plugin_ids: Sequence[str]) -> Sequence[PluginInstallation]:
        """
        根据插件ID获取安装信息。

        参数:
        tenant_id: 租户ID
        plugin_ids: 插件ID列表

        返回:
        包含PluginInstallation对象的序列，表示安装信息
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/installation/fetch/batch",
            list[PluginInstallation],
            data={"plugin_ids": plugin_ids},
            headers={"Content-Type": "application/json"},
        )

    # 获取缺失的依赖项
    def fetch_missing_dependencies(self, tenant_id: str, plugin_unique_identifiers: list[str]) -> list[MissingPluginDependency]:
        """
        获取指定插件标识符的缺失依赖项。

        参数:
        tenant_id: 租户ID
        plugin_unique_identifiers: 插件唯一标识符列表

        返回:
        包含MissingPluginDependency对象的列表，表示缺失的依赖项
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/installation/missing",
            list[MissingPluginDependency],
            data={"plugin_unique_identifiers": plugin_unique_identifiers},
            headers={"Content-Type": "application/json"},
        )

    # 卸载插件
    def uninstall(self, tenant_id: str, plugin_installation_id: str) -> bool:
        """
        卸载指定的插件。

        参数:
        tenant_id: 租户ID
        plugin_installation_id: 插件安装ID

        返回:
        布尔值，表示卸载是否成功
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/uninstall",
            bool,
            data={
                "plugin_installation_id": plugin_installation_id,
            },
            headers={"Content-Type": "application/json"},
        )

    # 升级插件
    def upgrade_plugin(
        self,
        tenant_id: str,
        original_plugin_unique_identifier: str,
        new_plugin_unique_identifier: str,
        source: PluginInstallationSource,
        meta: dict,
    ) -> PluginInstallTaskStartResponse:
        """
        升级指定的插件，返回任务启动响应。

        参数:
        tenant_id: 租户ID
        original_plugin_unique_identifier: 原插件唯一标识符
        new_plugin_unique_identifier: 新插件唯一标识符
        source: 插件安装源
        meta: 元数据

        返回:
        PluginInstallTaskStartResponse对象，表示升级任务已启动
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/install/upgrade",
            PluginInstallTaskStartResponse,
            data={
                "original_plugin_unique_identifier": original_plugin_unique_identifier,
                "new_plugin_unique_identifier": new_plugin_unique_identifier,
                "source": source,
                "meta": meta,
            },
            headers={"Content-Type": "application/json"},
        )

    # 检查工具的存在性
    def check_tools_existence(self, tenant_id: str, provider_ids: Sequence[GenericProviderID]) -> Sequence[bool]:
        """
        检查指定提供者ID的工具是否存在。

        参数:
        tenant_id: 租户ID
        provider_ids: 提供者ID列表

        返回:
        包含布尔值的序列，表示每个工具是否存在
        """
        return self._request_with_plugin_daemon_response(
            "POST",
            f"plugin/{tenant_id}/management/tools/check_existence",
            list[bool],
            data={
                "provider_ids": [
                    {
                        "plugin_id": provider_id.plugin_id,
                        "provider_name": provider_id.provider_name,
                    }
                    for provider_id in provider_ids
                ]
            },
            headers={"Content-Type": "application/json"},
        )
