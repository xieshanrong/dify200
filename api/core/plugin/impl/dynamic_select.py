from collections.abc import Mapping
from typing import Any

# 导入PluginDynamicSelectOptionsResponse类，用于处理插件动态选择选项的响应
from core.plugin.entities.plugin_daemon import PluginDynamicSelectOptionsResponse
# 导入BasePluginClient基类，作为DynamicSelectClient的父类
from core.plugin.impl.base import BasePluginClient


# 导入GenericProviderID类，用于处理提供者ID


class DynamicSelectClient(BasePluginClient):
    """用于与插件动态选择服务进行交互的客户端类"""

    def fetch_dynamic_select_options(
        self,
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        action: str,
        credentials: Mapping[str, Any],
        parameter: str,
    ) -> PluginDynamicSelectOptionsResponse:
        """获取指定插件参数的动态选择选项

        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            plugin_id: 插件ID
            provider: 提供者名称
            action: 操作名称
            credentials: 认证信息，类型为字典
            parameter: 要获取选项的参数名称

        Returns:
            PluginDynamicSelectOptionsResponse: 包含动态选择选项的响应对象
        """
        # 发送POST请求以获取动态选择选项
        response = self._request_with_plugin_daemon_response_stream(
            "POST",
            # 构建请求的URL路径，包含tenant_id
            f"plugin/{tenant_id}/dispatch/dynamic_select/fetch_parameter_options",
            # 指定响应的类型为PluginDynamicSelectOptionsResponse
            PluginDynamicSelectOptionsResponse,
            # 请求的数据，包含用户ID和详细数据
            data={
                "user_id": user_id,
                "data": {
                    # 将provider转换为提供者名称
                    "provider": GenericProviderID(provider).provider_name,
                    "credentials": credentials,
                    "provider_action": action,
                    "parameter": parameter,
                },
            },
            # 请求头，包含插件ID和内容类型
            headers={
                "X-Plugin-ID": plugin_id,
                "Content-Type": "application/json",
            },
        )

        # 遍历响应结果，逐个处理选项
        for options in response:
            return options

        # 如果没有选项返回，抛出异常
        raise ValueError(f"Plugin service returned no options for parameter '{parameter}' in provider '{provider}'")
