from typing import Any  # 导入类型提示模块

# 导入插件工具管理器
from core.plugin.impl.tool import PluginToolManager
# 导入工具运行时基类
from core.tools.__base.tool_runtime import ToolRuntime
# 导入内置工具提供者控制器
from core.tools.builtin_tool.provider import BuiltinToolProviderController
# 导入工具提供者实体和类型
from core.tools.entities.tool_entities import ToolProviderEntityWithPlugin, ToolProviderType
# 导入工具提供者凭证验证错误
from core.tools.errors import ToolProviderCredentialValidationError
# 导入插件工具类
from core.tools.plugin_tool.tool import PluginTool

class PluginToolProviderController(BuiltinToolProviderController):
    """插件工具提供者控制器，继承自内置工具提供者控制器"""

    # 实体属性，类型为ToolProviderEntityWithPlugin
    entity: ToolProviderEntityWithPlugin
    # 租户ID，字符串类型
    tenant_id: str
    # 插件ID，字符串类型
    plugin_id: str
    # 插件唯一标识符，字符串类型
    plugin_unique_identifier: str

    def __init__(
        self, entity: ToolProviderEntityWithPlugin, plugin_id: str, plugin_unique_identifier: str, tenant_id: str
    ):
        """初始化插件工具提供者控制器

        Args:
            entity: 工具提供者实体
            plugin_id: 插件ID
            plugin_unique_identifier: 插件唯一标识符
            tenant_id: 租户ID
        """
        self.entity = entity
        self.tenant_id = tenant_id
        self.plugin_id = plugin_id
        self.plugin_unique_identifier = plugin_unique_identifier

    @property
    def provider_type(self) -> ToolProviderType:
        """返回提供者类型

        Returns:
            ToolProviderType: 提供者类型，始终返回PLUGIN
        """
        return ToolProviderType.PLUGIN

    def _validate_credentials(self, user_id: str, credentials: dict[str, Any]):
        """验证提供者的凭证

        Args:
            user_id: 用户ID
            credentials: 凭证字典
        """
        manager = PluginToolManager()
        if not manager.validate_provider_credentials(
            tenant_id=self.tenant_id,
            user_id=user_id,
            provider=self.entity.identity.name,
            credentials=credentials,
        ):
            raise ToolProviderCredentialValidationError("Invalid credentials")

    def get_tool(self, tool_name: str) -> PluginTool:  # type: ignore
        """根据工具名称获取指定工具

        Args:
            tool_name: 工具名称

        Returns:
            PluginTool: 包含工具实体的PluginTool对象

        Raises:
            ValueError: 如果未找到指定工具
        """
        tool_entity = next(
            (tool_entity for tool_entity in self.entity.tools if tool_entity.identity.name == tool_name), None
        )

        if not tool_entity:
            raise ValueError(f"Tool with name {tool_name} not found")

        return PluginTool(
            entity=tool_entity,
            runtime=ToolRuntime(tenant_id=self.tenant_id),
            tenant_id=self.tenant_id,
            icon=self.entity.identity.icon,
            plugin_unique_identifier=self.plugin_unique_identifier,
        )

    def get_tools(self) -> list[PluginTool]:  # type: ignore
        """获取所有工具列表

        Returns:
            list[PluginTool]: 包含所有工具的PluginTool对象列表
        """
        return [
            PluginTool(
                entity=tool_entity,
                runtime=ToolRuntime(tenant_id=self.tenant_id),
                tenant_id=self.tenant_id,
                icon=self.entity.identity.icon,
                plugin_unique_identifier=self.plugin_unique_identifier,
            )
            for tool_entity in self.entity.tools
        ]
