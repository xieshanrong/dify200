from collections.abc import Generator
from typing import Any, Optional

from core.plugin.impl.tool import PluginToolManager
from core.plugin.utils.converter import convert_parameters_to_plugin_format
from core.tools.__base.tool import Tool
from core.tools.__base.tool_runtime import ToolRuntime
from core.tools.entities.tool_entities import ToolEntity, ToolInvokeMessage, ToolParameter, ToolProviderType


class PluginTool(Tool):
    """
    PluginTool类用于管理和执行插件工具

    该类继承自Tool基类，提供了插件工具的特定实现，包括工具调用、参数处理等功能。
    主要负责与插件管理器交互，处理插件工具的调用和参数管理。

    Attributes:
        tenant_id (str): 租户ID，用于标识插件所属的租户
        icon (str): 工具图标，用于前端展示
        plugin_unique_identifier (str): 插件唯一标识符，用于唯一标识一个插件实例
        runtime_parameters (Optional[list[ToolParameter]]): 运行时参数列表，缓存从插件获取的参数信息
    """

    def __init__(
        self, entity: ToolEntity, runtime: ToolRuntime, tenant_id: str, icon: str, plugin_unique_identifier: str
    ):
        """
        初始化PluginTool实例

        Args:
            entity (ToolEntity): 工具实体对象，包含工具的基本信息和参数定义
            runtime (ToolRuntime): 工具运行时对象，包含工具运行时的凭证和配置信息
            tenant_id (str): 租户ID，用于标识插件所属的租户
            icon (str): 工具图标，用于前端展示
            plugin_unique_identifier (str): 插件唯一标识符，用于唯一标识一个插件实例
        """
        super().__init__(entity, runtime)
        self.tenant_id = tenant_id
        self.icon = icon
        self.plugin_unique_identifier = plugin_unique_identifier
        self.runtime_parameters: Optional[list[ToolParameter]] = None

    def tool_provider_type(self) -> ToolProviderType:
        """
        获取工具提供者类型

        Returns:
            ToolProviderType: 工具提供者类型，对于插件工具返回PLUGIN类型
        """
        return ToolProviderType.PLUGIN

    def _invoke(
        self,
        user_id: str,
        tool_parameters: dict[str, Any],
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        调用插件工具

        通过PluginToolManager管理器执行插件工具调用，首先将参数转换为插件格式，
        然后使用管理器执行实际的工具调用操作。

        Args:
            user_id (str): 用户ID，标识调用工具的用户
            tool_parameters (dict[str, Any]): 工具参数字典，包含调用工具所需的各种参数
            conversation_id (Optional[str]): 对话ID，可选，用于标识当前对话上下文
            app_id (Optional[str]): 应用ID，可选，用于标识当前应用
            message_id (Optional[str]): 消息ID，可选，用于标识当前消息

        Returns:
            Generator[ToolInvokeMessage, None, None]: 工具调用消息生成器，用于逐个返回工具执行结果
        """
        # 创建插件工具管理器实例
        manager = PluginToolManager()

        # 将工具参数转换为插件格式
        tool_parameters = convert_parameters_to_plugin_format(tool_parameters)

        # 通过管理器调用插件工具并返回结果
        yield from manager.invoke(
            tenant_id=self.tenant_id,
            user_id=user_id,
            tool_provider=self.entity.identity.provider,
            tool_name=self.entity.identity.name,
            credentials=self.runtime.credentials,
            credential_type=self.runtime.credential_type,
            tool_parameters=tool_parameters,
            conversation_id=conversation_id,
            app_id=app_id,
            message_id=message_id,
        )

    def fork_tool_runtime(self, runtime: ToolRuntime) -> "PluginTool":
        """
        复制工具运行时，创建一个新的PluginTool实例

        当需要在不同的运行时环境中使用相同工具时，通过此方法创建一个新的实例，
        保持工具的基本属性不变，但使用新的运行时配置。

        Args:
            runtime (ToolRuntime): 新的工具运行时对象，包含新的凭证和配置信息

        Returns:
            PluginTool: 新的PluginTool实例，具有相同的工具实体和基本属性，但使用新的运行时
        """
        return PluginTool(
            entity=self.entity,
            runtime=runtime,
            tenant_id=self.tenant_id,
            icon=self.icon,
            plugin_unique_identifier=self.plugin_unique_identifier,
        )

    def get_runtime_parameters(
        self,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[ToolParameter]:
        """
        获取运行时参数

        如果工具没有运行时参数，则直接返回实体参数；
        如果已缓存运行时参数，则直接返回缓存；
        否则通过PluginToolManager获取并缓存运行时参数。

        Args:
            conversation_id (Optional[str]): 对话ID，可选，用于标识当前对话上下文
            app_id (Optional[str]): 应用ID，可选，用于标识当前应用
            message_id (Optional[str]): 消息ID，可选，用于标识当前消息

        Returns:
            list[ToolParameter]: 工具参数列表，包含工具运行所需的所有参数定义
        """
        # 如果工具没有运行时参数，直接返回实体参数
        if not self.entity.has_runtime_parameters:
            return self.entity.parameters

        # 如果已缓存运行时参数，直接返回缓存结果
        if self.runtime_parameters is not None:
            return self.runtime_parameters

        # 创建插件工具管理器实例
        manager = PluginToolManager()

        # 通过管理器获取运行时参数并缓存
        self.runtime_parameters = manager.get_runtime_parameters(
            tenant_id=self.tenant_id,
            user_id="",
            provider=self.entity.identity.provider,
            tool=self.entity.identity.name,
            credentials=self.runtime.credentials,
            conversation_id=conversation_id,
            app_id=app_id,
            message_id=message_id,
        )

        return self.runtime_parameters
