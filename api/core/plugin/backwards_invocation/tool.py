from collections.abc import Generator
from typing import Any, Optional

from core.callback_handler.workflow_tool_callback_handler import DifyWorkflowCallbackHandler
from core.plugin.backwards_invocation.base import BaseBackwardsInvocation
from core.tools.entities.tool_entities import ToolInvokeMessage, ToolProviderType
from core.tools.tool_engine import ToolEngine
from core.tools.tool_manager import ToolManager
from core.tools.utils.message_transformer import ToolFileMessageTransformer

class PluginToolBackwardsInvocation(BaseBackwardsInvocation):
    """反向调用插件工具的基类"""

    @classmethod
    def invoke_tool(
        cls,
        tenant_id: str,
        user_id: str,
        tool_type: ToolProviderType,
        provider: str,
        tool_name: str,
        tool_parameters: dict[str, Any],
        credential_id: Optional[str] = None,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """执行工具调用的方法

        Args:
            tenant_id: 租户的唯一标识符
            user_id: 用户的唯一标识符
            tool_type: 工具的提供者类型
            provider: 工具的提供者名称
            tool_name: 工具的名称
            tool_parameters: 工具的参数字典
            credential_id: 可选的凭据标识符

        Returns:
            生成器，生成ToolInvokeMessage对象

        Raises:
            Exception: 当工具调用过程中发生异常时抛出异常
        """
        # 获取工具运行时
        try:
            tool_runtime = ToolManager.get_tool_runtime_from_plugin(
                tool_type, tenant_id, provider, tool_name, tool_parameters, credential_id
            )

            # 调用工具引擎执行工具调用
            response = ToolEngine.generic_invoke(
                tool_runtime, tool_parameters, user_id, DifyWorkflowCallbackHandler(), workflow_call_depth=1
            )

            # 转换工具调用消息
            response = ToolFileMessageTransformer.transform_tool_invoke_messages(
                response, user_id=user_id, tenant_id=tenant_id
            )

            return response
        except Exception as e:
            # 捕获并重新抛出异常
            raise e
