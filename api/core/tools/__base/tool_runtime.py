from typing import Any, Optional

from openai import BaseModel
from pydantic import Field

from core.app.entities.app_invoke_entities import InvokeFrom
from core.tools.entities.tool_entities import CredentialType, ToolInvokeFrom


class ToolRuntime(BaseModel):
    """
    工具调用处理的元数据类

    用于存储工具运行时所需的各种配置和上下文信息

    Attributes:
        tenant_id (str): 租户ID，标识工具所属的租户
        tool_id (Optional[str]): 工具ID，唯一标识一个工具实例
        invoke_from (Optional[InvokeFrom]): 调用来源，表示工具被哪个组件调用
        tool_invoke_from (Optional[ToolInvokeFrom]): 工具调用来源，表示工具的调用上下文
        credentials (dict[str, Any]): 凭据信息，存储工具认证所需的键值对
        credential_type (CredentialType): 凭据类型，默认为API_KEY类型
        runtime_parameters (dict[str, Any]): 运行时参数，存储工具执行时的动态配置
    """

    tenant_id: str
    tool_id: Optional[str] = None
    invoke_from: Optional[InvokeFrom] = None
    tool_invoke_from: Optional[ToolInvokeFrom] = None
    credentials: dict[str, Any] = Field(default_factory=dict)
    credential_type: CredentialType = Field(default=CredentialType.API_KEY)
    runtime_parameters: dict[str, Any] = Field(default_factory=dict)


class FakeToolRuntime(ToolRuntime):
    """
    用于测试的虚假工具运行时类

    继承自ToolRuntime，提供预设的测试数据用于单元测试和调试
    """

    def __init__(self):
        """
        初始化虚假工具运行时实例

        设置预定义的测试值来模拟真实的工具运行环境
        """
        super().__init__(
            tenant_id="fake_tenant_id",
            tool_id="fake_tool_id",
            invoke_from=InvokeFrom.DEBUGGER,
            tool_invoke_from=ToolInvokeFrom.AGENT,
            credentials={},
            runtime_parameters={},
        )
