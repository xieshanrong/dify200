from enum import StrEnum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from core.tools.entities.tool_entities import ToolInvokeMessage, ToolProviderType


class AgentToolEntity(BaseModel):
    """
    代理工具实体类，用于表示代理可以使用的工具信息。

    该类继承自BaseModel，用于定义代理工具的基本属性，包括提供者类型、ID、工具名称等。
    """

    provider_type: ToolProviderType
    provider_id: str
    tool_name: str
    tool_parameters: dict[str, Any] = Field(default_factory=dict)
    plugin_unique_identifier: str | None = None
    credential_id: str | None = None


class AgentPromptEntity(BaseModel):
    """
    代理提示实体类，用于定义代理使用的提示模板。

    该类包含代理在不同阶段使用的提示词模板。
    """

    first_prompt: str
    next_iteration: str


class AgentScratchpadUnit(BaseModel):
    """
    代理便笺单元类，用于记录代理在推理过程中的中间状态。

    该类记录了代理在执行过程中的思考、行动、观察等信息，是代理推理过程的基本单元。
    """

    class Action(BaseModel):
        """
        行动实体类，用于表示代理执行的具体行动。

        该类定义了代理执行的行动名称和输入参数。
        """

        action_name: str
        action_input: Union[dict, str]

        def to_dict(self):
            """
            将行动实体转换为字典格式。

            Returns:
                dict: 包含行动名称和行动输入的字典
            """
            return {
                "action": self.action_name,
                "action_input": self.action_input,
            }

    agent_response: Optional[str] = None
    thought: Optional[str] = None
    action_str: Optional[str] = None
    observation: Optional[str] = None
    action: Optional[Action] = None

    def is_final(self) -> bool:
        """
        检查便笺单元是否为最终结果。

        判断标准为：action为空，或者action_name中同时包含"final"和"answer"关键词。

        Returns:
            bool: 如果是最终结果返回True，否则返回False
        """
        return self.action is None or (
            "final" in self.action.action_name.lower() and "answer" in self.action.action_name.lower()
        )


class AgentEntity(BaseModel):
    """
    代理实体类，用于定义代理的基本配置和属性。

    该类包含了代理的核心配置信息，如模型提供者、模型名称、策略等。
    """

    class Strategy(StrEnum):
        """
        代理策略枚举类，定义了代理可使用的不同推理策略。
        """

        CHAIN_OF_THOUGHT = "chain-of-thought"
        FUNCTION_CALLING = "function-calling"

    provider: str
    model: str
    strategy: Strategy
    prompt: Optional[AgentPromptEntity] = None
    tools: Optional[list[AgentToolEntity]] = None
    max_iteration: int = 10


class AgentInvokeMessage(ToolInvokeMessage):
    """
    代理调用消息类，用于表示代理调用过程中的消息。

    该类继承自ToolInvokeMessage，用于处理代理调用相关的信息。
    """

    pass
