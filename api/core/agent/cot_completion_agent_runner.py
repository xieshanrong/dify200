import json
from typing import Optional

from core.agent.cot_agent_runner import CotAgentRunner
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.utils.encoders import jsonable_encoder


class CotCompletionAgentRunner(CotAgentRunner):
    """
    Chain-of-Thought补全代理运行器类，用于处理基于思维链的补全代理应用。

    该类继承自CotAgentRunner，专门用于处理补全场景下的代理应用，负责组织指令提示、
    历史提示和完整的提示消息序列。
    """

    def _organize_instruction_prompt(self) -> str:
        """
        组织指令提示消息，包含代理执行任务所需的上下文信息。

        该方法构建系统级别的提示信息，包括用户指令、可用工具列表和工具名称等，
        为代理执行提供必要的上下文。

        Returns:
            str: 组织好的指令提示字符串
        """
        # 检查代理配置是否存在
        if self.app_config.agent is None:
            raise ValueError("Agent configuration is not set")
        prompt_entity = self.app_config.agent.prompt
        # 检查提示实体是否存在
        if prompt_entity is None:
            raise ValueError("prompt entity is not set")
        first_prompt = prompt_entity.first_prompt

        # 构建系统提示消息，替换模板中的占位符
        system_prompt = (
            first_prompt.replace("{{instruction}}", self._instruction)
            .replace("{{tools}}", json.dumps(jsonable_encoder(self._prompt_messages_tools)))
            .replace("{{tool_names}}", ", ".join([tool.name for tool in self._prompt_messages_tools]))
        )

        return system_prompt

    def _organize_historic_prompt(self, current_session_messages: Optional[list[PromptMessage]] = None) -> str:
        """
        组织历史提示消息，将历史对话转换为文本格式。

        该方法处理历史对话消息，将用户消息和助手消息格式化为文本字符串，
        用于在当前对话中提供上下文信息。

        Args:
            current_session_messages (Optional[list[PromptMessage]]): 当前会话消息列表

        Returns:
            str: 格式化后的历史提示字符串
        """
        # 组织历史提示消息
        historic_prompt_messages = self._organize_historic_prompt_messages(current_session_messages)
        historic_prompt = ""

        # 遍历历史消息，根据消息类型进行格式化
        for message in historic_prompt_messages:
            if isinstance(message, UserPromptMessage):
                # 用户消息格式化为"Question: {内容}"
                historic_prompt += f"Question: {message.content}\n\n"
            elif isinstance(message, AssistantPromptMessage):
                # 助手消息根据内容类型进行处理
                if isinstance(message.content, str):
                    # 字符串内容直接添加
                    historic_prompt += message.content + "\n\n"
                elif isinstance(message.content, list):
                    # 列表内容遍历处理文本内容
                    for content in message.content:
                        if not isinstance(content, TextPromptMessageContent):
                            continue
                        historic_prompt += content.data

        return historic_prompt

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        """
        组织完整的提示消息序列，用于补全场景。

        该方法是补全代理的核心方法，负责将系统提示、历史对话、当前思考过程和用户查询
        组合成一个完整的提示消息，供大语言模型使用。

        Returns:
            list[PromptMessage]: 包含单个用户提示消息的列表
        """
        # 组织系统指令提示
        system_prompt = self._organize_instruction_prompt()

        # 组织历史提示消息
        historic_prompt = self._organize_historic_prompt()

        # 组织当前助手消息（基于代理便笺）
        agent_scratchpad = self._agent_scratchpad
        assistant_prompt = ""
        # 遍历代理想要记录的单元
        for unit in agent_scratchpad or []:
            if unit.is_final():
                # 如果是最终答案，则添加最终答案内容
                assistant_prompt += f"Final Answer: {unit.agent_response}"
            else:
                # 如果不是最终答案，则添加思考过程
                assistant_prompt += f"Thought: {unit.thought}\n\n"
                # 如果有行动字符串，则添加行动信息
                if unit.action_str:
                    assistant_prompt += f"Action: {unit.action_str}\n\n"
                # 如果有观察结果，则添加观察结果
                if unit.observation:
                    assistant_prompt += f"Observation: {unit.observation}\n\n"

        # 组织查询提示
        query_prompt = f"Question: {self._query}"

        # 合并所有消息，替换模板中的占位符
        prompt = (
            system_prompt.replace("{{historic_messages}}", historic_prompt)
            .replace("{{agent_scratchpad}}", assistant_prompt)
            .replace("{{query}}", query_prompt)
        )

        # 返回包含单个用户提示消息的列表
        return [UserPromptMessage(content=prompt)]
