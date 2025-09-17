import json

from core.agent.cot_agent_runner import CotAgentRunner
from core.file import file_manager
from core.model_runtime.entities import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import ImagePromptMessageContent, PromptMessageContentUnionTypes
from core.model_runtime.utils.encoders import jsonable_encoder


class CotChatAgentRunner(CotAgentRunner):
    """
    Chain-of-Thought聊天代理运行器类，用于处理基于思维链的聊天代理应用。

    该类继承自CotAgentRunner，专门用于处理聊天场景下的代理应用，负责组织系统提示、
    用户查询和完整的提示消息序列。
    """

    def _organize_system_prompt(self) -> SystemPromptMessage:
        """
        组织系统提示消息，包含指令和工具信息。

        系统提示消息是代理执行的第一条消息，包含了代理执行任务所需的上下文信息，
        包括用户指令、可用工具列表和工具名称等。

        Returns:
            SystemPromptMessage: 系统提示消息对象
        """
        # 确保代理配置和提示配置存在
        assert self.app_config.agent
        assert self.app_config.agent.prompt

        prompt_entity = self.app_config.agent.prompt
        if not prompt_entity:
            raise ValueError("Agent prompt configuration is not set")
        first_prompt = prompt_entity.first_prompt

        # 构建系统提示消息，替换模板中的占位符
        system_prompt = (
            first_prompt.replace("{{instruction}}", self._instruction)
            .replace("{{tools}}", json.dumps(jsonable_encoder(self._prompt_messages_tools)))
            .replace("{{tool_names}}", ", ".join([tool.name for tool in self._prompt_messages_tools]))
        )

        return SystemPromptMessage(content=system_prompt)

    def _organize_user_query(self, query, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        组织用户查询消息，支持文件和图像输入。

        该方法处理用户输入的查询内容，如果包含文件则进行特殊处理，将文件内容
        转换为适合模型理解的格式。

        Args:
            query (str): 用户查询内容
            prompt_messages (list[PromptMessage]): 提示消息列表

        Returns:
            list[PromptMessage]: 更新后的提示消息列表
        """
        # 检查是否有文件需要处理
        if self.files:
            # 获取图像详细配置
            image_detail_config = (
                self.application_generate_entity.file_upload_config.image_config.detail
                if (
                    self.application_generate_entity.file_upload_config
                    and self.application_generate_entity.file_upload_config.image_config
                )
                else None
            )
            # 如果没有设置图像详细配置，则使用默认的低详细度配置
            image_detail_config = image_detail_config or ImagePromptMessageContent.DETAIL.LOW

            # 初始化提示消息内容列表
            prompt_message_contents: list[PromptMessageContentUnionTypes] = []

            # 为每个文件创建提示消息内容
            for file in self.files:
                prompt_message_contents.append(
                    file_manager.to_prompt_message_content(
                        file,
                        image_detail_config=image_detail_config,
                    )
                )
            # 添加文本查询内容
            prompt_message_contents.append(TextPromptMessageContent(data=query))

            # 将用户消息添加到提示消息列表中
            prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
        else:
            # 如果没有文件，直接添加文本查询内容
            prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        """
        组织完整的提示消息序列，包括系统消息、历史消息、用户查询和助手消息。

        这是代理运行的核心方法之一，负责将所有相关的提示消息按照正确的顺序组织起来，
        供大语言模型使用。消息顺序通常为：系统消息 -> 历史消息 -> 用户查询 -> 助手响应。

        Returns:
            list[PromptMessage]: 完整的提示消息序列
        """
        # 组织系统提示消息
        system_message = self._organize_system_prompt()

        # 组织当前助手消息（基于代理便笺）
        agent_scratchpad = self._agent_scratchpad
        if not agent_scratchpad:
            # 如果没有便笺，则助手消息为空列表
            assistant_messages = []
        else:
            # 创建助手消息对象
            assistant_message = AssistantPromptMessage(content="")
            assistant_message.content = ""  # FIXME: type check tell mypy that assistant_message.content is str

            # 遍历代理便笺单元，构建助手消息内容
            for unit in agent_scratchpad:
                if unit.is_final():
                    # 如果是最终答案，则添加最终答案内容
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Final Answer: {unit.agent_response}"
                else:
                    # 如果不是最终答案，则添加思考过程
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Thought: {unit.thought}\n\n"
                    # 如果有行动字符串，则添加行动信息
                    if unit.action_str:
                        assistant_message.content += f"Action: {unit.action_str}\n\n"
                    # 如果有观察结果，则添加观察结果
                    if unit.observation:
                        assistant_message.content += f"Observation: {unit.observation}\n\n"

            assistant_messages = [assistant_message]

        # 组织查询消息
        query_messages = self._organize_user_query(self._query, [])

        # 根据是否存在助手消息，组织不同的消息结构
        if assistant_messages:
            # 如果有助手消息，则在历史消息中包含所有消息并添加"continue"提示
            historic_messages = self._organize_historic_prompt_messages(
                [system_message, *query_messages, *assistant_messages, UserPromptMessage(content="continue")]
            )
            messages = [
                system_message,
                *historic_messages,
                *query_messages,
                *assistant_messages,
                UserPromptMessage(content="continue"),
            ]
        else:
            # 如果没有助手消息，则历史消息只包含系统消息和查询消息
            historic_messages = self._organize_historic_prompt_messages([system_message, *query_messages])
            messages = [system_message, *historic_messages, *query_messages]

        # 返回所有消息的组合
        return messages
