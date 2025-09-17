import json
import logging
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Optional, Union

from core.agent.base_agent_runner import BaseAgentRunner
from core.app.apps.base_app_queue_manager import PublishFrom
from core.app.entities.queue_entities import QueueAgentThoughtEvent, QueueMessageEndEvent, QueueMessageFileEvent
from core.file import file_manager
from core.model_runtime.entities import (
    AssistantPromptMessage,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
    PromptMessage,
    PromptMessageContentType,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import ImagePromptMessageContent, PromptMessageContentUnionTypes
from core.prompt.agent_history_prompt_transform import AgentHistoryPromptTransform
from core.tools.entities.tool_entities import ToolInvokeMeta
from core.tools.tool_engine import ToolEngine
from models.model import Message

logger = logging.getLogger(__name__)


class FunctionCallAgentRunner(BaseAgentRunner):
    """
    函数调用代理运行器类，用于处理基于函数调用的代理应用。

    该类继承自BaseAgentRunner，实现了基于函数调用的代理逻辑。它通过迭代执行以下步骤：
    1. 组织提示消息
    2. 调用支持函数调用的大语言模型
    3. 解析模型返回的函数调用
    4. 执行函数调用并获取结果
    5. 重复直到得出最终答案或达到最大迭代次数
    """

    def run(self, message: Message, query: str, **kwargs: Any) -> Generator[LLMResultChunk, None, None]:
        """
        运行函数调用代理应用，执行完整的推理过程。

        该方法是函数调用代理应用的核心执行逻辑，通过迭代执行以下步骤：
        1. 组织提示消息
        2. 调用大语言模型（支持函数调用）
        3. 解析模型输出的函数调用
        4. 执行函数调用并处理结果
        5. 重复直到达到最终答案或最大迭代次数

        Args:
            message (Message): 消息对象
            query (str): 用户查询
            **kwargs (Any): 其他参数

        Yields:
            Generator[LLMResultChunk, None, None]: 返回LLM结果块的生成器
        """
        self.query = query
        app_generate_entity = self.application_generate_entity

        app_config = self.app_config
        assert app_config is not None, "app_config is required"
        assert app_config.agent is not None, "app_config.agent is required"

        # 将工具转换为ModelRuntime工具格式
        tool_instances, prompt_messages_tools = self._init_prompt_tools()

        assert app_config.agent

        iteration_step = 1
        max_iteration_steps = min(app_config.agent.max_iteration, 99) + 1

        # 继续运行直到没有任何工具调用
        function_call_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        # 获取跟踪实例
        trace_manager = app_generate_entity.trace_manager

        def increase_usage(final_llm_usage_dict: dict[str, Optional[LLMUsage]], usage: LLMUsage):
            """
            增加LLM使用量统计

            Args:
                final_llm_usage_dict (dict[str, Optional[LLMUsage]]): 最终LLM使用量字典
                usage (LLMUsage): 当前使用量
            """
            if not final_llm_usage_dict["usage"]:
                final_llm_usage_dict["usage"] = usage
            else:
                llm_usage = final_llm_usage_dict["usage"]
                llm_usage.prompt_tokens += usage.prompt_tokens
                llm_usage.completion_tokens += usage.completion_tokens
                llm_usage.total_tokens += usage.total_tokens
                llm_usage.prompt_price += usage.prompt_price
                llm_usage.completion_price += usage.completion_price
                llm_usage.total_price += usage.total_price

        model_instance = self.model_instance

        # 迭代执行推理过程，直到没有函数调用或达到最大迭代次数
        while function_call_state and iteration_step <= max_iteration_steps:
            function_call_state = False

            # 如果是最后一次迭代，移除所有工具
            if iteration_step == max_iteration_steps:
                prompt_messages_tools = []

            message_file_ids: list[str] = []
            # 创建代理思考记录
            agent_thought_id = self.create_agent_thought(
                message_id=message.id, message="", tool_name="", tool_input="", messages_ids=message_file_ids
            )

            # 重新计算LLM最大令牌数
            prompt_messages = self._organize_prompt_messages()
            self.recalc_llm_max_tokens(self.model_config, prompt_messages)

            # 调用模型
            chunks: Union[Generator[LLMResultChunk, None, None], LLMResult] = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters=app_generate_entity.model_conf.parameters,
                tools=prompt_messages_tools,
                stop=app_generate_entity.model_conf.stop,
                stream=self.stream_tool_call,
                user=self.user_id,
                callbacks=[],
            )

            # 存储工具调用信息
            tool_calls: list[tuple[str, str, dict[str, Any]]] = []

            # 保存完整响应
            response = ""

            # 保存工具调用名称和输入
            tool_call_names = ""
            tool_call_inputs = ""

            current_llm_usage = None

            # 处理模型返回的结果
            if isinstance(chunks, Generator):
                is_first_chunk = True
                for chunk in chunks:
                    if is_first_chunk:
                        self.queue_manager.publish(
                            QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                        )
                        is_first_chunk = False
                    # 检查是否有任何工具调用
                    if self.check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self.extract_tool_calls(chunk) or [])
                        tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])
                        try:
                            tool_call_inputs = json.dumps(
                                {tool_call[1]: tool_call[2] for tool_call in tool_calls}, ensure_ascii=False
                            )
                        except TypeError:
                            # 降级处理：强制使用ASCII编码处理不可序列化的对象
                            tool_call_inputs = json.dumps({tool_call[1]: tool_call[2] for tool_call in tool_calls})

                    # 累加响应内容
                    if chunk.delta.message and chunk.delta.message.content:
                        if isinstance(chunk.delta.message.content, list):
                            for content in chunk.delta.message.content:
                                response += content.data
                        else:
                            response += str(chunk.delta.message.content)

                    # 累加使用量
                    if chunk.delta.usage:
                        increase_usage(llm_usage, chunk.delta.usage)
                        current_llm_usage = chunk.delta.usage

                    yield chunk
            else:
                result = chunks
                # 检查是否有任何阻塞式工具调用
                if self.check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self.extract_blocking_tool_calls(result) or [])
                    tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])
                    try:
                        tool_call_inputs = json.dumps(
                            {tool_call[1]: tool_call[2] for tool_call in tool_calls}, ensure_ascii=False
                        )
                    except TypeError:
                        # 降级处理：强制使用ASCII编码处理不可序列化的对象
                        tool_call_inputs = json.dumps({tool_call[1]: tool_call[2] for tool_call in tool_calls})

                # 处理使用量
                if result.usage:
                    increase_usage(llm_usage, result.usage)
                    current_llm_usage = result.usage

                # 处理消息内容
                if result.message and result.message.content:
                    if isinstance(result.message.content, list):
                        for content in result.message.content:
                            response += content.data
                    else:
                        response += str(result.message.content)

                if not result.message.content:
                    result.message.content = ""

                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                )

                yield LLMResultChunk(
                    model=model_instance.model,
                    prompt_messages=result.prompt_messages,
                    system_fingerprint=result.system_fingerprint,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=result.message,
                        usage=result.usage,
                    ),
                )

            # 创建助手消息
            assistant_message = AssistantPromptMessage(content="", tool_calls=[])
            if tool_calls:
                # 如果有工具调用，则设置工具调用信息
                assistant_message.tool_calls = [
                    AssistantPromptMessage.ToolCall(
                        id=tool_call[0],
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tool_call[1], arguments=json.dumps(tool_call[2], ensure_ascii=False)
                        ),
                    )
                    for tool_call in tool_calls
                ]
            else:
                # 如果没有工具调用，则设置响应内容
                assistant_message.content = response

            self._current_thoughts.append(assistant_message)

            # 保存思考记录
            self.save_agent_thought(
                agent_thought_id=agent_thought_id,
                tool_name=tool_call_names,
                tool_input=tool_call_inputs,
                thought=response,
                tool_invoke_meta=None,
                observation=None,
                answer=response,
                messages_ids=[],
                llm_usage=current_llm_usage,
            )
            self.queue_manager.publish(
                QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
            )

            final_answer += response + "\n"

            # 调用工具
            tool_responses = []
            for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                tool_instance = tool_instances.get(tool_call_name)
                if not tool_instance:
                    # 如果工具实例不存在，则返回错误信息
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": f"there is not a tool named {tool_call_name}",
                        "meta": ToolInvokeMeta.error_instance(f"there is not a tool named {tool_call_name}").to_dict(),
                    }
                else:
                    # 调用工具
                    tool_invoke_response, message_files, tool_invoke_meta = ToolEngine.agent_invoke(
                        tool=tool_instance,
                        tool_parameters=tool_call_args,
                        user_id=self.user_id,
                        tenant_id=self.tenant_id,
                        message=self.message,
                        invoke_from=self.application_generate_entity.invoke_from,
                        agent_tool_callback=self.agent_callback,
                        trace_manager=trace_manager,
                        app_id=self.application_generate_entity.app_config.app_id,
                        message_id=self.message.id,
                        conversation_id=self.conversation.id,
                    )
                    # 发布文件
                    for message_file_id in message_files:
                        # 发布消息文件
                        self.queue_manager.publish(
                            QueueMessageFileEvent(message_file_id=message_file_id), PublishFrom.APPLICATION_MANAGER
                        )
                        # 添加消息文件ID
                        message_file_ids.append(message_file_id)

                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": tool_invoke_response,
                        "meta": tool_invoke_meta.to_dict(),
                    }

                tool_responses.append(tool_response)
                if tool_response["tool_response"] is not None:
                    # 将工具响应添加到当前思考中
                    self._current_thoughts.append(
                        ToolPromptMessage(
                            content=str(tool_response["tool_response"]),
                            tool_call_id=tool_call_id,
                            name=tool_call_name,
                        )
                    )

            # 如果有工具响应，则保存代理思考
            if len(tool_responses) > 0:
                # 保存代理思考
                self.save_agent_thought(
                    agent_thought_id=agent_thought_id,
                    tool_name="",
                    tool_input="",
                    thought="",
                    tool_invoke_meta={
                        tool_response["tool_call_name"]: tool_response["meta"] for tool_response in tool_responses
                    },
                    observation={
                        tool_response["tool_call_name"]: tool_response["tool_response"]
                        for tool_response in tool_responses
                    },
                    answer="",
                    messages_ids=message_file_ids,
                )
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                )

            # 更新提示工具
            for prompt_tool in prompt_messages_tools:
                self.update_prompt_message_tool(tool_instances[prompt_tool.name], prompt_tool)

            iteration_step += 1

        # 发布结束事件
        self.queue_manager.publish(
            QueueMessageEndEvent(
                llm_result=LLMResult(
                    model=model_instance.model,
                    prompt_messages=prompt_messages,
                    message=AssistantPromptMessage(content=final_answer),
                    usage=llm_usage["usage"] or LLMUsage.empty_usage(),
                    system_fingerprint="",
                )
            ),
            PublishFrom.APPLICATION_MANAGER,
        )

    def check_tool_calls(self, llm_result_chunk: LLMResultChunk) -> bool:
        """
        检查LLM结果块中是否有任何工具调用。

        Args:
            llm_result_chunk (LLMResultChunk): LLM结果块

        Returns:
            bool: 如果有工具调用返回True，否则返回False
        """
        if llm_result_chunk.delta.message.tool_calls:
            return True
        return False

    def check_blocking_tool_calls(self, llm_result: LLMResult) -> bool:
        """
        检查LLM结果中是否有任何阻塞式工具调用。

        Args:
            llm_result (LLMResult): LLM结果

        Returns:
            bool: 如果有工具调用返回True，否则返回False
        """
        if llm_result.message.tool_calls:
            return True
        return False

    def extract_tool_calls(self, llm_result_chunk: LLMResultChunk) -> list[tuple[str, str, dict[str, Any]]]:
        """
        从LLM结果块中提取工具调用信息。

        Args:
            llm_result_chunk (LLMResultChunk): LLM结果块

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: 工具调用信息列表，每个元素为(工具调用ID, 工具调用名称, 工具调用参数)
        """
        tool_calls = []
        for prompt_message in llm_result_chunk.delta.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def extract_blocking_tool_calls(self, llm_result: LLMResult) -> list[tuple[str, str, dict[str, Any]]]:
        """
        从阻塞式LLM结果中提取工具调用信息。

        Args:
            llm_result (LLMResult): LLM结果

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: 工具调用信息列表，每个元素为(工具调用ID, 工具调用名称, 工具调用参数)
        """
        tool_calls = []
        for prompt_message in llm_result.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def _init_system_message(self, prompt_template: str, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        初始化系统消息。

        Args:
            prompt_template (str): 提示模板
            prompt_messages (list[PromptMessage]): 提示消息列表

        Returns:
            list[PromptMessage]: 更新后的提示消息列表
        """
        if not prompt_messages and prompt_template:
            return [
                SystemPromptMessage(content=prompt_template),
            ]

        if prompt_messages and not isinstance(prompt_messages[0], SystemPromptMessage) and prompt_template:
            prompt_messages.insert(0, SystemPromptMessage(content=prompt_template))

        return prompt_messages or []

    def _organize_user_query(self, query: str, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        组织用户查询消息，支持文件和图像输入。

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

    def _clear_user_prompt_image_messages(self, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        清除用户提示中的图像消息。

        目前GPT在第一次迭代时同时支持函数调用和视觉功能。
        我们需要在第一次迭代后从提示消息中移除图像消息。

        Args:
            prompt_messages (list[PromptMessage]): 提示消息列表

        Returns:
            list[PromptMessage]: 清理后的提示消息列表
        """
        prompt_messages = deepcopy(prompt_messages)

        for prompt_message in prompt_messages:
            if isinstance(prompt_message, UserPromptMessage):
                if isinstance(prompt_message.content, list):
                    prompt_message.content = "\n".join(
                        [
                            content.data
                            if content.type == PromptMessageContentType.TEXT
                            else "[image]"
                            if content.type == PromptMessageContentType.IMAGE
                            else "[file]"
                            for content in prompt_message.content
                        ]
                    )

        return prompt_messages

    def _organize_prompt_messages(self):
        """
        组织提示消息。

        Returns:
            list[PromptMessage]: 组织后的提示消息列表
        """
        prompt_template = self.app_config.prompt_template.simple_prompt_template or ""
        self.history_prompt_messages = self._init_system_message(prompt_template, self.history_prompt_messages)
        query_prompt_messages = self._organize_user_query(self.query or "", [])

        self.history_prompt_messages = AgentHistoryPromptTransform(
            model_config=self.model_config,
            prompt_messages=[*query_prompt_messages, *self._current_thoughts],
            history_messages=self.history_prompt_messages,
            memory=self.memory,
        ).get_prompt()

        prompt_messages = [*self.history_prompt_messages, *query_prompt_messages, *self._current_thoughts]
        if len(self._current_thoughts) != 0:
            # 第一次迭代后清除消息
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)
        return prompt_messages
