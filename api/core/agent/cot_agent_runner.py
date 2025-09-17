import json
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Optional

from core.agent.base_agent_runner import BaseAgentRunner
from core.agent.entities import AgentScratchpadUnit
from core.agent.output_parser.cot_output_parser import CotAgentOutputParser
from core.app.apps.base_app_queue_manager import PublishFrom
from core.app.entities.queue_entities import QueueAgentThoughtEvent, QueueMessageEndEvent, QueueMessageFileEvent
from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.ops.ops_trace_manager import TraceQueueManager
from core.prompt.agent_history_prompt_transform import AgentHistoryPromptTransform
from core.tools.__base.tool import Tool
from core.tools.entities.tool_entities import ToolInvokeMeta
from core.tools.tool_engine import ToolEngine
from models.model import Message


class CotAgentRunner(BaseAgentRunner, ABC):
    """
    Chain-of-Thought代理运行器类，用于执行基于思维链的代理应用。

    该类继承自BaseAgentRunner和ABC，实现了基于思维链(Chain-of-Thought)的代理逻辑。
    它通过迭代推理过程，根据模型输出调用工具并处理结果，直到得出最终答案。
    """
    _is_first_iteration = True
    _ignore_observation_providers = ["wenxin"]
    _historic_prompt_messages: list[PromptMessage]
    _agent_scratchpad: list[AgentScratchpadUnit]
    _instruction: str
    _query: str
    _prompt_messages_tools: Sequence[PromptMessageTool]

    def run(
        self,
        message: Message,
        query: str,
        inputs: Mapping[str, str],
    ) -> Generator:
        """
        运行Cot代理应用，执行完整的推理过程。

        该方法是代理应用的核心执行逻辑，通过迭代执行以下步骤：
        1. 组织提示消息
        2. 调用大语言模型
        3. 解析模型输出
        4. 执行工具调用（如果需要）
        5. 重复直到达到最终答案或最大迭代次数

        Args:
            message (Message): 消息对象
            query (str): 用户查询
            inputs (Mapping[str, str]): 输入参数映射

        Yields:
            Generator: 返回LLM结果块的生成器
        """

        app_generate_entity = self.application_generate_entity
        self._repack_app_generate_entity(app_generate_entity)
        self._init_react_state(query)

        trace_manager = app_generate_entity.trace_manager

        # 检查模型模式，确保停止词包含"Observation"
        if "Observation" not in app_generate_entity.model_conf.stop:
            if app_generate_entity.model_conf.provider not in self._ignore_observation_providers:
                app_generate_entity.model_conf.stop.append("Observation")

        app_config = self.app_config
        assert app_config.agent

        # 初始化指令
        inputs = inputs or {}
        instruction = app_config.prompt_template.simple_prompt_template or ""
        self._instruction = self._fill_in_inputs_from_external_data_tools(instruction, inputs)

        iteration_step = 1
        max_iteration_steps = min(app_config.agent.max_iteration, 99) + 1

        # 将工具转换为ModelRuntime工具格式
        tool_instances, prompt_messages_tools = self._init_prompt_tools()
        self._prompt_messages_tools = prompt_messages_tools

        function_call_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

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

        # 迭代执行推理过程，直到没有工具调用或达到最大迭代次数
        while function_call_state and iteration_step <= max_iteration_steps:
            # 继续运行直到没有任何工具调用
            function_call_state = False

            if iteration_step == max_iteration_steps:
                # 最后一次迭代，移除所有工具
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # 创建代理思考记录
            agent_thought_id = self.create_agent_thought(
                message_id=message.id, message="", tool_name="", tool_input="", messages_ids=message_file_ids
            )

            # 如果不是第一次迭代，发布代理思考事件
            if iteration_step > 1:
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                )

            # 重新计算LLM最大令牌数
            prompt_messages = self._organize_prompt_messages()
            self.recalc_llm_max_tokens(self.model_config, prompt_messages)

            # 调用模型
            chunks = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters=app_generate_entity.model_conf.parameters,
                tools=[],
                stop=app_generate_entity.model_conf.stop,
                stream=True,
                user=self.user_id,
                callbacks=[],
            )

            usage_dict: dict[str, Optional[LLMUsage]] = {}
            # 处理模型输出流
            react_chunks = CotAgentOutputParser.handle_react_stream_output(chunks, usage_dict)
            # 初始化便笺单元
            scratchpad = AgentScratchpadUnit(
                agent_response="",
                thought="",
                action_str="",
                observation="",
                action=None,
            )

            # 如果是第一次迭代，发布代理思考事件
            if iteration_step == 1:
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                )

            # 处理模型输出的每个块
            for chunk in react_chunks:
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    action = chunk
                    # 检测到行动
                    assert scratchpad.agent_response is not None
                    scratchpad.agent_response += json.dumps(chunk.model_dump())
                    scratchpad.action_str = json.dumps(chunk.model_dump())
                    scratchpad.action = action
                else:
                    assert scratchpad.agent_response is not None
                    scratchpad.agent_response += chunk
                    assert scratchpad.thought is not None
                    scratchpad.thought += chunk
                    # 生成结果块
                    yield LLMResultChunk(
                        model=self.model_config.model,
                        prompt_messages=prompt_messages,
                        system_fingerprint="",
                        delta=LLMResultChunkDelta(index=0, message=AssistantPromptMessage(content=chunk), usage=None),
                    )

            # 确保思考内容不为空
            assert scratchpad.thought is not None
            scratchpad.thought = scratchpad.thought.strip() or "I am thinking about how to help you"
            self._agent_scratchpad.append(scratchpad)

            # 获取LLM使用量
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            # 保存代理思考
            self.save_agent_thought(
                agent_thought_id=agent_thought_id,
                tool_name=(scratchpad.action.action_name if scratchpad.action and not scratchpad.is_final() else ""),
                tool_input={scratchpad.action.action_name: scratchpad.action.action_input} if scratchpad.action else {},
                tool_invoke_meta={},
                thought=scratchpad.thought or "",
                observation="",
                answer=scratchpad.agent_response or "",
                messages_ids=[],
                llm_usage=usage_dict["usage"],
            )

            # 如果不是最终结果，发布代理思考事件
            if not scratchpad.is_final():
                self.queue_manager.publish(
                    QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                )

            # 处理行动结果
            if not scratchpad.action:
                # 无法提取行动，直接返回最终答案
                final_answer = ""
            else:
                if scratchpad.action.action_name.lower() == "final answer":
                    # 行动是最终答案，直接返回最终答案
                    try:
                        if isinstance(scratchpad.action.action_input, dict):
                            final_answer = json.dumps(scratchpad.action.action_input, ensure_ascii=False)
                        elif isinstance(scratchpad.action.action_input, str):
                            final_answer = scratchpad.action.action_input
                        else:
                            final_answer = f"{scratchpad.action.action_input}"
                    except TypeError:
                        final_answer = f"{scratchpad.action.action_input}"
                else:
                    function_call_state = True
                    # 行动是工具调用，执行工具
                    tool_invoke_response, tool_invoke_meta = self._handle_invoke_action(
                        action=scratchpad.action,
                        tool_instances=tool_instances,
                        message_file_ids=message_file_ids,
                        trace_manager=trace_manager,
                    )
                    scratchpad.observation = tool_invoke_response
                    scratchpad.agent_response = tool_invoke_response

                    # 保存代理思考
                    self.save_agent_thought(
                        agent_thought_id=agent_thought_id,
                        tool_name=scratchpad.action.action_name,
                        tool_input={scratchpad.action.action_name: scratchpad.action.action_input},
                        thought=scratchpad.thought or "",
                        observation={scratchpad.action.action_name: tool_invoke_response},
                        tool_invoke_meta={scratchpad.action.action_name: tool_invoke_meta.to_dict()},
                        answer=scratchpad.agent_response,
                        messages_ids=message_file_ids,
                        llm_usage=usage_dict["usage"],
                    )

                    # 发布代理思考事件
                    self.queue_manager.publish(
                        QueueAgentThoughtEvent(agent_thought_id=agent_thought_id), PublishFrom.APPLICATION_MANAGER
                    )

                # 更新提示工具消息
                for prompt_tool in self._prompt_messages_tools:
                    self.update_prompt_message_tool(tool_instances[prompt_tool.name], prompt_tool)

            iteration_step += 1

        # 生成最终结果块
        yield LLMResultChunk(
            model=model_instance.model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0, message=AssistantPromptMessage(content=final_answer), usage=llm_usage["usage"]
            ),
            system_fingerprint="",
        )

        # 保存代理思考
        self.save_agent_thought(
            agent_thought_id=agent_thought_id,
            tool_name="",
            tool_input={},
            tool_invoke_meta={},
            thought=final_answer,
            observation={},
            answer=final_answer,
            messages_ids=[],
        )

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

    def _handle_invoke_action(
        self,
        action: AgentScratchpadUnit.Action,
        tool_instances: Mapping[str, Tool],
        message_file_ids: list[str],
        trace_manager: Optional[TraceQueueManager] = None,
    ) -> tuple[str, ToolInvokeMeta]:
        """
        处理执行行动的逻辑。

        Args:
            action (AgentScratchpadUnit.Action): 要执行的行动
            tool_instances (Mapping[str, Tool]): 工具实例映射
            message_file_ids (list[str]): 消息文件ID列表
            trace_manager (Optional[TraceQueueManager]): 跟踪管理器

        Returns:
            tuple[str, ToolInvokeMeta]: 返回观察结果和工具调用元数据的元组
        """
        # 行动是工具调用，执行工具
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)

        # 检查工具实例是否存在
        if not tool_instance:
            answer = f"there is not a tool named {tool_call_name}"
            return answer, ToolInvokeMeta.error_instance(answer)

        # 如果工具调用参数是字符串，尝试解析为JSON
        if isinstance(tool_call_args, str):
            try:
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError:
                pass

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
        )

        # 发布文件
        for message_file_id in message_files:
            # 发布消息文件事件
            self.queue_manager.publish(
                QueueMessageFileEvent(message_file_id=message_file_id), PublishFrom.APPLICATION_MANAGER
            )
            # 添加消息文件ID
            message_file_ids.append(message_file_id)

        return tool_invoke_response, tool_invoke_meta

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        将字典转换为行动对象。

        Args:
            action (dict): 包含行动信息的字典

        Returns:
            AgentScratchpadUnit.Action: 行动对象
        """
        return AgentScratchpadUnit.Action(action_name=action["action"], action_input=action["action_input"])

    def _fill_in_inputs_from_external_data_tools(self, instruction: str, inputs: Mapping[str, Any]) -> str:
        """
        从外部数据工具填充输入。

        Args:
            instruction (str): 指令模板
            inputs (Mapping[str, Any]): 输入参数映射

        Returns:
            str: 填充后的指令
        """
        for key, value in inputs.items():
            try:
                instruction = instruction.replace(f"{{{{{key}}}}}", str(value))
            except Exception:
                continue

        return instruction

    def _init_react_state(self, query):
        """
        初始化代理便笺。

        Args:
            query (str): 用户查询
        """
        self._query = query
        self._agent_scratchpad = []
        self._historic_prompt_messages = self._organize_historic_prompt_messages()

    @abstractmethod
    def _organize_prompt_messages(self) -> list[PromptMessage]:
        """
        组织提示消息，这是一个抽象方法，需要子类实现。

        Returns:
            list[PromptMessage]: 提示消息列表
        """

    def _format_assistant_message(self, agent_scratchpad: list[AgentScratchpadUnit]) -> str:
        """
        格式化助手消息。

        Args:
            agent_scratchpad (list[AgentScratchpadUnit]): 代理便笺列表

        Returns:
            str: 格式化后的助手消息
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message

    def _organize_historic_prompt_messages(
        self, current_session_messages: list[PromptMessage] | None = None
    ) -> list[PromptMessage]:
        """
        组织历史提示消息。

        Args:
            current_session_messages (list[PromptMessage] | None): 当前会话消息列表

        Returns:
            list[PromptMessage]: 组织后的历史提示消息列表
        """
        result: list[PromptMessage] = []
        scratchpads: list[AgentScratchpadUnit] = []
        current_scratchpad: AgentScratchpadUnit | None = None

        # 遍历历史提示消息
        for message in self.history_prompt_messages:
            if isinstance(message, AssistantPromptMessage):
                if not current_scratchpad:
                    assert isinstance(message.content, str)
                    current_scratchpad = AgentScratchpadUnit(
                        agent_response=message.content,
                        thought=message.content or "I am thinking about how to help you",
                        action_str="",
                        action=None,
                        observation=None,
                    )
                    scratchpads.append(current_scratchpad)
                if message.tool_calls:
                    try:
                        current_scratchpad.action = AgentScratchpadUnit.Action(
                            action_name=message.tool_calls[0].function.name,
                            action_input=json.loads(message.tool_calls[0].function.arguments),
                        )
                        current_scratchpad.action_str = json.dumps(current_scratchpad.action.to_dict())
                    except:
                        pass
            elif isinstance(message, ToolPromptMessage):
                if current_scratchpad:
                    assert isinstance(message.content, str)
                    current_scratchpad.observation = message.content
                else:
                    raise NotImplementedError("expected str type")
            elif isinstance(message, UserPromptMessage):
                if scratchpads:
                    result.append(AssistantPromptMessage(content=self._format_assistant_message(scratchpads)))
                    scratchpads = []
                    current_scratchpad = None

                result.append(message)

        # 处理剩余的便笺
        if scratchpads:
            result.append(AssistantPromptMessage(content=self._format_assistant_message(scratchpads)))

        # 使用历史提示转换器处理历史消息
        historic_prompts = AgentHistoryPromptTransform(
            model_config=self.model_config,
            prompt_messages=current_session_messages or [],
            history_messages=result,
            memory=self.memory,
        ).get_prompt()
        return historic_prompts
