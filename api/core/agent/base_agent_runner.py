import json
import logging
import uuid
from typing import Optional, Union, cast

from sqlalchemy import select

from core.agent.entities import AgentEntity, AgentToolEntity
from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfig
from core.app.apps.base_app_queue_manager import AppQueueManager
from core.app.apps.base_app_runner import AppRunner
from core.app.entities.app_invoke_entities import (
    AgentChatAppGenerateEntity,
    ModelConfigWithCredentialsEntity,
)
from core.callback_handler.agent_tool_callback_handler import DifyAgentCallbackHandler
from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.file import file_manager
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    LLMUsage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import ImagePromptMessageContent, PromptMessageContentUnionTypes
from core.model_runtime.entities.model_entities import ModelFeature
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.prompt.utils.extract_thread_messages import extract_thread_messages
from core.tools.__base.tool import Tool
from core.tools.entities.tool_entities import (
    ToolParameter,
)
from core.tools.tool_manager import ToolManager
from core.tools.utils.dataset_retriever_tool import DatasetRetrieverTool
from extensions.ext_database import db
from factories import file_factory
from models.model import Conversation, Message, MessageAgentThought, MessageFile

logger = logging.getLogger(__name__)


class BaseAgentRunner(AppRunner):
    """
    基础代理运行器类，用于处理代理应用的执行逻辑。

    该类是所有代理运行器的基类，负责管理代理的运行时状态、工具调用、历史记录组织等核心功能。
    它继承自AppRunner类，提供了代理特有的功能实现。
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        application_generate_entity: AgentChatAppGenerateEntity,
        conversation: Conversation,
        app_config: AgentChatAppConfig,
        model_config: ModelConfigWithCredentialsEntity,
        config: AgentEntity,
        queue_manager: AppQueueManager,
        message: Message,
        user_id: str,
        model_instance: ModelInstance,
        memory: Optional[TokenBufferMemory] = None,
        prompt_messages: Optional[list[PromptMessage]] = None,
    ):
        """
        初始化基础代理运行器实例。

        Args:
            tenant_id (str): 租户ID，用于标识当前租户
            application_generate_entity (AgentChatAppGenerateEntity): 应用生成实体，包含应用调用的相关信息
            conversation (Conversation): 对话对象，表示当前对话上下文
            app_config (AgentChatAppConfig): 应用配置对象，包含代理应用的配置信息
            model_config (ModelConfigWithCredentialsEntity): 模型配置（包含凭证），用于配置模型参数
            config (AgentEntity): 代理实体配置，包含代理的详细配置
            queue_manager (AppQueueManager): 队列管理器，用于管理应用执行过程中的消息队列
            message (Message): 消息对象，表示当前处理的消息
            user_id (str): 用户ID，标识当前操作的用户
            model_instance (ModelInstance): 模型实例，提供模型调用的接口
            memory (Optional[TokenBufferMemory]): 令牌缓冲内存，用于管理对话历史（可选）
            prompt_messages (Optional[list[PromptMessage]]): 提示消息列表，包含历史提示消息（可选）
        """
        # 基础属性设置
        self.tenant_id = tenant_id
        self.application_generate_entity = application_generate_entity
        self.conversation = conversation
        self.app_config = app_config
        self.model_config = model_config
        self.config = config
        self.queue_manager = queue_manager
        self.message = message
        self.user_id = user_id
        self.memory = memory
        # 组织代理历史消息
        self.history_prompt_messages = self.organize_agent_history(prompt_messages=prompt_messages or [])
        self.model_instance = model_instance

        # 初始化代理回调处理器，用于处理代理执行过程中的回调事件
        self.agent_callback = DifyAgentCallbackHandler()

        # 初始化数据集工具回调处理器和数据集工具
        # 数据集工具回调处理器用于处理数据集检索过程中的回调事件
        hit_callback = DatasetIndexToolCallbackHandler(
            queue_manager=queue_manager,
            app_id=self.app_config.app_id,
            message_id=message.id,
            user_id=user_id,
            invoke_from=self.application_generate_entity.invoke_from,
        )
        # 获取数据集工具列表，用于检索相关数据集
        self.dataset_tools = DatasetRetrieverTool.get_dataset_tools(
            tenant_id=tenant_id,
            dataset_ids=app_config.dataset.dataset_ids if app_config.dataset else [],
            retrieve_config=app_config.dataset.retrieve_config if app_config.dataset else None,
            return_resource=(
                app_config.additional_features.show_retrieve_source if app_config.additional_features else False
            ),
            invoke_from=application_generate_entity.invoke_from,
            hit_callback=hit_callback,
            user_id=user_id,
            inputs=cast(dict, application_generate_entity.inputs),
        )

        # 获取当前消息已创建的代理思考数量，用于为新的思考分配位置
        self.agent_thought_count = (
            db.session.query(MessageAgentThought)
            .where(
                MessageAgentThought.message_id == self.message.id,
                )
            .count()
        )
        db.session.close()

        # 检查模型是否支持流式工具调用功能
        llm_model = cast(LargeLanguageModel, model_instance.model_type_instance)
        model_schema = llm_model.get_model_schema(model_instance.model, model_instance.credentials)
        features = model_schema.features if model_schema and model_schema.features else []
        self.stream_tool_call = ModelFeature.STREAM_TOOL_CALL in features

        # 如果模型支持视觉功能，则设置文件列表
        self.files = application_generate_entity.files if ModelFeature.VISION in features else []
        self.query: Optional[str] = ""
        self._current_thoughts: list[PromptMessage] = []

    def _repack_app_generate_entity(
        self, app_generate_entity: AgentChatAppGenerateEntity
    ) -> AgentChatAppGenerateEntity:
        """
        重新打包应用生成实体，确保简单提示模板不为空。

        Args:
            app_generate_entity (AgentChatAppGenerateEntity): 应用生成实体

        Returns:
            AgentChatAppGenerateEntity: 处理后的应用生成实体
        """
        # 如果简单提示模板为空，则设置为空字符串
        if app_generate_entity.app_config.prompt_template.simple_prompt_template is None:
            app_generate_entity.app_config.prompt_template.simple_prompt_template = ""

        return app_generate_entity

    def _convert_tool_to_prompt_message_tool(self, tool: AgentToolEntity) -> tuple[PromptMessageTool, Tool]:
        """
        将代理工具实体转换为提示消息工具格式。

        Args:
            tool (AgentToolEntity): 代理工具实体

        Returns:
            tuple[PromptMessageTool, Tool]: 包含提示消息工具和工具实体的元组
        """
        # 获取代理工具运行时实例
        tool_entity = ToolManager.get_agent_tool_runtime(
            tenant_id=self.tenant_id,
            app_id=self.app_config.app_id,
            agent_tool=tool,
            invoke_from=self.application_generate_entity.invoke_from,
        )
        # 确保工具实体有描述信息
        assert tool_entity.entity.description

        # 创建提示消息工具对象
        message_tool = PromptMessageTool(
            name=tool.tool_name,
            description=tool_entity.entity.description.llm,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

        # 获取工具的合并运行时参数
        parameters = tool_entity.get_merged_runtime_parameters()
        # 遍历参数，筛选出需要传递给LLM的参数
        for parameter in parameters:
            # 只处理表单类型为LLM的参数
            if parameter.form != ToolParameter.ToolParameterForm.LLM:
                continue

            # 获取参数类型
            parameter_type = parameter.type.as_normal_type()
            # 跳过系统文件、文件和文件列表类型的参数
            if parameter.type in {
                ToolParameter.ToolParameterType.SYSTEM_FILES,
                ToolParameter.ToolParameterType.FILE,
                ToolParameter.ToolParameterType.FILES,
            }:
                continue

            # 处理选择类型参数的枚举值
            enum = []
            if parameter.type == ToolParameter.ToolParameterType.SELECT:
                enum = [option.value for option in parameter.options] if parameter.options else []

            # 设置参数属性
            message_tool.parameters["properties"][parameter.name] = (
                {
                    "type": parameter_type,
                    "description": parameter.llm_description or "",
                }
                if parameter.input_schema is None
                else parameter.input_schema
            )

            # 如果有枚举值，则添加到参数属性中
            if len(enum) > 0:
                message_tool.parameters["properties"][parameter.name]["enum"] = enum

            # 如果参数是必需的，则添加到required列表中
            if parameter.required:
                message_tool.parameters["required"].append(parameter.name)

        return message_tool, tool_entity

    def _convert_dataset_retriever_tool_to_prompt_message_tool(self, tool: DatasetRetrieverTool) -> PromptMessageTool:
        """
        将数据集检索工具转换为提示消息工具格式。

        Args:
            tool (DatasetRetrieverTool): 数据集检索工具

        Returns:
            PromptMessageTool: 提示消息工具
        """
        # 确保工具实体有描述信息
        assert tool.entity.description

        # 创建提示消息工具对象
        prompt_tool = PromptMessageTool(
            name=tool.entity.identity.name,
            description=tool.entity.description.llm,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

        # 获取工具的运行时参数
        for parameter in tool.get_runtime_parameters():
            parameter_type = "string"

            # 设置参数属性
            prompt_tool.parameters["properties"][parameter.name] = {
                "type": parameter_type,
                "description": parameter.llm_description or "",
            }

            # 如果参数是必需的，则添加到required列表中
            if parameter.required:
                if parameter.name not in prompt_tool.parameters["required"]:
                    prompt_tool.parameters["required"].append(parameter.name)

        return prompt_tool

    def _init_prompt_tools(self) -> tuple[dict[str, Tool], list[PromptMessageTool]]:
        """
        初始化代理工具，包括应用配置中的工具和数据集工具。

        Returns:
            tuple[dict[str, Tool], list[PromptMessageTool]]: 包含工具实例字典和提示消息工具列表的元组
        """
        # 工具实例字典，用于存储工具名称和工具实例的映射关系
        tool_instances = {}
        # 提示消息工具列表，用于存储转换后的提示消息工具
        prompt_messages_tools = []

        # 处理应用配置中的代理工具
        for tool in self.app_config.agent.tools or [] if self.app_config.agent else []:
            try:
                # 将工具转换为提示消息工具格式
                prompt_tool, tool_entity = self._convert_tool_to_prompt_message_tool(tool)
            except Exception:
                # API工具可能已被删除，跳过该工具
                continue
            # 保存工具实体到工具实例字典中
            tool_instances[tool.tool_name] = tool_entity
            # 保存提示工具到提示消息工具列表中
            prompt_messages_tools.append(prompt_tool)

        # 转换数据集工具为ModelRuntime工具格式
        for dataset_tool in self.dataset_tools:
            # 将数据集检索工具转换为提示消息工具
            prompt_tool = self._convert_dataset_retriever_tool_to_prompt_message_tool(dataset_tool)
            # 保存提示工具到提示消息工具列表中
            prompt_messages_tools.append(prompt_tool)
            # 保存工具实体到工具实例字典中
            tool_instances[dataset_tool.entity.identity.name] = dataset_tool

        return tool_instances, prompt_messages_tools

    def update_prompt_message_tool(self, tool: Tool, prompt_tool: PromptMessageTool) -> PromptMessageTool:
        """
        更新提示消息工具的参数信息。

        Args:
            tool (Tool): 工具对象
            prompt_tool (PromptMessageTool): 提示消息工具

        Returns:
            PromptMessageTool: 更新后的提示消息工具
        """
        # 获取工具运行时参数
        tool_runtime_parameters = tool.get_runtime_parameters()

        # 遍历工具运行时参数并更新提示消息工具
        for parameter in tool_runtime_parameters:
            # 只处理表单类型为LLM的参数
            if parameter.form != ToolParameter.ToolParameterForm.LLM:
                continue

            # 获取参数类型
            parameter_type = parameter.type.as_normal_type()
            # 跳过系统文件、文件和文件列表类型的参数
            if parameter.type in {
                ToolParameter.ToolParameterType.SYSTEM_FILES,
                ToolParameter.ToolParameterType.FILE,
                ToolParameter.ToolParameterType.FILES,
            }:
                continue

            # 处理选择类型参数的枚举值
            enum = []
            if parameter.type == ToolParameter.ToolParameterType.SELECT:
                enum = [option.value for option in parameter.options] if parameter.options else []

            # 设置参数属性
            prompt_tool.parameters["properties"][parameter.name] = (
                {
                    "type": parameter_type,
                    "description": parameter.llm_description or "",
                }
                if parameter.input_schema is None
                else parameter.input_schema
            )

            # 如果有枚举值，则添加到参数属性中
            if len(enum) > 0:
                prompt_tool.parameters["properties"][parameter.name]["enum"] = enum

            # 如果参数是必需的且不在required列表中，则添加到required列表中
            if parameter.required:
                if parameter.name not in prompt_tool.parameters["required"]:
                    prompt_tool.parameters["required"].append(parameter.name)

        return prompt_tool

    def create_agent_thought(
        self, message_id: str, message: str, tool_name: str, tool_input: str, messages_ids: list[str]
    ) -> str:
        """
        创建代理思考记录，用于记录代理的思考过程。

        Args:
            message_id (str): 消息ID
            message (str): 消息内容
            tool_name (str): 使用的工具名称
            tool_input (str): 工具输入
            messages_ids (list[str]): 消息ID列表

        Returns:
            str: 创建的代理思考记录ID
        """
        # 创建代理思考记录对象
        thought = MessageAgentThought(
            message_id=message_id,
            message_chain_id=None,
            thought="",
            tool=tool_name,
            tool_labels_str="{}",
            tool_meta_str="{}",
            tool_input=tool_input,
            message=message,
            message_token=0,
            message_unit_price=0,
            message_price_unit=0,
            message_files=json.dumps(messages_ids) if messages_ids else "",
            answer="",
            observation="",
            answer_token=0,
            answer_unit_price=0,
            answer_price_unit=0,
            tokens=0,
            total_price=0,
            position=self.agent_thought_count + 1,
            currency="USD",
            latency=0,
            created_by_role="account",
            created_by=self.user_id,
        )

        # 将代理思考记录添加到数据库并提交
        db.session.add(thought)
        db.session.commit()
        # 获取代理思考记录ID
        agent_thought_id = str(thought.id)
        # 更新代理思考计数
        self.agent_thought_count += 1
        db.session.close()

        return agent_thought_id

    def save_agent_thought(
        self,
        agent_thought_id: str,
        tool_name: str | None,
        tool_input: Union[str, dict, None],
        thought: str | None,
        observation: Union[str, dict, None],
        tool_invoke_meta: Union[str, dict, None],
        answer: str | None,
        messages_ids: list[str],
        llm_usage: LLMUsage | None = None,
    ):
        """
        保存代理思考记录的详细信息。

        Args:
            agent_thought_id (str): 代理思考记录ID
            tool_name (str | None): 工具名称
            tool_input (Union[str, dict, None]): 工具输入
            thought (str | None): 思考内容
            observation (Union[str, dict, None]): 观察结果
            tool_invoke_meta (Union[str, dict, None]): 工具调用元数据
            answer (str | None): 回答内容
            messages_ids (list[str]): 消息ID列表
            llm_usage (LLMUsage | None): 大语言模型使用情况（可选）
        """
        # 查询代理思考记录
        stmt = select(MessageAgentThought).where(MessageAgentThought.id == agent_thought_id)
        agent_thought = db.session.scalar(stmt)
        # 如果未找到代理思考记录，则抛出异常
        if not agent_thought:
            raise ValueError("agent thought not found")

        # 更新思考内容
        if thought:
            agent_thought.thought += thought

        # 更新工具名称
        if tool_name:
            agent_thought.tool = tool_name

        # 更新工具输入
        if tool_input:
            if isinstance(tool_input, dict):
                try:
                    tool_input = json.dumps(tool_input, ensure_ascii=False)
                except Exception:
                    tool_input = json.dumps(tool_input)

            agent_thought.tool_input = tool_input

        # 更新观察结果
        if observation:
            if isinstance(observation, dict):
                try:
                    observation = json.dumps(observation, ensure_ascii=False)
                except Exception:
                    observation = json.dumps(observation)

            agent_thought.observation = observation

        # 更新回答内容
        if answer:
            agent_thought.answer = answer

        # 更新消息文件
        if messages_ids is not None and len(messages_ids) > 0:
            agent_thought.message_files = json.dumps(messages_ids)

        # 更新大语言模型使用情况
        if llm_usage:
            agent_thought.message_token = llm_usage.prompt_tokens
            agent_thought.message_price_unit = llm_usage.prompt_price_unit
            agent_thought.message_unit_price = llm_usage.prompt_unit_price
            agent_thought.answer_token = llm_usage.completion_tokens
            agent_thought.answer_price_unit = llm_usage.completion_price_unit
            agent_thought.answer_unit_price = llm_usage.completion_unit_price
            agent_thought.tokens = llm_usage.total_tokens
            agent_thought.total_price = llm_usage.total_price

        # 检查并更新工具标签
        labels = agent_thought.tool_labels or {}
        tools = agent_thought.tool.split(";") if agent_thought.tool else []
        for tool in tools:
            if not tool:
                continue
            if tool not in labels:
                tool_label = ToolManager.get_tool_label(tool)
                if tool_label:
                    labels[tool] = tool_label.to_dict()
                else:
                    labels[tool] = {"en_US": tool, "zh_Hans": tool}

        agent_thought.tool_labels_str = json.dumps(labels)

        # 更新工具调用元数据
        if tool_invoke_meta is not None:
            if isinstance(tool_invoke_meta, dict):
                try:
                    tool_invoke_meta = json.dumps(tool_invoke_meta, ensure_ascii=False)
                except Exception:
                    tool_invoke_meta = json.dumps(tool_invoke_meta)

            agent_thought.tool_meta_str = tool_invoke_meta

        # 提交更改并关闭数据库会话
        db.session.commit()
        db.session.close()

    def organize_agent_history(self, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        组织代理历史记录，包括系统消息、用户消息和代理思考记录。

        Args:
            prompt_messages (list[PromptMessage]): 提示消息列表

        Returns:
            list[PromptMessage]: 组织后的历史提示消息列表
        """
        # 初始化结果列表
        result: list[PromptMessage] = []

        # 检查提示消息列表中是否有系统消息，如果有则添加到结果列表中
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, SystemPromptMessage):
                result.append(prompt_message)

        # 查询当前对话中的所有消息并按创建时间倒序排列
        messages = (
            (
                db.session.execute(
                    select(Message)
                    .where(Message.conversation_id == self.message.conversation_id)
                    .order_by(Message.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        # 提取线程消息并反转顺序
        messages = list(reversed(extract_thread_messages(messages)))

        # 遍历消息列表，组织历史记录
        for message in messages:
            # 跳过当前消息
            if message.id == self.message.id:
                continue

            # 添加用户提示消息
            result.append(self.organize_agent_user_prompt(message))

            # 获取代理思考记录列表
            agent_thoughts: list[MessageAgentThought] = message.agent_thoughts
            if agent_thoughts:
                # 遍历代理思考记录
                for agent_thought in agent_thoughts:
                    tools = agent_thought.tool
                    # 如果有工具调用
                    if tools:
                        tools = tools.split(";")
                        # 初始化工具调用和工具调用响应列表
                        tool_calls: list[AssistantPromptMessage.ToolCall] = []
                        tool_call_response: list[ToolPromptMessage] = []

                        # 解析工具输入和响应
                        try:
                            tool_inputs = json.loads(agent_thought.tool_input)
                        except Exception:
                            tool_inputs = {tool: {} for tool in tools}
                        try:
                            tool_responses = json.loads(agent_thought.observation)
                        except Exception:
                            tool_responses = dict.fromkeys(tools, agent_thought.observation)

                        # 为每个工具创建工具调用和响应
                        for tool in tools:
                            # 生成工具调用ID
                            tool_call_id = str(uuid.uuid4())
                            # 添加工具调用
                            tool_calls.append(
                                AssistantPromptMessage.ToolCall(
                                    id=tool_call_id,
                                    type="function",
                                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                        name=tool,
                                        arguments=json.dumps(tool_inputs.get(tool, {})),
                                    ),
                                )
                            )
                            # 添加工具调用响应
                            tool_call_response.append(
                                ToolPromptMessage(
                                    content=tool_responses.get(tool, agent_thought.observation),
                                    name=tool,
                                    tool_call_id=tool_call_id,
                                )
                            )

                        # 扩展结果列表，添加代理思考和工具调用响应
                        result.extend(
                            [
                                AssistantPromptMessage(
                                    content=agent_thought.thought,
                                    tool_calls=tool_calls,
                                ),
                                *tool_call_response,
                            ]
                        )
                    # 如果没有工具调用，则只添加代理思考
                    if not tools:
                        result.append(AssistantPromptMessage(content=agent_thought.thought))
            else:
                # 如果没有代理思考记录但有回答，则添加回答
                if message.answer:
                    result.append(AssistantPromptMessage(content=message.answer))

        # 关闭数据库会话
        db.session.close()

        return result

    def organize_agent_user_prompt(self, message: Message) -> UserPromptMessage:
        """
        组织代理用户提示，处理包含文件的消息。

        Args:
            message (Message): 消息对象

        Returns:
            UserPromptMessage: 用户提示消息
        """
        # 查询消息关联的文件
        stmt = select(MessageFile).where(MessageFile.message_id == message.id)
        files = db.session.scalars(stmt).all()

        # 如果没有文件，则直接返回用户消息
        if not files:
            return UserPromptMessage(content=message.query)

        # 获取文件额外配置
        if message.app_model_config:
            file_extra_config = FileUploadConfigManager.convert(message.app_model_config.to_dict())
        else:
            file_extra_config = None

        # 如果没有文件额外配置，则直接返回用户消息
        if not file_extra_config:
            return UserPromptMessage(content=message.query)

        # 获取图像详细配置
        image_detail_config = file_extra_config.image_config.detail if file_extra_config.image_config else None
        image_detail_config = image_detail_config or ImagePromptMessageContent.DETAIL.LOW

        # 构建文件对象列表
        file_objs = file_factory.build_from_message_files(
            message_files=files, tenant_id=self.tenant_id, config=file_extra_config
        )

        # 如果没有文件对象，则直接返回用户消息
        if not file_objs:
            return UserPromptMessage(content=message.query)

        # 初始化提示消息内容列表
        prompt_message_contents: list[PromptMessageContentUnionTypes] = []

        # 为每个文件创建提示消息内容
        for file in file_objs:
            prompt_message_contents.append(
                file_manager.to_prompt_message_content(
                    file,
                    image_detail_config=image_detail_config,
                )
            )

        # 添加文本内容
        prompt_message_contents.append(TextPromptMessageContent(data=message.query))

        # 返回用户提示消息
        return UserPromptMessage(content=prompt_message_contents)
