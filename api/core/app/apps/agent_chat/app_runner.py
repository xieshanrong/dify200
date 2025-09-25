import logging
from typing import cast

from sqlalchemy import select

from core.agent.cot_chat_agent_runner import CotChatAgentRunner
from core.agent.cot_completion_agent_runner import CotCompletionAgentRunner
from core.agent.entities import AgentEntity
from core.agent.fc_agent_runner import FunctionCallAgentRunner
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfig
from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.base_app_runner import AppRunner
from core.app.entities.app_invoke_entities import AgentChatAppGenerateEntity
from core.app.entities.queue_entities import QueueAnnotationReplyEvent
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities.llm_entities import LLMMode
from core.model_runtime.entities.model_entities import ModelFeature, ModelPropertyKey
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.moderation.base import ModerationError
from extensions.ext_database import db
from models.model import App, Conversation, Message

logger = logging.getLogger(__name__)


class AgentChatAppRunner(AppRunner):
    """
    Agent Application Runner类

    该类负责运行Agent聊天应用，处理从用户输入到模型响应的整个流程。
    包括输入处理、内容审核、注释回复、外部数据工具集成、模型选择和代理执行等功能。
    """

    def run(
        self,
        application_generate_entity: AgentChatAppGenerateEntity,
        queue_manager: AppQueueManager,
        conversation: Conversation,
        message: Message,
    ):
        """
        运行Agent聊天应用的主要逻辑

        该方法处理应用执行的完整流程，包括：
        1. 获取应用配置和记录
        2. 初始化对话内存
        3. 组织提示消息
        4. 执行内容审核
        5. 处理注释回复
        6. 集成外部数据工具
        7. 检查托管内容审核
        8. 根据模型特性选择合适的代理策略
        9. 执行代理并处理结果

        Args:
            application_generate_entity (AgentChatAppGenerateEntity): 应用生成实体，
                包含应用配置、输入、查询等信息
            queue_manager (AppQueueManager): 应用队列管理器，用于发布事件和消息
            conversation (Conversation): 对话记录对象
            message (Message): 消息对象

        Returns:
            None: 该方法不返回值，而是通过队列管理器发布结果
        """
        # 获取应用配置并转换为Agent聊天应用配置类型
        app_config = application_generate_entity.app_config
        app_config = cast(AgentChatAppConfig, app_config)

        # 查询应用记录
        app_stmt = select(App).where(App.id == app_config.app_id)
        app_record = db.session.scalar(app_stmt)
        if not app_record:
            raise ValueError("App not found")

        # 提取输入、查询和文件信息
        inputs = application_generate_entity.inputs
        query = application_generate_entity.query
        files = application_generate_entity.files

        # 初始化对话内存
        memory = None
        if application_generate_entity.conversation_id:
            # 获取对话内存（只读）
            model_instance = ModelInstance(
                provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
                model=application_generate_entity.model_conf.model,
            )

            memory = TokenBufferMemory(conversation=conversation, model_instance=model_instance)

        # 组织所有输入和模板到提示消息
        # 包括：提示模板、输入、查询（可选）、文件（可选）、内存（可选）
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query,
            memory=memory,
        )

        # 内容审核处理
        try:
            # 处理敏感词规避
            _, inputs, query = self.moderation_for_inputs(
                app_id=app_record.id,
                tenant_id=app_config.tenant_id,
                app_generate_entity=application_generate_entity,
                inputs=dict(inputs),
                query=query or "",
                message_id=message.id,
            )
        except ModerationError as e:
            # 如果内容审核失败，直接输出错误信息
            self.direct_output(
                queue_manager=queue_manager,
                app_generate_entity=application_generate_entity,
                prompt_messages=prompt_messages,
                text=str(e),
                stream=application_generate_entity.stream,
            )
            return

        # 处理注释回复
        if query:
            # 查询应用注释以进行回复
            annotation_reply = self.query_app_annotations_to_reply(
                app_record=app_record,
                message=message,
                query=query,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from,
            )

            if annotation_reply:
                # 发布注释回复事件
                queue_manager.publish(
                    QueueAnnotationReplyEvent(message_annotation_id=annotation_reply.id),
                    PublishFrom.APPLICATION_MANAGER,
                )

                # 直接输出注释内容
                self.direct_output(
                    queue_manager=queue_manager,
                    app_generate_entity=application_generate_entity,
                    prompt_messages=prompt_messages,
                    text=annotation_reply.content,
                    stream=application_generate_entity.stream,
                )
                return

        # 从外部数据工具填充变量输入（如果存在）
        external_data_tools = app_config.external_data_variables
        if external_data_tools:
            inputs = self.fill_in_inputs_from_external_data_tools(
                tenant_id=app_record.tenant_id,
                app_id=app_record.id,
                external_data_tools=external_data_tools,
                inputs=inputs,
                query=query,
            )

        # 重新组织所有输入和模板到提示消息
        # 包括：提示模板、输入、查询（可选）、文件（可选）
        #      内存（可选）、外部数据、数据集上下文（可选）
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query or "",
            memory=memory,
        )

        # 检查托管内容审核
        hosting_moderation_result = self.check_hosting_moderation(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            prompt_messages=prompt_messages,
        )

        if hosting_moderation_result:
            return

        # 获取代理实体配置
        agent_entity = app_config.agent
        assert agent_entity is not None

        # 初始化模型实例
        model_instance = ModelInstance(
            provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
            model=application_generate_entity.model_conf.model,
        )

        # 组织提示消息
        prompt_message, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query or "",
            memory=memory,
        )

        # 根据LLM模型更改函数调用策略
        llm_model = cast(LargeLanguageModel, model_instance.model_type_instance)
        model_schema = llm_model.get_model_schema(model_instance.model, model_instance.credentials)
        if not model_schema:
            raise ValueError("Model schema not found")

        # 检查模型是否支持工具调用功能
        if {ModelFeature.MULTI_TOOL_CALL, ModelFeature.TOOL_CALL}.intersection(model_schema.features or []):
            agent_entity.strategy = AgentEntity.Strategy.FUNCTION_CALLING

        # 重新获取对话和消息记录
        conversation_stmt = select(Conversation).where(Conversation.id == conversation.id)
        conversation_result = db.session.scalar(conversation_stmt)
        if conversation_result is None:
            raise ValueError("Conversation not found")
        msg_stmt = select(Message).where(Message.id == message.id)
        message_result = db.session.scalar(msg_stmt)
        if message_result is None:
            raise ValueError("Message not found")
        db.session.close()

        # 根据代理策略选择合适的运行器类
        runner_cls: type[FunctionCallAgentRunner] | type[CotChatAgentRunner] | type[CotCompletionAgentRunner]
        # 启动代理运行器
        if agent_entity.strategy == AgentEntity.Strategy.CHAIN_OF_THOUGHT:
            # 检查LLM模式
            if model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.CHAT.value:
                runner_cls = CotChatAgentRunner
            elif model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.COMPLETION.value:
                runner_cls = CotCompletionAgentRunner
            else:
                raise ValueError(f"Invalid LLM mode: {model_schema.model_properties.get(ModelPropertyKey.MODE)}")
        elif agent_entity.strategy == AgentEntity.Strategy.FUNCTION_CALLING:
            runner_cls = FunctionCallAgentRunner
        else:
            raise ValueError(f"Invalid agent strategy: {agent_entity.strategy}")

        # 创建代理运行器实例
        runner = runner_cls(
            tenant_id=app_config.tenant_id,
            application_generate_entity=application_generate_entity,
            conversation=conversation_result,
            app_config=app_config,
            model_config=application_generate_entity.model_conf,
            config=agent_entity,
            queue_manager=queue_manager,
            message=message_result,
            user_id=application_generate_entity.user_id,
            memory=memory,
            prompt_messages=prompt_message,
            model_instance=model_instance,
        )

        # 执行代理运行器
        invoke_result = runner.run(
            message=message,
            query=query,
            inputs=inputs,
        )

        # 处理执行结果
        self._handle_invoke_result(
            invoke_result=invoke_result,
            queue_manager=queue_manager,
            stream=application_generate_entity.stream,
            agent=True,
        )
