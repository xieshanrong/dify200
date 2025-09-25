from collections.abc import Generator, Mapping
from typing import Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from controllers.service_api.wraps import create_or_update_end_user_for_user_id
from core.app.app_config.common.parameters_mapping import get_parameters_from_feature_dict
from core.app.apps.advanced_chat.app_generator import AdvancedChatAppGenerator
from core.app.apps.agent_chat.app_generator import AgentChatAppGenerator
from core.app.apps.chat.app_generator import ChatAppGenerator
from core.app.apps.completion.app_generator import CompletionAppGenerator
from core.app.apps.workflow.app_generator import WorkflowAppGenerator
from core.app.entities.app_invoke_entities import InvokeFrom
from core.plugin.backwards_invocation.base import BaseBackwardsInvocation
from extensions.ext_database import db
from models.account import Account
from models.model import App, AppMode, EndUser


class PluginAppBackwardsInvocation(BaseBackwardsInvocation):
    @classmethod
    def fetch_app_info(cls, app_id: str, tenant_id: str) -> Mapping:
        """
        获取应用信息

        Args:
            app_id (str): 应用ID
            tenant_id (str): 租户ID

        Returns:
            Mapping: 包含应用参数信息的字典
        """
        app = cls._get_app(app_id, tenant_id)

        # 根据应用模式获取参数
        if app.mode in {AppMode.ADVANCED_CHAT.value, AppMode.WORKFLOW.value}:
            workflow = app.workflow
            if workflow is None:
                raise ValueError("unexpected app type")

            features_dict = workflow.features_dict
            user_input_form = workflow.user_input_form(to_old_structure=True)
        else:
            app_model_config = app.app_model_config
            if app_model_config is None:
                raise ValueError("unexpected app type")

            features_dict = app_model_config.to_dict()

            user_input_form = features_dict.get("user_input_form", [])

        return {
            "data": get_parameters_from_feature_dict(features_dict=features_dict, user_input_form=user_input_form),
        }

    @classmethod
    def invoke_app(
        cls,
        app_id: str,
        user_id: str,
        tenant_id: str,
        conversation_id: Optional[str],
        query: Optional[str],
        stream: bool,
        inputs: Mapping,
        files: list[dict],
    ) -> Generator[Mapping | str, None, None] | Mapping:
        """
        调用应用

        Args:
            app_id (str): 应用ID
            user_id (str): 用户ID
            tenant_id (str): 租户ID
            conversation_id (Optional[str]): 对话ID
            query (Optional[str]): 查询内容
            stream (bool): 是否流式输出
            inputs (Mapping): 输入参数
            files (list[dict]): 文件列表

        Returns:
            Generator[Mapping | str, None, None] | Mapping: 应用执行结果
        """
        app = cls._get_app(app_id, tenant_id)
        if not user_id:
            user = create_or_update_end_user_for_user_id(app)
        else:
            user = cls._get_user(user_id)

        conversation_id = conversation_id or ""

        if app.mode in {AppMode.ADVANCED_CHAT.value, AppMode.AGENT_CHAT.value, AppMode.CHAT.value}:
            if not query:
                raise ValueError("missing query")

            return cls.invoke_chat_app(app, user, conversation_id, query, stream, inputs, files)
        elif app.mode == AppMode.WORKFLOW:
            return cls.invoke_workflow_app(app, user, stream, inputs, files)
        elif app.mode == AppMode.COMPLETION:
            return cls.invoke_completion_app(app, user, stream, inputs, files)

        raise ValueError("unexpected app type")

    @classmethod
    def invoke_chat_app(
        cls,
        app: App,
        user: Account | EndUser,
        conversation_id: str,
        query: str,
        stream: bool,
        inputs: Mapping,
        files: list[dict],
    ) -> Generator[Mapping | str, None, None] | Mapping:
        """
        调用聊天应用

        Args:
            app (App): 应用对象
            user (Account | EndUser): 用户对象
            conversation_id (str): 对话ID
            query (str): 查询内容
            stream (bool): 是否流式输出
            inputs (Mapping): 输入参数
            files (list[dict]): 文件列表

        Returns:
            Generator[Mapping | str, None, None] | Mapping: 应用执行结果
        """
        if app.mode == AppMode.ADVANCED_CHAT.value:
            workflow = app.workflow
            if not workflow:
                raise ValueError("unexpected app type")

            return AdvancedChatAppGenerator().generate(
                app_model=app,
                workflow=workflow,
                user=user,
                args={
                    "inputs": inputs,
                    "query": query,
                    "files": files,
                    "conversation_id": conversation_id,
                },
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=stream,
            )
        elif app.mode == AppMode.AGENT_CHAT.value:
            return AgentChatAppGenerator().generate(
                app_model=app,
                user=user,
                args={
                    "inputs": inputs,
                    "query": query,
                    "files": files,
                    "conversation_id": conversation_id,
                },
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=stream,
            )
        elif app.mode == AppMode.CHAT.value:
            return ChatAppGenerator().generate(
                app_model=app,
                user=user,
                args={
                    "inputs": inputs,
                    "query": query,
                    "files": files,
                    "conversation_id": conversation_id,
                },
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=stream,
            )
        else:
            raise ValueError("unexpected app type")

    @classmethod
    def invoke_workflow_app(
        cls,
        app: App,
        user: EndUser | Account,
        stream: bool,
        inputs: Mapping,
        files: list[dict],
    ) -> Generator[Mapping | str, None, None] | Mapping:
        """
        调用工作流应用

        Args:
            app (App): 应用对象
            user (EndUser | Account): 用户对象
            stream (bool): 是否流式输出
            inputs (Mapping): 输入参数
            files (list[dict]): 文件列表

        Returns:
            Generator[Mapping | str, None, None] | Mapping: 应用执行结果
        """
        workflow = app.workflow
        if not workflow:
            raise ValueError("unexpected app type")

        return WorkflowAppGenerator().generate(
            app_model=app,
            workflow=workflow,
            user=user,
            args={"inputs": inputs, "files": files},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=stream,
            call_depth=1,
        )

    @classmethod
    def invoke_completion_app(
        cls,
        app: App,
        user: EndUser | Account,
        stream: bool,
        inputs: Mapping,
        files: list[dict],
    ) -> Generator[Mapping | str, None, None] | Mapping:
        """
        调用补全应用

        Args:
            app (App): 应用对象
            user (EndUser | Account): 用户对象
            stream (bool): 是否流式输出
            inputs (Mapping): 输入参数
            files (list[dict]): 文件列表

        Returns:
            Generator[Mapping | str, None, None] | Mapping: 应用执行结果
        """
        return CompletionAppGenerator().generate(
            app_model=app,
            user=user,
            args={"inputs": inputs, "files": files},
            invoke_from=InvokeFrom.SERVICE_API,
            streaming=stream,
        )

    @classmethod
    def _get_user(cls, user_id: str) -> Union[EndUser, Account]:
        """
        根据用户ID获取用户

        Args:
            user_id (str): 用户ID

        Returns:
            Union[EndUser, Account]: 用户对象
        """
        with Session(db.engine, expire_on_commit=False) as session:
            stmt = select(EndUser).where(EndUser.id == user_id)
            user = session.scalar(stmt)
            if not user:
                stmt = select(Account).where(Account.id == user_id)
                user = session.scalar(stmt)

        if not user:
            raise ValueError("user not found")

        return user

    @classmethod
    def _get_app(cls, app_id: str, tenant_id: str) -> App:
        """
        获取应用

        Args:
            app_id (str): 应用ID
            tenant_id (str): 租户ID

        Returns:
            App: 应用对象
        """
        try:
            app = db.session.query(App).where(App.id == app_id).where(App.tenant_id == tenant_id).first()
        except Exception:
            raise ValueError("app not found")

        if not app:
            raise ValueError("app not found")

        return app

