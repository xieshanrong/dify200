from collections.abc import Generator
from typing import Any, Optional

from core.agent.entities import AgentInvokeMessage
from core.plugin.entities.plugin_daemon import (
    PluginAgentProviderEntity,
)
from core.plugin.entities.request import PluginInvokeContext
from core.plugin.impl.base import BasePluginClient
from core.plugin.utils.chunk_merger import merge_blob_chunks
from models.provider_ids import GenericProviderID


class PluginAgentClient(BasePluginClient):
    def fetch_agent_strategy_providers(self, tenant_id: str) -> list[PluginAgentProviderEntity]:
        """
        获取指定租户的代理策略提供商列表
        
        :param tenant_id: 租户ID
        :return: 代理策略提供商实体列表
        """

        def transformer(json_response: dict[str, Any]):
            """
            转换响应数据，为每个策略设置提供者名称
            
            :param json_response: 原始JSON响应
            :return: 转换后的JSON响应
            """
            for provider in json_response.get("data", []):
                declaration = provider.get("declaration", {}) or {}
                provider_name = declaration.get("identity", {}).get("name")
                for strategy in declaration.get("strategies", []):
                    strategy["identity"]["provider"] = provider_name

            return json_response

        response = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/agent_strategies",
            list[PluginAgentProviderEntity],
            params={"page": 1, "page_size": 256},
            transformer=transformer,
        )

        # 为每个提供商和策略设置完整的名称标识
        for provider in response:
            provider.declaration.identity.name = f"{provider.plugin_id}/{provider.declaration.identity.name}"

            # override the provider name for each tool to plugin_id/provider_name
            for strategy in provider.declaration.strategies:
                strategy.identity.provider = provider.declaration.identity.name

        return response

    def fetch_agent_strategy_provider(self, tenant_id: str, provider: str) -> PluginAgentProviderEntity:
        """
        获取指定租户和插件的工具提供商
        
        :param tenant_id: 租户ID
        :param provider: 提供商名称
        :return: 代理策略提供商实体
        """
        agent_provider_id = GenericProviderID(provider)

        def transformer(json_response: dict[str, Any]):
            """
            转换响应数据，为策略设置提供者名称
            
            :param json_response: 原始JSON响应
            :return: 转换后的JSON响应
            """
            # skip if error occurs
            if json_response.get("data") is None or json_response.get("data", {}).get("declaration") is None:
                return json_response

            for strategy in json_response.get("data", {}).get("declaration", {}).get("strategies", []):
                strategy["identity"]["provider"] = agent_provider_id.provider_name

            return json_response

        response = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/agent_strategy",
            PluginAgentProviderEntity,
            params={"provider": agent_provider_id.provider_name, "plugin_id": agent_provider_id.plugin_id},
            transformer=transformer,
        )

        response.declaration.identity.name = f"{response.plugin_id}/{response.declaration.identity.name}"

        # override the provider name for each tool to plugin_id/provider_name
        for strategy in response.declaration.strategies:
            strategy.identity.provider = response.declaration.identity.name

        return response

    def invoke(
        self,
        tenant_id: str,
        user_id: str,
        agent_provider: str,
        agent_strategy: str,
        agent_params: dict[str, Any],
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
        context: Optional[PluginInvokeContext] = None,
    ) -> Generator[AgentInvokeMessage, None, None]:
        """
        调用代理策略执行任务
        
        :param tenant_id: 租户ID
        :param user_id: 用户ID
        :param agent_provider: 代理提供商
        :param agent_strategy: 代理策略
        :param agent_params: 代理参数
        :param conversation_id: 对话ID，可选
        :param app_id: 应用ID，可选
        :param message_id: 消息ID，可选
        :param context: 插件调用上下文，可选
        :return: 代理调用消息生成器
        """

        agent_provider_id = GenericProviderID(agent_provider)

        response = self._request_with_plugin_daemon_response_stream(
            "POST",
            f"plugin/{tenant_id}/dispatch/agent_strategy/invoke",
            AgentInvokeMessage,
            data={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "app_id": app_id,
                "message_id": message_id,
                "context": context.model_dump() if context else {},
                "data": {
                    "agent_strategy_provider": agent_provider_id.provider_name,
                    "agent_strategy": agent_strategy,
                    "agent_strategy_params": agent_params,
                },
            },
            headers={
                "X-Plugin-ID": agent_provider_id.plugin_id,
                "Content-Type": "application/json",
            },
        )
        return merge_blob_chunks(response)