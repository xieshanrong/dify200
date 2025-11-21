from collections.abc import Mapping, Sequence
from typing import Any, Literal

from sqlalchemy.orm import Session

from core.plugin.entities.parameters import PluginParameterOption
from core.plugin.impl.dynamic_select import DynamicSelectClient
from core.tools.tool_manager import ToolManager
from core.tools.utils.encryption import create_tool_provider_encrypter
from extensions.ext_database import db
from models.tools import BuiltinToolProvider


class PluginParameterService:
    @staticmethod
    def get_dynamic_select_options(
        tenant_id: str,
        user_id: str,
        plugin_id: str,
        provider: str,
        action: str,
        parameter: str,
        provider_type: Literal["tool"],
    ) -> Sequence[PluginParameterOption]:
        """
        获取插件参数的动态下拉选项。

        Args:
            tenant_id (str): 租户ID。
            user_id (str): 用户ID。
            plugin_id (str): 插件ID。
            provider (str): 提供者名称。
            action (str): 操作名称。
            parameter (str): 参数名称。
            provider_type (Literal["tool"]): 提供者类型，当前仅支持 "tool"。

        Returns:
            Sequence[PluginParameterOption]: 动态下拉选项列表。

        Raises:
            ValueError: 当提供者类型无效或找不到对应的内置提供者时抛出异常。
        """
        credentials: Mapping[str, Any] = {}

        # 根据不同的 provider_type 处理逻辑
        match provider_type:
            case "tool":
                # 获取工具提供者的控制器实例
                provider_controller = ToolManager.get_builtin_provider(provider, tenant_id)

                # 初始化加密器用于解密凭据
                encrypter, _ = create_tool_provider_encrypter(
                    tenant_id=tenant_id,
                    controller=provider_controller,
                )

                # 判断是否需要凭据信息
                if not provider_controller.need_credentials:
                    credentials = {}
                else:
                    # 从数据库中获取并解密凭据信息
                    with Session(db.engine) as session:
                        db_record = (
                            session.query(BuiltinToolProvider)
                            .where(
                                BuiltinToolProvider.tenant_id == tenant_id,
                                BuiltinToolProvider.provider == provider,
                            )
                            .first()
                        )

                    if db_record is None:
                        raise ValueError(f"Builtin provider {provider} not found when fetching credentials")

                    credentials = encrypter.decrypt(db_record.credentials)
            case _:
                raise ValueError(f"Invalid provider type: {provider_type}")

        # 调用客户端获取动态选择项
        return (
            DynamicSelectClient()
            .fetch_dynamic_select_options(tenant_id, user_id, plugin_id, provider, action, credentials, parameter)
            .options
        )
