
import enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from core.entities.parameter_entities import CommonParameterType
from core.plugin.entities.parameters import (
    PluginParameter,
    as_normal_type,
    cast_parameter_value,
    init_frontend_parameter,
)
from core.tools.entities.common_entities import I18nObject
from core.tools.entities.tool_entities import (
    ToolIdentity,
    ToolProviderIdentity,
)


class AgentStrategyProviderIdentity(ToolProviderIdentity):
    """
    继承自 ToolProviderIdentity，未添加任何额外字段。
    用于标识 Agent 策略提供者的身份信息。
    """


class AgentStrategyParameter(PluginParameter):
    class AgentStrategyParameterType(enum.StrEnum):
        """
        参数类型枚举，继承自 PluginParameterType。
        用于定义 Agent 策略中支持的各种参数类型。
        """

        STRING = CommonParameterType.STRING.value
        NUMBER = CommonParameterType.NUMBER.value
        BOOLEAN = CommonParameterType.BOOLEAN.value
        SELECT = CommonParameterType.SELECT.value
        SECRET_INPUT = CommonParameterType.SECRET_INPUT.value
        FILE = CommonParameterType.FILE.value
        FILES = CommonParameterType.FILES.value
        APP_SELECTOR = CommonParameterType.APP_SELECTOR.value
        MODEL_SELECTOR = CommonParameterType.MODEL_SELECTOR.value
        TOOLS_SELECTOR = CommonParameterType.TOOLS_SELECTOR.value
        ANY = CommonParameterType.ANY.value

        # 已弃用，不应再使用
        SYSTEM_FILES = CommonParameterType.SYSTEM_FILES.value

        def as_normal_type(self):
            """
            将当前枚举值转换为标准类型。

            :return: 标准类型表示
            """
            return as_normal_type(self)

        def cast_value(self, value: Any):
            """
            将给定值转换为当前参数类型对应的值。

            :param value: 待转换的原始值
            :return: 转换后的值
            """
            return cast_parameter_value(self, value)

    type: AgentStrategyParameterType = Field(..., description="参数的类型")
    help: Optional[I18nObject] = None

    def init_frontend_parameter(self, value: Any):
        """
        初始化前端使用的参数对象。

        :param value: 参数的初始值
        :return: 前端参数对象
        """
        return init_frontend_parameter(self, self.type, value)


class AgentStrategyProviderEntity(BaseModel):
    """
    表示一个 Agent 策略提供者实体。

    :param identity: 提供者身份信息
    :param plugin_id: 插件 ID（可选）
    """
    identity: AgentStrategyProviderIdentity
    plugin_id: Optional[str] = Field(None, description="插件的 ID")


class AgentStrategyIdentity(ToolIdentity):
    """
    继承自 ToolIdentity，未添加任何额外字段。
    用于标识 Agent 策略的身份信息。
    """


class AgentFeature(enum.StrEnum):
    """
    Agent 功能枚举，用于描述 Agent 策略支持的功能特性。
    """

    HISTORY_MESSAGES = "history-messages"


class AgentStrategyEntity(BaseModel):
    """
    表示一个 Agent 策略实体。

    :param identity: 策略身份信息
    :param parameters: 策略参数列表，默认为空列表
    :param description: 策略描述信息
    :param output_schema: 输出结构定义（可选）
    :param features: 支持的功能列表（可选）
    :param meta_version: 元数据版本号（可选）
    """
    identity: AgentStrategyIdentity
    parameters: list[AgentStrategyParameter] = Field(default_factory=list)
    description: I18nObject = Field(..., description="Agent 策略的描述信息")
    output_schema: Optional[dict] = None
    features: Optional[list[AgentFeature]] = None
    meta_version: Optional[str] = None
    # pydantic 配置
    model_config = ConfigDict(protected_namespaces=())

    @field_validator("parameters", mode="before")
    @classmethod
    def set_parameters(cls, v, validation_info: ValidationInfo) -> list[AgentStrategyParameter]:
        """
        参数验证器，确保 parameters 字段不会为 None。

        :param v: 待验证的参数值
        :param validation_info: 验证上下文信息
        :return: 处理后的参数列表
        """
        return v or []


class AgentProviderEntityWithPlugin(AgentStrategyProviderEntity):
    """
    带有策略列表的 Agent 提供者实体。

    :param strategies: 策略实体列表，默认为空列表
    """
    strategies: list[AgentStrategyEntity] = Field(default_factory=list)
