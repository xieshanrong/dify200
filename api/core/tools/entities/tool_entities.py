import base64
import contextlib
import enum
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_serializer, field_validator, model_validator

from core.entities.provider_entities import ProviderConfig
from core.plugin.entities.parameters import (
    MCPServerParameterType,
    PluginParameter,
    PluginParameterOption,
    PluginParameterType,
    as_normal_type,
    cast_parameter_value,
    init_frontend_parameter,
)
from core.rag.entities.citation_metadata import RetrievalSourceMetadata
from core.tools.entities.common_entities import I18nObject
from core.tools.entities.constants import TOOL_SELECTOR_MODEL_IDENTITY


class ToolLabelEnum(Enum):
    """
    工具标签枚举类，用于对工具进行分类标记
    每个枚举值代表一种工具功能类别，便于工具的管理和检索
    """
    SEARCH = "search"               # 搜索类工具，如搜索引擎、数据库查询等
    IMAGE = "image"                 # 图像处理工具，如图像识别、编辑、生成等
    VIDEOS = "videos"               # 视频处理工具，如视频编辑、转码、分析等
    WEATHER = "weather"             # 天气查询工具，如天气预报、气候数据获取等
    FINANCE = "finance"             # 金融数据工具，如股票价格、汇率、财经资讯等
    DESIGN = "design"               # 设计类工具，如图形设计、UI设计辅助等
    TRAVEL = "travel"               # 出行旅游工具，如航班查询、酒店预订、路线规划等
    SOCIAL = "social"               # 社交媒体工具，如社交平台内容获取、分析等
    NEWS = "news"                   # 新闻资讯工具，如新闻聚合、实时资讯获取等
    MEDICAL = "medical"             # 医疗健康工具，如健康咨询、医疗数据查询等
    PRODUCTIVITY = "productivity"   # 生产力工具，如文档处理、任务管理等
    EDUCATION = "education"         # 教育学习工具，如在线课程、学习资料查询等
    BUSINESS = "business"           # 商业办公工具，如企业数据查询、办公自动化等
    ENTERTAINMENT = "entertainment" # 娱乐休闲工具，如游戏、音乐、影视等
    UTILITIES = "utilities"         # 实用工具，如计算器、单位转换、日常工具等
    RAG = "rag"                     # RAG（检索增强生成）相关工具，用于增强AI回答的准确性
    OTHER = "other"                 # 其他未分类工具


class ToolProviderType(enum.StrEnum):
    """
    工具提供者类型枚举类，定义了不同类型的工具来源
    用于区分工具是由插件、内置功能还是其他方式提供
    """

    PLUGIN = "plugin"                           # 第三方插件提供的工具，通过插件系统集成
    BUILT_IN = "builtin"                        # 系统内置工具，由平台原生提供
    WORKFLOW = "workflow"                       # 工作流定义的工具，通过工作流编排创建
    API = "api"                                 # 通过API接入的外部工具，如RESTful API
    APP = "app"                                 # 应用程序提供的工具，通常为独立应用集成
    DATASET_RETRIEVAL = "dataset-retrieval"     # 数据集检索工具，专门用于从数据集中检索信息
    MCP = "mcp"                                 # MCP协议支持的工具，遵循Model Context Protocol标准

    @classmethod
    def value_of(cls, value: str) -> "ToolProviderType":
        """
        根据字符串值获取对应的枚举实例

        :param value: 工具提供者类型的字符串表示
        :return: 对应的ToolProviderType枚举实例
        :raises ValueError: 当传入无效的类型值时抛出异常
        """
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"invalid mode value {value}")


class ApiProviderSchemaType(Enum):
    """
    API提供者模式类型枚举类，定义了不同的API规范格式
    用于识别和解析不同格式的API文档，以便正确地集成API工具
    """

    OPENAPI = "openapi"                 # OpenAPI 3.0规范，目前最主流的API描述规范
    SWAGGER = "swagger"                 # Swagger 2.0规范，OpenAPI的前身版本
    OPENAI_PLUGIN = "openai_plugin"     # OpenAI插件规范，专为OpenAI插件设计的格式
    OPENAI_ACTIONS = "openai_actions"   # OpenAI Actions规范，另一种OpenAI工具描述格式

    @classmethod
    def value_of(cls, value: str) -> "ApiProviderSchemaType":
        """
        根据字符串值获取对应的API模式类型枚举实例

        :param value: API模式类型的字符串表示
        :return: 对应的ApiProviderSchemaType枚举实例
        :raises ValueError: 当传入无效的模式值时抛出异常
        """
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"invalid mode value {value}")


class ApiProviderAuthType(Enum):
    """
    API提供者认证类型枚举类，定义了不同的API认证方式
    用于处理API调用时的身份验证机制，确保安全访问
    """

    NONE = "none"                      # 无需认证，公开API接口
    API_KEY_HEADER = "api_key_header"  # 通过HTTP头部传递API密钥进行认证
    API_KEY_QUERY = "api_key_query"    # 通过URL查询参数传递API密钥进行认证

    @classmethod
    def value_of(cls, value: str) -> "ApiProviderAuthType":
        """
        根据字符串值获取对应的认证类型枚举实例
        支持向后兼容处理，如将已废弃的'api_key'转换为'api_key_header'

        :param value: 认证类型的字符串表示
        :return: 对应的ApiProviderAuthType枚举实例
        :raises ValueError: 当传入无效的认证类型值时抛出异常
        """
        # 'api_key' 在PR #21656中已被弃用
        # 进行标准化处理以保持向后兼容性
        v = (value or "").strip().lower()
        if v == "api_key":
            v = cls.API_KEY_HEADER.value

        for mode in cls:
            if mode.value == v:
                return mode

        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"invalid mode value '{value}', expected one of: {valid}")


class ToolInvokeMessage(BaseModel):
    """
    工具调用消息类，定义了工具执行后返回的各种类型消息
    用于统一处理工具的不同输出格式，包括文本、JSON、二进制数据等
    """

    class TextMessage(BaseModel):
        """纯文本消息格式，用于返回简单的文本内容"""
        text: str

    class JsonMessage(BaseModel):
        """JSON对象消息格式，用于返回结构化数据"""
        json_object: dict

    class BlobMessage(BaseModel):
        """二进制数据消息格式，用于返回文件、图片等二进制内容"""
        blob: bytes

    class BlobChunkMessage(BaseModel):
        """分块传输的二进制数据消息格式，用于大文件的流式传输"""
        id: str = Field(..., description="The id of the blob")
        sequence: int = Field(..., description="The sequence of the chunk")
        total_length: int = Field(..., description="The total length of the blob")
        blob: bytes = Field(..., description="The blob data of the chunk")
        end: bool = Field(..., description="Whether the chunk is the last chunk")

    class FileMessage(BaseModel):
        """文件消息格式基类，用于返回文件相关信息"""
        pass

    class VariableMessage(BaseModel):
        """变量消息格式，用于在工具间传递变量值"""
        variable_name: str = Field(..., description="The name of the variable")
        variable_value: Any = Field(..., description="The value of the variable")
        stream: bool = Field(default=False, description="Whether the variable is streamed")

        @model_validator(mode="before")
        @classmethod
        def transform_variable_value(cls, values):
            """
            验证变量值的类型，只允许基本类型和列表
            确保变量值在传输过程中的安全性和兼容性

            :param values: 待验证的变量值字典
            :return: 验证通过后的值
            :raises ValueError: 当变量值类型不符合要求时抛出异常
            """
            value = values.get("variable_value")
            if not isinstance(value, dict | list | str | int | float | bool):
                raise ValueError("Only basic types and lists are allowed.")

            # 如果是流式传输，变量值必须是字符串类型
            if values.get("stream"):
                if not isinstance(value, str):
                    raise ValueError("When 'stream' is True, 'variable_value' must be a string.")

            return values

        @field_validator("variable_name", mode="before")
        @classmethod
        def transform_variable_name(cls, value: str) -> str:
            """
            验证变量名称的有效性，禁止使用保留关键字
            防止变量名与系统关键字冲突

            :param value: 变量名称
            :return: 验证通过的变量名称
            :raises ValueError: 当变量名是保留关键字时抛出异常
            """
            if value in {"json", "text", "files"}:
                raise ValueError(f"The variable name '{value}' is reserved.")
            return value

    class LogMessage(BaseModel):
        """日志消息格式，用于记录工具执行过程中的日志信息"""

        class LogStatus(Enum):
            """日志状态枚举，定义了日志的不同状态类型"""
            START = "start"      # 开始执行
            ERROR = "error"      # 执行出错
            SUCCESS = "success"  # 执行成功

        id: str
        label: str = Field(..., description="The label of the log")
        parent_id: Optional[str] = Field(default=None, description="Leave empty for root log")
        error: Optional[str] = Field(default=None, description="The error message")
        status: LogStatus = Field(..., description="The status of the log")
        data: Mapping[str, Any] = Field(..., description="Detailed log data")
        metadata: Optional[Mapping[str, Any]] = Field(default=None, description="The metadata of the log")

    class RetrieverResourceMessage(BaseModel):
        """检索资源消息格式，用于返回检索到的资源信息"""
        retriever_resources: list[RetrievalSourceMetadata] = Field(..., description="retriever resources")
        context: str = Field(..., description="context")

    class MessageType(Enum):
        """消息类型枚举，定义了工具返回消息的各种类型"""
        TEXT = "text"                               # 纯文本消息
        IMAGE = "image"                             # 图片消息
        LINK = "link"                               # 链接消息
        BLOB = "blob"                               # 二进制数据消息
        JSON = "json"                               # JSON对象消息
        IMAGE_LINK = "image_link"                   # 图片链接消息
        BINARY_LINK = "binary_link"                 # 二进制文件链接消息
        VARIABLE = "variable"                       # 变量消息
        FILE = "file"                               # 文件消息
        LOG = "log"                                 # 日志消息
        BLOB_CHUNK = "blob_chunk"                   # 二进制数据块消息
        RETRIEVER_RESOURCES = "retriever_resources" # 检索资源消息

    type: MessageType = MessageType.TEXT
    """
        消息类型，默认为文本类型
        plain text, image url or link url
    """
    message: (
        JsonMessage
        | TextMessage
        | BlobChunkMessage
        | BlobMessage
        | LogMessage
        | FileMessage
        | None
        | VariableMessage
        | RetrieverResourceMessage
    )
    meta: dict[str, Any] | None = None

    @field_validator("message", mode="before")
    @classmethod
    def decode_blob_message(cls, v):
        """
        在验证前解码Base64编码的二进制数据
        确保二进制数据能正确传输和处理

        :param v: 原始消息数据
        :return: 解码后的消息数据
        """
        if isinstance(v, dict) and "blob" in v:
            with contextlib.suppress(Exception):
                v["blob"] = base64.b64decode(v["blob"])
        return v

    @field_serializer("message")
    def serialize_message(self, v):
        """
        序列化消息时将二进制数据编码为Base64字符串
        确保二进制数据在网络传输中的完整性

        :param v: 消息对象
        :return: 序列化后的消息数据
        """
        if isinstance(v, self.BlobMessage):
            return {"blob": base64.b64encode(v.blob).decode("utf-8")}
        return v


class ToolInvokeMessageBinary(BaseModel):
    """
    工具调用二进制消息类，专门用于处理二进制文件消息
    包含MIME类型、URL和文件变量信息
    """
    mimetype: str = Field(..., description="The mimetype of the binary")
    url: str = Field(..., description="The url of the binary")
    file_var: Optional[dict[str, Any]] = None


class ToolParameter(PluginParameter):
    """
    工具参数类，继承自PluginParameter并重写了类型定义
    用于定义工具执行时需要的参数信息
    Overrides type
    """

    class ToolParameterType(enum.StrEnum):
        """
        工具参数类型枚举，从PluginParameterType中移除了TOOLS_SELECTOR类型
        定义了工具参数支持的各种数据类型
        removes TOOLS_SELECTOR from PluginParameterType
        """

        STRING = PluginParameterType.STRING.value                   # 字符串类型参数
        NUMBER = PluginParameterType.NUMBER.value                   # 数字类型参数
        BOOLEAN = PluginParameterType.BOOLEAN.value                 # 布尔类型参数
        SELECT = PluginParameterType.SELECT.value                   # 下拉选择类型参数
        SECRET_INPUT = PluginParameterType.SECRET_INPUT.value       # 密文输入类型参数
        FILE = PluginParameterType.FILE.value                       # 单文件类型参数
        FILES = PluginParameterType.FILES.value                     # 多文件类型参数
        APP_SELECTOR = PluginParameterType.APP_SELECTOR.value       # 应用选择器类型参数
        MODEL_SELECTOR = PluginParameterType.MODEL_SELECTOR.value   # 模型选择器类型参数
        ANY = PluginParameterType.ANY.value                         # 任意类型参数
        DYNAMIC_SELECT = PluginParameterType.DYNAMIC_SELECT.value   # 动态下拉选择类型参数

        # MCP对象和数组类型参数
        ARRAY = MCPServerParameterType.ARRAY.value                  # 数组类型参数（MCP协议支持）
        OBJECT = MCPServerParameterType.OBJECT.value                # 对象类型参数（MCP协议支持）

        # 已弃用，不应使用
        SYSTEM_FILES = PluginParameterType.SYSTEM_FILES.value       # 系统文件类型参数（已弃用）

        def as_normal_type(self):
            """将参数类型转换为普通类型表示"""
            return as_normal_type(self)

        def cast_value(self, value: Any):
            """根据参数类型转换值的类型"""
            return cast_parameter_value(self, value)

    class ToolParameterForm(Enum):
        """参数表单来源枚举，定义了参数在不同阶段的设置方式"""
        SCHEMA = "schema"  # 应在添加工具时设置，定义参数的基本结构
        FORM = "form"      # 应在调用工具前设置，由用户填写具体值
        LLM = "llm"        # 将由LLM设置，由AI模型自动填充

    type: ToolParameterType = Field(..., description="The type of the parameter")
    human_description: Optional[I18nObject] = Field(default=None, description="The description presented to the user")
    form: ToolParameterForm = Field(..., description="The form of the parameter, schema/form/llm")
    llm_description: Optional[str] = None
    # MCP对象和数组类型参数使用此字段存储模式定义
    input_schema: Optional[dict] = None

    @classmethod
    def get_simple_instance(
        cls,
        name: str,
        llm_description: str,
        typ: ToolParameterType,
        required: bool,
        options: Optional[list[str]] = None,
    ) -> "ToolParameter":
        """
        创建一个简单的工具参数实例
        用于快速创建基本的工具参数对象

        :param name: 参数的名称
        :param llm_description: 提供给LLM的参数描述
        :param typ: 参数的类型
        :param required: 参数是否为必需
        :param options: 参数的可选项列表（适用于选择类型参数）
        """
        # 将选项转换为ToolParameterOption对象
        if options:
            option_objs = [
                PluginParameterOption(value=option, label=I18nObject(en_US=option, zh_Hans=option))
                for option in options
            ]
        else:
            option_objs = []

        return cls(
            name=name,
            label=I18nObject(en_US="", zh_Hans=""),
            placeholder=None,
            human_description=I18nObject(en_US="", zh_Hans=""),
            type=typ,
            form=cls.ToolParameterForm.LLM,
            llm_description=llm_description,
            required=required,
            options=option_objs,
        )

    def init_frontend_parameter(self, value: Any):
        """初始化前端参数，用于前端界面展示"""
        return init_frontend_parameter(self, self.type, value)


class ToolProviderIdentity(BaseModel):
    """
    工具提供者身份信息类
    包含工具提供者的基本元数据信息
    """
    author: str = Field(..., description="The author of the tool")
    name: str = Field(..., description="The name of the tool")
    description: I18nObject = Field(..., description="The description of the tool")
    icon: str = Field(..., description="The icon of the tool")
    icon_dark: Optional[str] = Field(default=None, description="The dark icon of the tool")
    label: I18nObject = Field(..., description="The label of the tool")
    tags: Optional[list[ToolLabelEnum]] = Field(
        default=[],
        description="The tags of the tool",
    )


class ToolIdentity(BaseModel):
    """
    工具身份信息类
    包含工具的基本识别信息
    """
    author: str = Field(..., description="The author of the tool")
    name: str = Field(..., description="The name of the tool")
    label: I18nObject = Field(..., description="The label of the tool")
    provider: str = Field(..., description="The provider of the tool")
    icon: Optional[str] = None


class ToolDescription(BaseModel):
    """
    工具描述类
    分别定义面向用户和面向LLM的工具描述
    """
    human: I18nObject = Field(..., description="The description presented to the user")
    llm: str = Field(..., description="The description presented to the LLM")


class ToolEntity(BaseModel):
    """
    工具实体类
    定义了一个完整工具的所有信息，包括身份、参数、描述等
    """
    identity: ToolIdentity
    parameters: list[ToolParameter] = Field(default_factory=list)
    description: Optional[ToolDescription] = None
    output_schema: Optional[dict] = None
    has_runtime_parameters: bool = Field(default=False, description="Whether the tool has runtime parameters")

    # pydantic configs
    model_config = ConfigDict(protected_namespaces=())

    @field_validator("parameters", mode="before")
    @classmethod
    def set_parameters(cls, v, validation_info: ValidationInfo) -> list[ToolParameter]:
        """确保参数字段始终是列表类型"""
        return v or []


class OAuthSchema(BaseModel):
    """
    OAuth认证模式类
    定义OAuth客户端和凭证的模式结构
    """
    client_schema: list[ProviderConfig] = Field(default_factory=list, description="The schema of the OAuth client")
    credentials_schema: list[ProviderConfig] = Field(
        default_factory=list, description="The schema of the OAuth credentials"
    )


class ToolProviderEntity(BaseModel):
    """
    工具提供者实体类
    表示一个工具提供者的完整信息，包括身份、认证等
    """
    identity: ToolProviderIdentity
    plugin_id: Optional[str] = None
    credentials_schema: list[ProviderConfig] = Field(default_factory=list)
    oauth_schema: Optional[OAuthSchema] = None


class ToolProviderEntityWithPlugin(ToolProviderEntity):
    """
    带插件信息的工具提供者实体类
    扩展了基础工具提供者实体，增加了具体的工具列表
    """
    tools: list[ToolEntity] = Field(default_factory=list)


class WorkflowToolParameterConfiguration(BaseModel):
    """
    工作流工具参数配置类
    定义工作流中工具参数的配置信息
    Workflow tool configuration
    """

    name: str = Field(..., description="The name of the parameter")
    description: str = Field(..., description="The description of the parameter")
    form: ToolParameter.ToolParameterForm = Field(..., description="The form of the parameter")


class ToolInvokeMeta(BaseModel):
    """
    工具调用元数据类
    记录工具执行的元信息，如耗时、错误等
    Tool invoke meta
    """

    time_cost: float = Field(..., description="The time cost of the tool invoke")
    error: Optional[str] = None
    tool_config: Optional[dict] = None

    @classmethod
    def empty(cls) -> "ToolInvokeMeta":
        """
        获取一个空的工具调用元数据实例
        用于初始化或默认情况

        Get an empty instance of ToolInvokeMeta
        """
        return cls(time_cost=0.0, error=None, tool_config={})

    @classmethod
    def error_instance(cls, error: str) -> "ToolInvokeMeta":
        """
        获取一个带错误信息的工具调用元数据实例
        用于记录执行失败的情况

        Get an instance of ToolInvokeMeta with error
        """
        return cls(time_cost=0.0, error=error, tool_config={})

    def to_dict(self):
        """将元数据转换为字典格式，便于序列化和传输"""
        return {
            "time_cost": self.time_cost,
            "error": self.error,
            "tool_config": self.tool_config,
        }


class ToolLabel(BaseModel):
    """
    工具标签类
    用于展示工具的标签信息
    Tool label
    """

    name: str = Field(..., description="The name of the tool")
    label: I18nObject = Field(..., description="The label of the tool")
    icon: str = Field(..., description="The icon of the tool")


class ToolInvokeFrom(Enum):
    """
    工具调用来源枚举类
    定义工具被调用的不同场景
    Enum class for tool invoke
    """

    WORKFLOW = "workflow"  # 从工作流中调用
    AGENT = "agent"        # 从智能体中调用
    PLUGIN = "plugin"      # 从插件中调用


class ToolSelector(BaseModel):
    """
    工具选择器类
    用于在运行时动态选择和配置工具
    """
    dify_model_identity: str = TOOL_SELECTOR_MODEL_IDENTITY

    class Parameter(BaseModel):
        """工具参数定义类"""
        name: str = Field(..., description="The name of the parameter")
        type: ToolParameter.ToolParameterType = Field(..., description="The type of the parameter")
        required: bool = Field(..., description="Whether the parameter is required")
        description: str = Field(..., description="The description of the parameter")
        default: Optional[Union[int, float, str]] = None
        options: Optional[list[PluginParameterOption]] = None

    provider_id: str = Field(..., description="The id of the provider")
    credential_id: Optional[str] = Field(default=None, description="The id of the credential")
    tool_name: str = Field(..., description="The name of the tool")
    tool_description: str = Field(..., description="The description of the tool")
    tool_configuration: Mapping[str, Any] = Field(..., description="Configuration, type form")
    tool_parameters: Mapping[str, Parameter] = Field(..., description="Parameters, type llm")

    def to_plugin_parameter(self) -> dict[str, Any]:
        """将工具选择器转换为插件参数字典"""
        return self.model_dump()


class CredentialType(enum.StrEnum):
    """
    凭证类型枚举类
    定义工具认证支持的不同凭证类型
    """
    API_KEY = "api-key"   # API密钥认证
    OAUTH2 = "oauth2"     # OAuth2认证

    def get_name(self):
        """
        获取凭证类型的显示名称
        用于前端界面展示
        """
        if self == CredentialType.API_KEY:
            return "API KEY"
        elif self == CredentialType.OAUTH2:
            return "AUTH"
        else:
            return self.value.replace("-", " ").upper()

    def is_editable(self):
        """
        判断凭证是否可编辑
        API密钥类型可编辑，OAuth2类型通常不可直接编辑
        """
        return self == CredentialType.API_KEY

    def is_validate_allowed(self):
        """
        判断是否允许验证凭证
        目前仅API密钥类型支持验证
        """
        return self == CredentialType.API_KEY

    @classmethod
    def values(cls):
        """获取所有有效的凭证类型值列表"""
        return [item.value for item in cls]

    @classmethod
    def of(cls, credential_type: str) -> "CredentialType":
        """
        根据字符串获取对应的凭证类型枚举实例
        支持多种别名形式

        :param credential_type: 凭证类型的字符串表示
        :return: 对应的CredentialType枚举实例
        :raises ValueError: 当传入无效的凭证类型时抛出异常
        """
        type_name = credential_type.lower()
        if type_name in {"api-key", "api_key"}:
            return cls.API_KEY
        elif type_name in {"oauth2", "oauth"}:
            return cls.OAUTH2
        else:
            raise ValueError(f"Invalid credential type: {credential_type}")
