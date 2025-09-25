from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from core.agent.plugin_entities import AgentProviderEntityWithPlugin
from core.datasource.entities.datasource_entities import DatasourceProviderEntityWithPlugin
from core.model_runtime.entities.model_entities import AIModelEntity
from core.model_runtime.entities.provider_entities import ProviderEntity
from core.plugin.entities.base import BasePluginEntity
from core.plugin.entities.parameters import PluginParameterOption
from core.plugin.entities.plugin import PluginDeclaration, PluginEntity
from core.tools.entities.common_entities import I18nObject
from core.tools.entities.tool_entities import ToolProviderEntityWithPlugin

T = TypeVar("T", bound=(BaseModel | dict | list | bool | str))

class PluginDaemonBasicResponse(BaseModel, Generic[T]):
    """基本的插件守护进程响应类。

    包含状态码、状态信息和响应数据。

    Attributes:
        code: 整数状态码。
        message: 状态信息字符串。
        data: 响应数据，类型为T。
    """

    code: int
    message: str
    data: Optional[T]

class InstallPluginMessage(BaseModel):
    """安装插件的消息类。

    包含消息类型和消息数据。

    Attributes:
        event: 消息类型，可以是'info'、'done'或'error'。
        data: 消息内容字符串。
    """

    class Event(StrEnum):
        Info = "info"
        Done = "done"
        Error = "error"

    event: Event
    data: str

class PluginToolProviderEntity(BaseModel):
    """插件工具提供程序实体类。

    描述工具提供程序的详细信息。

    Attributes:
        provider: 提供程序的名称。
        plugin_unique_identifier: 插件的唯一标识符。
        plugin_id: 插件的ID。
        declaration: 工具提供程序的声明。
    """

    provider: str
    plugin_unique_identifier: str
    plugin_id: str
    declaration: ToolProviderEntityWithPlugin

class PluginDatasourceProviderEntity(BaseModel):
    """插件数据源提供程序实体类。

    描述数据源提供程序的详细信息。

    Attributes:
        provider: 提供程序的名称。
        plugin_unique_identifier: 插件的唯一标识符。
        plugin_id: 插件的ID。
        is_authorized: 标识提供程序是否授权。
        declaration: 数据源提供程序的声明。
    """

    provider: str
    plugin_unique_identifier: str
    plugin_id: str
    is_authorized: bool = False
    declaration: DatasourceProviderEntityWithPlugin

class PluginAgentProviderEntity(BaseModel):
    """插件代理提供程序实体类。

    描述代理提供程序的详细信息。

    Attributes:
        provider: 提供程序的名称。
        plugin_unique_identifier: 插件的唯一标识符。
        plugin_id: 插件的ID。
        declaration: 代理提供程序的声明。
        meta: 插件声明的元数据。
    """

    provider: str
    plugin_unique_identifier: str
    plugin_id: str
    declaration: AgentProviderEntityWithPlugin
    meta: PluginDeclaration.Meta

class PluginBasicBooleanResponse(BaseModel):
    """基本的布尔响应类。

    包含布尔结果和可选的凭据。

    Attributes:
        result: 布尔值，表示操作是否成功。
        credentials: 可选的凭据字典。
    """

    result: bool
    credentials: dict | None = None

class PluginModelSchemaEntity(BaseModel):
    """模型架构实体类。

    包含模型的架构信息。

    Attributes:
        model_schema: 描述模型架构的AIModelEntity对象。
    """

    model_schema: AIModelEntity = Field(description="The model schema.")

    # pydantic 配置，保护命名空间设置为空
    model_config = ConfigDict(protected_namespaces=())

class PluginModelProviderEntity(BaseModel):
    """模型提供程序实体类。

    包含模型提供程序的详细信息。

    Attributes:
        id: 提供程序的ID。
        created_at: 提供程序的创建时间。
        updated_at: 提供程序的更新时间。
        provider: 提供程序的名称。
        tenant_id: 租户ID。
        plugin_unique_identifier: 插件的唯一标识符。
        plugin_id: 插件的ID。
        declaration: 提供程序的声明。
    """

    id: str = Field(description="ID")
    created_at: datetime = Field(description="The created at time of the model provider.")
    updated_at: datetime = Field(description="The updated at time of the model provider.")
    provider: str = Field(description="The provider of the model.")
    tenant_id: str = Field(description="The tenant ID.")
    plugin_unique_identifier: str = Field(description="The plugin unique identifier.")
    plugin_id: str = Field(description="The plugin ID.")
    declaration: ProviderEntity = Field(description="The declaration of the model provider.")

class PluginTextEmbeddingNumTokensResponse(BaseModel):
    """文本嵌入数量响应类。

    包含嵌入的标记数量列表。

    Attributes:
        num_tokens: 标记数量的列表。
    """

    num_tokens: list[int] = Field(description="The number of tokens.")

class PluginLLMNumTokensResponse(BaseModel):
    """大语言模型嵌入数量响应类。

    包含嵌入的标记数量。

    Attributes:
        num_tokens: 标记数量。
    """

    num_tokens: int = Field(description="The number of tokens.")

class PluginStringResultResponse(BaseModel):
    """字符串结果响应类。

    包含字符串结果。

    Attributes:
        result: 结果字符串。
    """

    result: str = Field(description="The result of the string.")

class PluginVoiceEntity(BaseModel):
    """语音实体类。

    包含语音的名称和值。

    Attributes:
        name: 语音的名称。
        value: 语音的值。
    """

    name: str = Field(description="The name of the voice.")
    value: str = Field(description="The value of the voice.")

class PluginVoicesResponse(BaseModel):
    """语音响应类。

    包含语音实体列表。

    Attributes:
        voices: 语音实体的列表。
    """

    voices: list[PluginVoiceEntity] = Field(description="The result of the voices.")

class PluginDaemonError(BaseModel):
    """插件守护进程错误类。

    包含错误类型和错误消息。

    Attributes:
        error_type: 错误类型字符串。
        message: 错误消息字符串。
    """

    error_type: str
    message: str

class PluginDaemonInnerError(Exception):
    """插件守护进程内部错误异常类。

    表示插件守护进程内部发生的错误。

    Attributes:
        code: 错误代码整数。
        message: 错误消息字符串。
    """

    code: int
    message: str

    def __init__(self, code: int, message: str):
        """初始化PluginDaemonInnerError异常。

        Args:
            code: 错误代码。
            message: 错误消息。
        """
        self.code = code
        self.message = message

class PluginInstallTaskStatus(StrEnum):
    """插件安装任务状态枚举类。

    定义了插件安装任务的可能状态。

    Attributes:
        Pending: 任务待处理。
        Running: 任务正在运行。
        Success: 任务完成。
        Failed: 任务失败。
    """
    Pending = "pending"
    Running = "running"
    Success = "success"
    Failed = "failed"

class PluginInstallTaskPluginStatus(BaseModel):
    """单个插件安装任务状态实体类。

    包含插件的唯一标识符、ID、状态、消息、图标和标签。

    Attributes:
        plugin_unique_identifier: 插件的唯一标识符。
        plugin_id: 插件的ID。
        status: 插件安装任务的状态。
        message: 状态消息。
        icon: 插件图标。
        labels: 插件标签，I18nObject类型。
    """

    plugin_unique_identifier: str = Field(description="The plugin unique identifier of the install task.")
    plugin_id: str = Field(description="The plugin ID of the install task.")
    status: PluginInstallTaskStatus = Field(description="The status of the install task.")
    message: str = Field(description="The message of the install task.")
    icon: str = Field(description="The icon of the plugin.")
    labels: I18nObject = Field(description="The labels of the plugin.")

class PluginInstallTask(BasePluginEntity):
    """插件安装任务实体类。

    包含安装任务的总体状态和详细信息。

    Attributes:
        status: 安装任务的当前状态。
        total_plugins: 待安装插件的总数。
        completed_plugins: 已安装插件的数量。
        plugins: 插件安装任务状态的列表。
    """

    status: PluginInstallTaskStatus = Field(description="The status of the install task.")
    total_plugins: int = Field(description="The total number of plugins to be installed.")
    completed_plugins: int = Field(description="The number of plugins that have been installed.")
    plugins: list[PluginInstallTaskPluginStatus] = Field(description="The status of the plugins.")

class PluginInstallTaskStartResponse(BaseModel):
    """插件安装任务启动响应类。

    包含安装任务的启动状态和任务ID。

    Attributes:
        all_installed: 标识所有插件是否已安装。
        task_id: 安装任务的ID。
    """

    all_installed: bool = Field(description="Whether all plugins are installed.")
    task_id: str = Field(description="The ID of the install task.")

class PluginVerification(BaseModel):
    """插件验证类。

    包含插件的授权类别。

    Attributes:
        authorized_category: 插件的授权类别，可以是'langgenius'、'partner'或'community'。
    """

    class AuthorizedCategory(StrEnum):
        Langgenius = "langgenius"
        Partner = "partner"
        Community = "community"

    authorized_category: AuthorizedCategory = Field(description="The authorized category of the plugin.")

class PluginDecodeResponse(BaseModel):
    """插件解码响应类。

    包含插件的唯一标识符、声明和可选的验证信息。

    Attributes:
        unique_identifier: 插件的唯一标识符。
        manifest: 插件的声明。
        verification: 可选的验证信息，PluginVerification类型。
    """

    unique_identifier: str = Field(description="The unique identifier of the plugin.")
    manifest: PluginDeclaration
    verification: Optional[PluginVerification] = Field(default=None, description="Basic verification information")

class PluginOAuthAuthorizationUrlResponse(BaseModel):
    """OAuth授权URL响应类。

    包含授权的URL。

    Attributes:
        authorization_url: 授权的URL字符串。
    """

    authorization_url: str = Field(description="The URL of the authorization.")

class PluginOAuthCredentialsResponse(BaseModel):
    """OAuth凭据响应类。

    包含OAuth的元数据、过期时间和凭据。

    Attributes:
        metadata: OAuth的元数据，如头像URL、名称等。
        expires_at: 凭据的过期时间，UTC时间戳。
        credentials: OAuth的凭据字典。
    """

    metadata: Mapping[str, Any] = Field(
        default_factory=dict, description="The metadata of the OAuth, like avatar url, name, etc."
    )
    expires_at: int = Field(default=-1, description="The expires at time of the credentials. UTC timestamp.")
    credentials: Mapping[str, Any] = Field(description="The credentials of the OAuth.")

class PluginListResponse(BaseModel):
    """插件列表响应类。

    包含插件列表和总数。

    Attributes:
        list: 插件实体列表。
        total: 插件的总数。
    """

    list: list[PluginEntity]
    total: int

class PluginDynamicSelectOptionsResponse(BaseModel):
    """动态选择选项响应类。

    包含动态选择的选项列表。

    Attributes:
        options: 插件参数选项的列表。
    """

    options: Sequence[PluginParameterOption] = Field(description="The options of the dynamic select.")
