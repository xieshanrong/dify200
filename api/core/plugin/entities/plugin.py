import datetime
import enum
from collections.abc import Mapping
from typing import Any, Optional

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, field_validator, model_validator

from core.agent.plugin_entities import AgentStrategyProviderEntity
from core.datasource.entities.datasource_entities import DatasourceProviderEntity
from core.model_runtime.entities.provider_entities import ProviderEntity
from core.plugin.entities.base import BasePluginEntity
from core.plugin.entities.endpoint import EndpointProviderDeclaration
from core.tools.entities.common_entities import I18nObject
from core.tools.entities.tool_entities import ToolProviderEntity


class PluginInstallationSource(enum.StrEnum):
    """
    插件安装来源枚举类
    """
    Github = "github"
    Marketplace = "marketplace"
    Package = "package"
    Remote = "remote"


class PluginResourceRequirements(BaseModel):
    """
    插件资源需求类
    """
    memory: int

    class Permission(BaseModel):
        """
        插件权限类
        """
        class Tool(BaseModel):
            """
            工具权限配置
            """
            enabled: Optional[bool] = Field(default=False)

        class Model(BaseModel):
            """
            模型权限配置
            """
            enabled: Optional[bool] = Field(default=False)
            llm: Optional[bool] = Field(default=False)
            text_embedding: Optional[bool] = Field(default=False)
            rerank: Optional[bool] = Field(default=False)
            tts: Optional[bool] = Field(default=False)
            speech2text: Optional[bool] = Field(default=False)
            moderation: Optional[bool] = Field(default=False)

        class Node(BaseModel):
            """
            节点权限配置
            """
            enabled: Optional[bool] = Field(default=False)

        class Endpoint(BaseModel):
            """
            端点权限配置
            """
            enabled: Optional[bool] = Field(default=False)

        class Storage(BaseModel):
            """
            存储权限配置
            """
            enabled: Optional[bool] = Field(default=False)
            size: int = Field(ge=1024, le=1073741824, default=1048576)

        tool: Optional[Tool] = Field(default=None)
        model: Optional[Model] = Field(default=None)
        node: Optional[Node] = Field(default=None)
        endpoint: Optional[Endpoint] = Field(default=None)
        storage: Optional[Storage] = Field(default=None)

    permission: Optional[Permission] = Field(default=None)


class PluginCategory(enum.StrEnum):
    """
    插件分类枚举类
    """
    Tool = "tool"
    Model = "model"
    Extension = "extension"
    AgentStrategy = "agent-strategy"
    Datasource = "datasource"


class PluginDeclaration(BaseModel):
    """
    插件声明类
    """
    class Plugins(BaseModel):
        """
        插件组件集合类
        """
        tools: Optional[list[str]] = Field(default_factory=list[str])
        models: Optional[list[str]] = Field(default_factory=list[str])
        endpoints: Optional[list[str]] = Field(default_factory=list[str])
        datasources: Optional[list[str]] = Field(default_factory=list[str])

    class Meta(BaseModel):
        """
        插件元信息类
        """
        minimum_dify_version: Optional[str] = Field(default=None)
        version: Optional[str] = Field(default=None)

        @field_validator("minimum_dify_version")
        @classmethod
        def validate_minimum_dify_version(cls, v: Optional[str]) -> Optional[str]:
            """
            验证最低Dify版本号格式
            
            :param v: 版本号字符串
            :return: 验证通过的版本号
            :raises ValueError: 版本号格式无效时抛出异常
            """
            if v is None:
                return v
            try:
                Version(v)
                return v
            except InvalidVersion as e:
                raise ValueError(f"Invalid version format: {v}") from e

    version: str = Field(...)
    author: Optional[str] = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    name: str = Field(..., pattern=r"^[a-z0-9_-]{1,128}$")
    description: I18nObject
    icon: str
    icon_dark: Optional[str] = Field(default=None)
    label: I18nObject
    category: PluginCategory
    created_at: datetime.datetime
    resource: PluginResourceRequirements
    plugins: Plugins
    tags: list[str] = Field(default_factory=list)
    repo: Optional[str] = Field(default=None)
    verified: bool = Field(default=False)
    tool: Optional[ToolProviderEntity] = None
    model: Optional[ProviderEntity] = None
    endpoint: Optional[EndpointProviderDeclaration] = None
    agent_strategy: Optional[AgentStrategyProviderEntity] = None
    datasource: Optional[DatasourceProviderEntity] = None
    meta: Meta

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """
        验证插件版本号格式
        
        :param v: 版本号字符串
        :return: 验证通过的版本号
        :raises ValueError: 版本号格式无效时抛出异常
        """
        try:
            Version(v)
            return v
        except InvalidVersion as e:
            raise ValueError(f"Invalid version format: {v}") from e

    @model_validator(mode="before")
    @classmethod
    def validate_category(cls, values: dict):
        """
        根据提供的组件自动检测插件类别
        
        :param values: 模型字段值字典
        :return: 更新后的字段值字典
        """
        # auto detect category
        if values.get("tool"):
            values["category"] = PluginCategory.Tool
        elif values.get("model"):
            values["category"] = PluginCategory.Model
        elif values.get("datasource"):
            values["category"] = PluginCategory.Datasource
        elif values.get("agent_strategy"):
            values["category"] = PluginCategory.AgentStrategy
        else:
            values["category"] = PluginCategory.Extension
        return values


class PluginInstallation(BasePluginEntity):
    """
    插件安装信息类
    
    :param tenant_id: 租户ID
    :param endpoints_setups: 端点设置数量
    :param endpoints_active: 活跃端点数量
    :param runtime_type: 运行时类型
    :param source: 安装来源
    :param meta: 元数据映射
    :param plugin_id: 插件ID
    :param plugin_unique_identifier: 插件唯一标识符
    :param version: 插件版本
    :param checksum: 校验和
    :param declaration: 插件声明
    """
    tenant_id: str
    endpoints_setups: int
    endpoints_active: int
    runtime_type: str
    source: PluginInstallationSource
    meta: Mapping[str, Any]
    plugin_id: str
    plugin_unique_identifier: str
    version: str
    checksum: str
    declaration: PluginDeclaration


class PluginEntity(PluginInstallation):
    """
    插件实体类
    
    :param name: 插件名称
    :param installation_id: 安装ID
    :param version: 插件版本
    """
    name: str
    installation_id: str
    version: str

    @model_validator(mode="after")
    def set_plugin_id(self):
        """
        设置插件ID到工具声明中
        
        :return: 当前实例
        """
        if self.declaration.tool:
            self.declaration.tool.plugin_id = self.plugin_id
        return self


class PluginDependency(BaseModel):
    """
    插件依赖类
    """
    class Type(enum.StrEnum):
        """
        依赖类型枚举
        """
        Github = PluginInstallationSource.Github.value
        Marketplace = PluginInstallationSource.Marketplace.value
        Package = PluginInstallationSource.Package.value

    class Github(BaseModel):
        """
        GitHub依赖信息
        
        :param repo: 仓库地址
        :param version: 版本号
        :param package: 包名
        :param github_plugin_unique_identifier: GitHub插件唯一标识符
        """
        repo: str
        version: str
        package: str
        github_plugin_unique_identifier: str

        @property
        def plugin_unique_identifier(self) -> str:
            """
            获取插件唯一标识符
            
            :return: 插件唯一标识符
            """
            return self.github_plugin_unique_identifier

    class Marketplace(BaseModel):
        """
        市场依赖信息
        
        :param marketplace_plugin_unique_identifier: 市场插件唯一标识符
        """
        marketplace_plugin_unique_identifier: str

        @property
        def plugin_unique_identifier(self) -> str:
            """
            获取插件唯一标识符
            
            :return: 插件唯一标识符
            """
            return self.marketplace_plugin_unique_identifier

    class Package(BaseModel):
        """
        包依赖信息
        
        :param plugin_unique_identifier: 插件唯一标识符
        """
        plugin_unique_identifier: str

    type: Type
    value: Github | Marketplace | Package
    current_identifier: Optional[str] = None


class MissingPluginDependency(BaseModel):
    """
    缺失的插件依赖类
    
    :param plugin_unique_identifier: 插件唯一标识符
    :param current_identifier: 当前标识符
    """
    plugin_unique_identifier: str
    current_identifier: Optional[str] = None