import hashlib
import logging
import os
from collections.abc import Sequence
from threading import Lock
from typing import Optional

from pydantic import BaseModel

import contexts
from core.helper.position_helper import get_provider_position_map, sort_to_dict_by_position_map
from core.model_runtime.entities.model_entities import AIModelEntity, ModelType
from core.model_runtime.entities.provider_entities import ProviderConfig, ProviderEntity, SimpleProviderEntity
from core.model_runtime.model_providers.__base.ai_model import AIModel
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.moderation_model import ModerationModel
from core.model_runtime.model_providers.__base.rerank_model import RerankModel
from core.model_runtime.model_providers.__base.speech2text_model import Speech2TextModel
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.model_runtime.model_providers.__base.tts_model import TTSModel
from core.model_runtime.schema_validators.model_credential_schema_validator import ModelCredentialSchemaValidator
from core.model_runtime.schema_validators.provider_credential_schema_validator import ProviderCredentialSchemaValidator
from core.plugin.entities.plugin_daemon import PluginModelProviderEntity
from models.provider_ids import ModelProviderID

logger = logging.getLogger(__name__)

# 定义一个扩展模型提供者的类，继承自Pydantic模型
class ModelProviderExtension(BaseModel):
    # 插件模型提供者实体
    plugin_model_provider_entity: PluginModelProviderEntity
    # 可选的位置参数，默认为None
    position: Optional[int] = None

# 定义一个提供者工厂类
class ModelProviderFactory:
    # 用于存储提供者位置映射的字典
    provider_position_map: dict[str, int]

    # 初始化方法，接收租户ID
    def __init__(self, tenant_id: str) -> None:
        # 导入插件模型客户端
        from core.plugin.impl.model import PluginModelClient

        # 初始化提供者位置映射字典为空
        self.provider_position_map = {}

        # 设置租户ID和插件模型管理器
        self.tenant_id = tenant_id
        self.plugin_model_manager = PluginModelClient()

        # 如果提供者位置映射字典为空，则初始化它
        if not self.provider_position_map:
            # 获取当前文件的绝对路径
            current_path = os.path.abspath(__file__)
            # 获取模型提供者路径
            model_providers_path = os.path.dirname(current_path)
            # 获取位置映射文件路径并更新字典
            self.provider_position_map = get_provider_position_map(model_providers_path)

    # 获取所有提供者的方法
    def get_providers(self) -> Sequence[ProviderEntity]:
        """
        获取所有提供者
        :return: 提供者列表
        """
        # 获取插件模型提供者
        plugin_providers = self.get_plugin_model_providers()

        # 将插件模型提供者实体转换为扩展类实例
        model_provider_extensions = []
        for provider in plugin_providers:
            model_provider_extensions.append(ModelProviderExtension(plugin_model_provider_entity=provider))

        # 根据位置映射对扩展类进行排序
        sorted_extensions = sort_to_dict_by_position_map(
            position_map=self.provider_position_map,
            data=model_provider_extensions,
            name_func=lambda x: x.plugin_model_provider_entity.declaration.provider,
        )

        # 返回排序后的提供者声明
        return [extension.plugin_model_provider_entity.declaration for extension in sorted_extensions.values()]

    # 获取所有插件模型提供者的方法
    def get_plugin_model_providers(self) -> Sequence["PluginModelProviderEntity"]:
        """
        获取所有插件模型提供者
        :return: 插件模型提供者列表
        """
        # 检查上下文是否已设置插件模型提供者
        try:
            contexts.plugin_model_providers.get()
        except LookupError:
            # 如果未设置，则初始化为空列表并设置锁
            contexts.plugin_model_providers.set(None)
            contexts.plugin_model_providers_lock.set(Lock())

        # 使用锁访问上下文中的插件模型提供者
        with contexts.plugin_model_providers_lock.get():
            plugin_model_providers = contexts.plugin_model_providers.get()
            if plugin_model_providers is not None:
                return plugin_model_providers

            # 如果上下文为空，则从远程获取插件模型提供者
            plugin_model_providers = []
            contexts.plugin_model_providers.set(plugin_model_providers)

            # 调用插件模型管理器获取提供者列表
            plugin_providers = self.plugin_model_manager.fetch_model_providers(self.tenant_id)

            # 更新提供者的声明信息
            for provider in plugin_providers:
                provider.declaration.provider = provider.plugin_id + "/" + provider.declaration.provider
                plugin_model_providers.append(provider)

            return plugin_model_providers

    # 根据提供者名称获取提供者模式的方法
    def get_provider_schema(self, provider: str) -> ProviderEntity:
        """
        获取指定提供者的模式
        :param provider: 提供者名称
        :return: 提供者模式
        """
        # 获取指定提供者的插件模型提供者实体
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)
        # 返回提供者的声明模式
        return plugin_model_provider_entity.declaration

    # 根据提供者名称获取插件模型提供者实体的方法
    def get_plugin_model_provider(self, provider: str) -> "PluginModelProviderEntity":
        """
        获取指定名称的插件模型提供者实体
        :param provider: 提供者名称
        :return: 插件模型提供者实体
        """
        # 如果提供者名称中不包含'/'，则将其转换为ModelProviderID格式
        if "/" not in provider:
            provider = str(ModelProviderID(provider))

        # 获取所有插件模型提供者实体
        plugin_model_provider_entities = self.get_plugin_model_providers()

        # 通过提供者名称查找对应的插件模型提供者实体
        plugin_model_provider_entity = next(
            (p for p in plugin_model_provider_entities if p.declaration.provider == provider),
            None,
        )

        # 如果未找到提供者，则抛出异常
        if not plugin_model_provider_entity:
            raise ValueError(f"Invalid provider: {provider}")

        return plugin_model_provider_entity

    # 验证提供者凭证的方法
    def provider_credentials_validate(self, *, provider: str, credentials: dict):
        """
        验证提供者的凭证
        :param provider: 提供者名称
        :param credentials: 凭证字典
        :return: 过滤后的凭证
        """
        # 获取指定提供者的插件模型提供者实体
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)

        # 获取提供者的凭证模式
        provider_credential_schema = plugin_model_provider_entity.declaration.provider_credential_schema
        # 如果提供者凭证模式不存在，则抛出异常
        if not provider_credential_schema:
            raise ValueError(f"Provider {provider} does not have provider_credential_schema")

        # 创建验证器并验证凭证
        validator = ProviderCredentialSchemaValidator(provider_credential_schema)
        filtered_credentials = validator.validate_and_filter(credentials)

        # 调用插件模型管理器进行凭证验证
        self.plugin_model_manager.validate_provider_credentials(
            tenant_id=self.tenant_id,
            user_id="unknown",
            plugin_id=plugin_model_provider_entity.plugin_id,
            provider=plugin_model_provider_entity.provider,
            credentials=filtered_credentials,
        )

        return filtered_credentials

    # 验证模型凭证的方法
    def model_credentials_validate(self, *, provider: str, model_type: ModelType, model: str, credentials: dict):
        """
        验证指定模型的凭证
        :param provider: 提供者名称
        :param model_type: 模型类型
        :param model: 模型名称
        :param credentials: 凭证字典
        :return: 过滤后的凭证
        """
        # 获取指定提供者的插件模型提供者实体
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)

        # 获取模型的凭证模式
        model_credential_schema = plugin_model_provider_entity.declaration.model_credential_schema
        # 如果模型凭证模式不存在，则抛出异常
        if not model_credential_schema:
            raise ValueError(f"Provider {provider} does not have model_credential_schema")

        # 创建验证器并验证凭证
        validator = ModelCredentialSchemaValidator(model_type, model_credential_schema)
        filtered_credentials = validator.validate_and_filter(credentials)

        # 调用插件模型管理器进行凭证验证
        self.plugin_model_manager.validate_model_credentials(
            tenant_id=self.tenant_id,
            user_id="unknown",
            plugin_id=plugin_model_provider_entity.plugin_id,
            provider=plugin_model_provider_entity.provider,
            model_type=model_type.value,
            model=model,
            credentials=filtered_credentials,
        )

        return filtered_credentials

    # 获取模型模式的方法
    def get_model_schema(
        self, *, provider: str, model_type: ModelType, model: str, credentials: dict | None
    ) -> AIModelEntity | None:
        """
        获取指定模型的模式
        :param provider: 提供者名称
        :param model_type: 模型类型
        :param model: 模型名称
        :param credentials: 凭证字典（可选）
        :return: 模型模式或None
        """
        # 解析提供者名称，获取插件ID和提供者名称
        plugin_id, provider_name = self.get_plugin_id_and_provider_name_from_provider(provider)
        # 生成缓存键
        cache_key = f"{self.tenant_id}:{plugin_id}:{provider_name}:{model_type.value}:{model}"
        # 如果有凭证，则对凭证进行排序并生成哈希部分
        sorted_credentials = sorted(credentials.items()) if credentials else []
        cache_key += ":".join([hashlib.md5(f"{k}:{v}".encode()).hexdigest() for k, v in sorted_credentials])

        # 检查上下文中的模式缓存
        try:
            contexts.plugin_model_schemas.get()
        except LookupError:
            # 如果缓存未初始化，则设置为空字典并设置锁
            contexts.plugin_model_schemas.set({})
            contexts.plugin_model_schema_lock.set(Lock())

        # 使用锁访问模式缓存
        with contexts.plugin_model_schema_lock.get():
            # 如果缓存中存在，则直接返回
            if cache_key in contexts.plugin_model_schemas.get():
                return contexts.plugin_model_schemas.get()[cache_key]

            # 如果不存在，则从远程获取模式
            schema = self.plugin_model_manager.get_model_schema(
                tenant_id=self.tenant_id,
                user_id="unknown",
                plugin_id=plugin_id,
                provider=provider_name,
                model_type=model_type.value,
                model=model,
                credentials=credentials or {},
            )

            # 如果获取到模式，则将其缓存
            if schema:
                contexts.plugin_model_schemas.get()[cache_key] = schema

            return schema

    # 获取模型列表的方法
    def get_models(
        self,
        *,
        provider: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        provider_configs: Optional[list[ProviderConfig]] = None,
    ) -> list[SimpleProviderEntity]:
        """
        获取指定条件下的模型列表
        :param provider: 提供者名称（可选）
        :param model_type: 模型类型（可选）
        :param provider_configs: 提供者配置列表（可选）
        :return: 模型列表
        """
        # 初始化提供者配置列表
        provider_configs = provider_configs or []

        # 获取所有插件模型提供者实体
        plugin_model_provider_entities = self.get_plugin_model_providers()

        # 遍历所有提供者实体，筛选符合条件的模型
        providers = []
        for plugin_model_provider_entity in plugin_model_provider_entities:
            # 如果指定提供者，则过滤不符合的
            if provider and plugin_model_provider_entity.declaration.provider != provider:
                continue

            # 获取提供者的模式
            provider_schema = plugin_model_provider_entity.declaration

            # 获取支持的模型类型
            model_types = provider_schema.supported_model_types
            # 如果指定模型类型，则过滤不符合的
            if model_type:
                if model_type not in model_types:
                    continue
                model_types = [model_type]

            # 收集指定类型的所有模型
            all_model_type_models = []
            for model_schema in provider_schema.models:
                if model_schema.model_type != model_type:
                    continue
                all_model_type_models.append(model_schema)

            # 创建简单的提供者模式
            simple_provider_schema = provider_schema.to_simple_provider()
            # 将指定类型的所有模型添加到简单模式中
            simple_provider_schema.models.extend(all_model_type_models)

            # 将简单模式添加到结果列表中
            providers.append(simple_provider_schema)

        return providers

    # 根据提供者和模型类型获取模型实例的方法
    def get_model_type_instance(self, provider: str, model_type: ModelType) -> AIModel:
        """
        根据提供者名称和模型类型获取对应的模型实例
        :param provider: 提供者名称
        :param model_type: 模型类型
        :return: 模型实例
        """
        # 解析提供者名称，获取插件ID和提供者名称
        plugin_id, provider_name = self.get_plugin_id_and_provider_name_from_provider(provider)
        # 初始化模型实例参数
        init_params = {
            "tenant_id": self.tenant_id,
            "plugin_id": plugin_id,
            "provider_name": provider_name,
            "plugin_model_provider": self.get_plugin_model_provider(provider),
        }

        # 根据模型类型返回对应的模型实例
        if model_type == ModelType.LLM:
            return LargeLanguageModel(**init_params)  # type: ignore
        elif model_type == ModelType.TEXT_EMBEDDING:
            return TextEmbeddingModel(**init_params)  # type: ignore
        elif model_type == ModelType.RERANK:
            return RerankModel(**init_params)  # type: ignore
        elif model_type == ModelType.SPEECH2TEXT:
            return Speech2TextModel(**init_params)  # type: ignore
        elif model_type == ModelType.MODERATION:
            return ModerationModel(**init_params)  # type: ignore
        elif model_type == ModelType.TTS:
            return TTSModel(**init_params)  # type: ignore

    # 获取提供者图标的字节数据和MIME类型的方法
    def get_provider_icon(self, provider: str, icon_type: str, lang: str) -> tuple[bytes, str]:
        """
        获取指定提供者的图标
        :param provider: 提供者名称
        :param icon_type: 图标类型（icon_small或icon_large）
        :param lang: 语言（zh_Hans或en_US）
        :return: 图标字节数据和MIME类型
        """
        # 获取提供者的模式
        provider_schema = self.get_provider_schema(provider)

        # 根据图标类型选择对应的图标
        if icon_type.lower() == "icon_small":
            # 检查是否具有小图标
            if not provider_schema.icon_small:
                raise ValueError(f"Provider {provider} does not have small icon.")
            # 根据语言选择图标
            if lang.lower() == "zh_hans":
                file_name = provider_schema.icon_small.zh_Hans
            else:
                file_name = provider_schema.icon_small.en_US
        else:
            # 检查是否具有大图标
            if not provider_schema.icon_large:
                raise ValueError(f"Provider {provider} does not have large icon.")
            # 根据语言选择图标
            if lang.lower() == "zh_hans":
                file_name = provider_schema.icon_large.zh_Hans
            else:
                file_name = provider_schema.icon_large.en_US

        # 如果未找到图标文件，则抛出异常
        if not file_name:
            raise ValueError(f"Provider {provider} does not have icon.")

        # 定义常见的图片MIME类型
        image_mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "tif": "image/tiff",
            "webp": "image/webp",
            "svg": "image/svg+xml",
            "ico": "image/vnd.microsoft.icon",
            "heif": "image/heif",
            "heic": "image/heic",
        }

        # 获取文件扩展名并确定MIME类型
        extension = file_name.split(".")[-1]
        mime_type = image_mime_types.get(extension, "image/png")

        # 从插件资产管理器获取图标字节数据
        from core.plugin.impl.asset import PluginAssetManager

        plugin_asset_manager = PluginAssetManager()
        return plugin_asset_manager.fetch_asset(tenant_id=self.tenant_id, id=file_name), mime_type

    # 根据提供者名称获取插件ID和提供者名称的方法
    def get_plugin_id_and_provider_name_from_provider(self, provider: str) -> tuple[str, str]:
        """
        根据提供者名称获取对应的插件ID和提供者名称
        :param provider: 提供者名称
        :return: 插件ID和提供者名称
        """
        # 将提供者名称转换为ModelProviderID对象
        provider_id = ModelProviderID(provider)
        # 返回插件ID和提供者名称
        return provider_id.plugin_id, provider_id.provider_name
