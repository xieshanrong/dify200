
import json
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Optional

from extensions.ext_redis import redis_client


class ProviderCredentialsCache(ABC):
    """提供者凭证缓存的基类

    该抽象类定义了使用Redis缓存提供者凭证的接口。
    提供获取、设置和删除缓存凭证的方法，并要求子类实现缓存键生成逻辑。
    """

    def __init__(self, **kwargs):
        """使用生成的缓存键初始化缓存

        Args:
            **kwargs: 用于生成缓存键的任意关键字参数
        """
        self.cache_key = self._generate_cache_key(**kwargs)

    @abstractmethod
    def _generate_cache_key(self, **kwargs) -> str:
        """根据子类实现生成缓存键

        此方法必须由子类实现，定义缓存键的生成方式。

        Args:
            **kwargs: 用于生成缓存键的关键字参数

        Returns:
            str: 生成的缓存键
        """
        pass

    def get(self) -> Optional[dict]:
        """获取缓存的提供者凭证

        从Redis检索缓存的凭证。如果凭证存在，则解码并从JSON格式解析。

        Returns:
            Optional[dict]: 缓存的凭证字典，如果未找到或无效则返回None
        """
        cached_credentials = redis_client.get(self.cache_key)
        if cached_credentials:
            try:
                cached_credentials = cached_credentials.decode("utf-8")
                return dict(json.loads(cached_credentials))
            except JSONDecodeError:
                return None
        return None

    def set(self, config: dict[str, Any]):
        """缓存提供者凭证

        将给定的凭证配置存储在Redis中，过期时间为24小时。

        Args:
            config (dict[str, Any]): 要缓存的凭证配置
        """
        redis_client.setex(self.cache_key, 86400, json.dumps(config))

    def delete(self):
        """删除缓存的提供者凭证

        从Redis中删除缓存的凭证。
        """
        redis_client.delete(self.cache_key)


class SingletonProviderCredentialsCache(ProviderCredentialsCache):
    """工具单例提供者凭证缓存

    该类实现了单例提供者凭证的缓存，根据租户ID、提供者类型和提供者标识生成缓存键。
    """

    def __init__(self, tenant_id: str, provider_type: str, provider_identity: str):
        """使用租户和提供者信息初始化缓存

        Args:
            tenant_id (str): 租户ID
            provider_type (str): 提供者类型
            provider_identity (str): 提供者标识
        """
        super().__init__(
            tenant_id=tenant_id,
            provider_type=provider_type,
            provider_identity=provider_identity,
        )

    def _generate_cache_key(self, **kwargs) -> str:
        """为单例提供者凭证生成缓存键

        缓存键使用提供者类型、租户ID和派生的身份ID构建。

        Args:
            **kwargs: 包含'tenant_id'、'provider_type'和'provider_identity'的关键字参数

        Returns:
            str: 生成的缓存键，格式为"{provider_type}_credentials:tenant_id:{tenant_id}:id:{identity_id}"
        """
        tenant_id = kwargs["tenant_id"]
        provider_type = kwargs["provider_type"]
        identity_name = kwargs["provider_identity"]
        identity_id = f"{provider_type}.{identity_name}"
        return f"{provider_type}_credentials:tenant_id:{tenant_id}:id:{identity_id}"


class ToolProviderCredentialsCache(ProviderCredentialsCache):
    """工具提供者凭证缓存

    该类实现了工具提供者凭证的缓存，根据租户ID、提供者名称和凭证ID生成缓存键。
    """

    def __init__(self, tenant_id: str, provider: str, credential_id: str):
        """使用租户和凭证信息初始化缓存

        Args:
            tenant_id (str): 租户ID
            provider (str): 提供者名称
            credential_id (str): 凭证ID
        """
        super().__init__(tenant_id=tenant_id, provider=provider, credential_id=credential_id)

    def _generate_cache_key(self, **kwargs) -> str:
        """为工具提供者凭证生成缓存键

        缓存键使用租户ID、提供者名称和凭证ID构建。

        Args:
            **kwargs: 包含'tenant_id'、'provider'和'credential_id'的关键字参数

        Returns:
            str: 生成的缓存键，格式为"tool_credentials:tenant_id:{tenant_id}:provider:{provider}:credential_id:{credential_id}"
        """
        tenant_id = kwargs["tenant_id"]
        provider = kwargs["provider"]
        credential_id = kwargs["credential_id"]
        return f"tool_credentials:tenant_id:{tenant_id}:provider:{provider}:credential_id:{credential_id}"


class NoOpProviderCredentialCache:
    """无操作提供者凭证缓存

    该类提供了提供者凭证缓存的无操作实现，
    适用于禁用或不需要缓存的情况。
    """

    def get(self) -> Optional[dict]:
        """获取缓存的提供者凭证

        始终返回None，模拟缓存未命中。

        Returns:
            Optional[dict]: 始终返回None
        """
        return None

    def set(self, config: dict[str, Any]):
        """缓存提供者凭证

        不执行任何操作，模拟无操作缓存设置。

        Args:
            config (dict[str, Any]): 凭证配置（被忽略）
        """
        pass

    def delete(self):
        """删除缓存的提供者凭证

        不执行任何操作，模拟无操作缓存删除。
        """
        pass
