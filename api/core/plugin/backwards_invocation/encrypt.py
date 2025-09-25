from core.helper.provider_cache import SingletonProviderCredentialsCache
from core.plugin.entities.request import RequestInvokeEncrypt
from core.tools.utils.encryption import create_provider_encrypter
from models.account import Tenant

class PluginEncrypter:
    """用于处理插件加密请求的工具类"""

    @classmethod
    def invoke_encrypt(cls, tenant: Tenant, payload: RequestInvokeEncrypt):
        """根据请求参数执行加密、解密或清除缓存操作

        Args:
            tenant: 当前租户信息
            payload: 包含加密请求参数的实体

        Returns:
            包含处理结果的字典，其中"data"字段存储处理后的数据

        Raises:
            ValueError: 当请求操作无效时抛出异常
        """
        # 创建加密器和缓存实例
        encrypter, cache = create_provider_encrypter(
            tenant_id=tenant.id,  # 当前租户的唯一标识符
            config=payload.config,  # 加密配置参数
            cache=SingletonProviderCredentialsCache(
                tenant_id=tenant.id,  # 当前租户的唯一标识符
                provider_type=payload.namespace,  # 提供者类型
                provider_identity=payload.identity  # 提供者唯一标识
            )
        )

        # 根据请求的操作类型执行相应的处理
        if payload.opt == "encrypt":
            # 执行加密操作
            return {
                "data": encrypter.encrypt(payload.data)
            }
        elif payload.opt == "decrypt":
            # 执行解密操作
            return {
                "data": encrypter.decrypt(payload.data)
            }
        elif payload.opt == "clear":
            # 清除缓存
            cache.delete()
            return {
                "data": {}
            }
        else:
            # 当操作类型无效时抛出异常
            raise ValueError(f"Invalid opt: {payload.opt}")
