from core.plugin.impl.base import BasePluginClient


class PluginAssetManager(BasePluginClient):
    """
    插件资产管理器类，用于获取插件相关的资产数据。

    继承自BasePluginClient，提供与插件资产相关的HTTP请求功能。
    """

    def fetch_asset(self, tenant_id: str, id: str) -> bytes:
        """
        根据资产ID获取插件资产数据。

        通过HTTP GET请求从服务器获取指定ID的资产数据，如果请求成功则返回资产的二进制内容，
        如果请求失败则抛出异常。

        Args:
            tenant_id (str): 租户ID，用于标识资产所属的租户
            id (str): 资产ID，用于唯一标识要获取的资产

        Returns:
            bytes: 资产的二进制数据内容

        Raises:
            ValueError: 当无法找到指定ID的资产时抛出此异常
        """
        # 发送HTTP GET请求获取资产数据
        response = self._request(method="GET", path=f"plugin/{tenant_id}/asset/{id}")

        # 检查响应状态码，非200表示获取失败
        if response.status_code != 200:
            raise ValueError(f"can not found asset {id}")

        # 返回资产的二进制内容
        return response.content
