import base64
import hashlib
import hmac
import os
import time
import urllib.parse

from configs import dify_config


def get_signed_file_url(upload_file_id: str, as_attachment=False) -> str:
    """
    生成带有签名的文件访问URL。

    该函数用于生成一个安全的文件访问URL，通过在URL中添加签名参数来验证访问权限。

    Args:
        upload_file_id: 上传文件的唯一标识符
        as_attachment: 是否以附件形式下载文件，默认为False

    Returns:
        str: 带有签名参数的文件访问URL
    """
    # 构造基础URL
    url = f"{dify_config.FILES_URL}/files/{upload_file_id}/file-preview"

    # 生成时间戳
    timestamp = str(int(time.time()))
    # 生成随机数nonce
    nonce = os.urandom(16).hex()
    # 获取密钥
    key = dify_config.SECRET_KEY.encode()
    # 构造签名消息
    msg = f"file-preview|{upload_file_id}|{timestamp}|{nonce}"
    # 使用HMAC-SHA256生成签名
    sign = hmac.new(key, msg.encode(), hashlib.sha256).digest()
    # 对签名进行base64编码
    encoded_sign = base64.urlsafe_b64encode(sign).decode()

    # 构造查询参数
    query = {
        "timestamp": timestamp,
        "nonce": nonce,
        "sign": encoded_sign
    }
    # 如果需要以附件形式下载，添加as_attachment参数
    if as_attachment:
        query["as_attachment"] = "true"

    # 将查询参数编码为URL查询字符串
    query_string = urllib.parse.urlencode(query)

    # 返回完整的URL
    return f"{url}?{query_string}"


def get_signed_file_url_for_plugin(filename: str, mimetype: str, tenant_id: str, user_id: str) -> str:
    """
    为插件生成带有签名的文件上传URL。

    该函数用于插件内部通信时生成文件上传的URL，使用内部URL以确保在Docker网络中的可达性。

    Args:
        filename: 文件名
        mimetype: 文件的MIME类型
        tenant_id: 租户ID
        user_id: 用户ID

    Returns:
        str: 带有签名参数的文件上传URL
    """
    # 使用内部URL或默认的文件URL
    base_url = dify_config.INTERNAL_FILES_URL or dify_config.FILES_URL
    url = f"{base_url}/files/upload/for-plugin"

    # 如果用户ID未提供，使用默认值
    if user_id is None:
        user_id = "DEFAULT-USER"

    # 生成时间戳
    timestamp = str(int(time.time()))
    # 生成随机数nonce
    nonce = os.urandom(16).hex()
    # 获取密钥
    key = dify_config.SECRET_KEY.encode()
    # 构造签名消息
    msg = f"upload|{filename}|{mimetype}|{tenant_id}|{user_id}|{timestamp}|{nonce}"
    # 使用HMAC-SHA256生成签名
    sign = hmac.new(key, msg.encode(), hashlib.sha256).digest()
    # 对签名进行base64编码
    encoded_sign = base64.urlsafe_b64encode(sign).decode()

    # 构造完整的URL，并将所有参数附加在查询字符串中
    return f"{url}?timestamp={timestamp}&nonce={nonce}&sign={encoded_sign}&user_id={user_id}&tenant_id={tenant_id}"


def verify_plugin_file_signature(
    *, filename: str, mimetype: str, tenant_id: str, user_id: str | None, timestamp: str, nonce: str, sign: str
) -> bool:
    """
    验证插件文件签名的有效性。

    该函数用于验证通过插件生成的文件访问签名是否有效，包括签名是否匹配以及时间戳是否在有效期内。

    Args:
        filename: 文件名
        mimetype: 文件的MIME类型
        tenant_id: 租户ID
        user_id: 用户ID，可以为None
        timestamp: 时间戳字符串
        nonce: 随机数字符串
        sign: 签名字符串

    Returns:
        bool: 签名是否有效
    """
    # 如果用户ID未提供，使用默认值
    if user_id is None:
        user_id = "DEFAULT-USER"

    # 构造签名消息
    data_to_sign = f"upload|{filename}|{mimetype}|{tenant_id}|{user_id}|{timestamp}|{nonce}"
    # 获取密钥
    secret_key = dify_config.SECRET_KEY.encode()
    # 使用HMAC-SHA256重新生成签名
    recalculated_sign = hmac.new(secret_key, data_to_sign.encode(), hashlib.sha256).digest()
    # 对重新生成的签名进行base64编码
    recalculated_encoded_sign = base64.urlsafe_b64encode(recalculated_sign).decode()

    # 验证签名是否匹配
    if sign != recalculated_encoded_sign:
        return False

    # 检查时间戳是否在有效期内
    current_time = int(time.time())
    return current_time - int(timestamp) <= dify_config.FILES_ACCESS_TIMEOUT


def verify_image_signature(*, upload_file_id: str, timestamp: str, nonce: str, sign: str) -> bool:
    """
    验证图像文件访问签名的有效性。

    该函数用于验证图像文件的访问签名是否有效，包括签名是否匹配以及时间戳是否在有效期内。

    Args:
        upload_file_id: 上传文件的唯一标识符
        timestamp: 时间戳字符串
        nonce: 随机数字符串
        sign: 签名字符串

    Returns:
        bool: 签名是否有效
    """
    # 构造签名消息
    data_to_sign = f"image-preview|{upload_file_id}|{timestamp}|{nonce}"
    # 获取密钥
    secret_key = dify_config.SECRET_KEY.encode()
    # 使用HMAC-SHA256重新生成签名
    recalculated_sign = hmac.new(secret_key, data_to_sign.encode(), hashlib.sha256).digest()
    # 对重新生成的签名进行base64编码
    recalculated_encoded_sign = base64.urlsafe_b64encode(recalculated_sign).decode()

    # 验证签名是否匹配
    if sign != recalculated_encoded_sign:
        return False

    # 检查时间戳是否在有效期内
    current_time = int(time.time())
    return current_time - int(timestamp) <= dify_config.FILES_ACCESS_TIMEOUT


def verify_file_signature(*, upload_file_id: str, timestamp: str, nonce: str, sign: str) -> bool:
    """
    验证普通文件访问签名的有效性。

    该函数用于验证普通文件的访问签名是否有效，包括签名是否匹配以及时间戳是否在有效期内。

    Args:
        upload_file_id: 上传文件的唯一标识符
        timestamp: 时间戳字符串
        nonce: 随机数字符串
        sign: 签名字符串

    Returns:
        bool: 签名是否有效
    """
    # 构造签名消息
    data_to_sign = f"file-preview|{upload_file_id}|{timestamp}|{nonce}"
    # 获取密钥
    secret_key = dify_config.SECRET_KEY.encode()
    # 使用HMAC-SHA256重新生成签名
    recalculated_sign = hmac.new(secret_key, data_to_sign.encode(), hashlib.sha256).digest()
    # 对重新生成的签名进行base64编码
    recalculated_encoded_sign = base64.urlsafe_b64encode(recalculated_sign).decode()

    # 验证签名是否匹配
    if sign != recalculated_encoded_sign:
        return False

    # 检查时间戳是否在有效期内
    current_time = int(time.time())
    return current_time - int(timestamp) <= dify_config.FILES_ACCESS_TIMEOUT
