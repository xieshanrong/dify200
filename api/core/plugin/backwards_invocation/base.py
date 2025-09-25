from collections.abc import Generator, Mapping
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel

class BaseBackwardsInvocation:
    """用于将响应转换为事件流的基础类"""
    @classmethod
    def convert_to_event_stream(cls, response: Generator[BaseModel | Mapping | str, None, None] | BaseModel | Mapping):
        """将响应转换为事件流，处理生成器或单个响应"""
        if isinstance(response, Generator):
            try:
                for chunk in response:
                    if isinstance(chunk, (BaseModel, dict)):
                        yield BaseBackwardsInvocationResponse(data=chunk).model_dump_json().encode()
            except Exception as e:
                error_message = BaseBackwardsInvocationResponse(error=str(e)).model_dump_json()
                yield error_message.encode()
        else:
            yield BaseBackwardsInvocationResponse(data=response).model_dump_json().encode()

# 定义类型变量T，限制其为字典、映射、字符串、布尔值、整数或Pydantic模型
T = TypeVar("T", bound=dict | Mapping | str | bool | int | BaseModel)


class BaseBackwardsInvocationResponse(BaseModel, Generic[T]):
    """反向调用的响应模型"""
    # 响应数据，类型为T，可选
    data: Optional[T] = None
    # 错误信息，字符串，默认为空
    error: str = ""
