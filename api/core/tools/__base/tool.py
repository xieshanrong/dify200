from abc import ABC, abstractmethod
from collections.abc import Generator
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from models.model import File

from core.tools.__base.tool_runtime import ToolRuntime
from core.tools.entities.tool_entities import (
    ToolEntity,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)


class Tool(ABC):
    """
    工具的基类

    该类定义了所有工具的基本结构和接口，提供了工具执行、参数处理和消息创建等功能。
    子类必须实现抽象方法以提供具体的功能实现。

    属性:
        entity (ToolEntity): 工具实体，包含工具的基本信息和参数定义
        runtime (ToolRuntime): 工具运行时环境，包含运行时的配置和上下文信息
    """

    def __init__(self, entity: ToolEntity, runtime: ToolRuntime):
        """
        初始化工具实例

        参数:
            entity (ToolEntity): 工具实体对象
            runtime (ToolRuntime): 工具运行时环境对象
        """
        self.entity = entity
        self.runtime = runtime

    def fork_tool_runtime(self, runtime: ToolRuntime) -> "Tool":
        """
        基于当前工具创建一个新的工具实例，使用新的运行时环境

        参数:
            runtime (ToolRuntime): 新的运行时环境

        返回:
            Tool: 使用新运行时环境的工具实例
        """
        return self.__class__(
            entity=self.entity.model_copy(),
            runtime=runtime,
        )

    @abstractmethod
    def tool_provider_type(self) -> ToolProviderType:
        """
        获取工具提供者类型

        抽象方法，子类必须实现该方法以返回工具的提供者类型。

        返回:
            ToolProviderType: 工具提供者类型枚举值
        """

    def invoke(
        self,
        user_id: str,
        tool_parameters: dict[str, Any],
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Generator[ToolInvokeMessage]:
        """
        调用工具执行具体功能

        该方法处理运行时参数，转换参数类型，并调用具体的执行方法。
        根据执行结果的类型，将其转换为生成器形式返回。

        参数:
            user_id (str): 用户ID
            tool_parameters (dict[str, Any]): 工具调用参数
            conversation_id (Optional[str]): 对话ID
            app_id (Optional[str]): 应用ID
            message_id (Optional[str]): 消息ID

        返回:
            Generator[ToolInvokeMessage]: 工具执行结果的消息生成器
        """
        if self.runtime and self.runtime.runtime_parameters:
            tool_parameters.update(self.runtime.runtime_parameters)

        # 尝试将工具参数转换为正确的类型
        tool_parameters = self._transform_tool_parameters_type(tool_parameters)

        result = self._invoke(
            user_id=user_id,
            tool_parameters=tool_parameters,
            conversation_id=conversation_id,
            app_id=app_id,
            message_id=message_id,
        )

        if isinstance(result, ToolInvokeMessage):
            # 如果结果是单个消息，创建一个生成单个消息的生成器

            def single_generator() -> Generator[ToolInvokeMessage, None, None]:
                yield result

            return single_generator()
        elif isinstance(result, list):
            # 如果结果是消息列表，创建一个生成列表中所有消息的生成器

            def generator() -> Generator[ToolInvokeMessage, None, None]:
                yield from result

            return generator()
        else:
            return result

    def _transform_tool_parameters_type(self, tool_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        转换工具参数类型

        根据工具实体中定义的参数类型，将传入的参数值转换为相应的类型。

        参数:
            tool_parameters (dict[str, Any]): 待转换的工具参数

        返回:
            dict[str, Any]: 转换后的工具参数
        """
        # 临时修复：在验证凭据时工具参数可能被转换为空的问题
        result = deepcopy(tool_parameters)
        for parameter in self.entity.parameters or []:
            if parameter.name in tool_parameters:
                result[parameter.name] = parameter.type.cast_value(tool_parameters[parameter.name])

        return result

    @abstractmethod
    def _invoke(
        self,
        user_id: str,
        tool_parameters: dict[str, Any],
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> ToolInvokeMessage | list[ToolInvokeMessage] | Generator[ToolInvokeMessage, None, None]:
        """
        工具具体执行逻辑

        抽象方法，子类必须实现该方法以提供具体的工具执行逻辑。

        参数:
            user_id (str): 用户ID
            tool_parameters (dict[str, Any]): 工具调用参数
            conversation_id (Optional[str]): 对话ID
            app_id (Optional[str]): 应用ID
            message_id (Optional[str]): 消息ID

        返回:
            ToolInvokeMessage | list[ToolInvokeMessage] | Generator[ToolInvokeMessage, None, None]:
            工具执行结果，可以是单个消息、消息列表或消息生成器
        """
        pass

    def get_runtime_parameters(
        self,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[ToolParameter]:
        """
        获取运行时参数

        为开发者提供动态更改工具参数的接口，可根据变量池的内容来调整工具参数。

        参数:
            conversation_id (Optional[str]): 对话ID
            app_id (Optional[str]): 应用ID
            message_id (Optional[str]): 消息ID

        返回:
            list[ToolParameter]: 运行时参数列表
        """
        return self.entity.parameters

    def get_merged_runtime_parameters(
        self,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[ToolParameter]:
        """
        获取合并后的运行时参数

        将工具实体中的参数与运行时参数进行合并，运行时参数会覆盖实体中的同名参数，
        如果有新的参数则添加到参数列表中。

        参数:
            conversation_id (Optional[str]): 对话ID
            app_id (Optional[str]): 应用ID
            message_id (Optional[str]): 消息ID

        返回:
            list[ToolParameter]: 合并后的运行时参数列表
        """
        parameters = self.entity.parameters
        parameters = parameters.copy()
        user_parameters = self.get_runtime_parameters() or []
        user_parameters = user_parameters.copy()

        # 覆盖参数
        for parameter in user_parameters:
            # 检查参数是否在工具参数中存在
            for tool_parameter in parameters:
                if tool_parameter.name == parameter.name:
                    # 覆盖参数属性
                    tool_parameter.type = parameter.type
                    tool_parameter.form = parameter.form
                    tool_parameter.required = parameter.required
                    tool_parameter.default = parameter.default
                    tool_parameter.options = parameter.options
                    tool_parameter.llm_description = parameter.llm_description
                    break
            else:
                # 添加新参数
                parameters.append(parameter)

        return parameters

    def create_image_message(
        self,
        image: str,
    ) -> ToolInvokeMessage:
        """
        创建图片消息

        参数:
            image (str): 图片的URL地址

        返回:
            ToolInvokeMessage: 图片消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.IMAGE, message=ToolInvokeMessage.TextMessage(text=image)
        )

    def create_file_message(self, file: "File") -> ToolInvokeMessage:
        """
        创建文件消息

        参数:
            file (File): 文件对象

        返回:
            ToolInvokeMessage: 文件消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.FILE,
            message=ToolInvokeMessage.FileMessage(),
            meta={"file": file},
        )

    def create_link_message(self, link: str) -> ToolInvokeMessage:
        """
        创建链接消息

        参数:
            link (str): 链接的URL地址

        返回:
            ToolInvokeMessage: 链接消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.LINK, message=ToolInvokeMessage.TextMessage(text=link)
        )

    def create_text_message(self, text: str) -> ToolInvokeMessage:
        """
        创建文本消息

        参数:
            text (str): 文本内容

        返回:
            ToolInvokeMessage: 文本消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text=text),
        )

    def create_blob_message(self, blob: bytes, meta: Optional[dict] = None) -> ToolInvokeMessage:
        """
        创建二进制消息

        参数:
            blob (bytes): 二进制数据
            meta (Optional[dict]): 二进制对象的元信息

        返回:
            ToolInvokeMessage: 二进制消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.BLOB,
            message=ToolInvokeMessage.BlobMessage(blob=blob),
            meta=meta,
        )

    def create_json_message(self, object: dict) -> ToolInvokeMessage:
        """
        创建JSON消息

        参数:
            object (dict): JSON对象

        返回:
            ToolInvokeMessage: JSON消息对象
        """
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.JSON, message=ToolInvokeMessage.JsonMessage(json_object=object)
        )
