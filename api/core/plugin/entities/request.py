from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.entities.provider_entities import BasicProviderConfig
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.entities.model_entities import ModelType
from core.workflow.nodes.parameter_extractor.entities import (
    ModelConfig as ParameterExtractorModelConfig,
)
from core.workflow.nodes.parameter_extractor.entities import (
    ParameterConfig,
)
from core.workflow.nodes.question_classifier.entities import (
    ClassConfig,
)
from core.workflow.nodes.question_classifier.entities import (
    ModelConfig as QuestionClassifierModelConfig,
)


class InvokeCredentials(BaseModel):
    """
    调用凭证类

    用于存储工具提供者的凭证ID映射关系。

    属性:
        tool_credentials (dict[str, str]): 工具提供者到凭证ID的映射字典
    """
    tool_credentials: dict[str, str] = Field(
        default_factory=dict,
        description="Map of tool provider to credential id, used to store the credential id for the tool provider.",
    )


class PluginInvokeContext(BaseModel):
    """
    插件调用上下文类

    包含插件调用或反向调用所需的凭证上下文信息。

    属性:
        credentials (Optional[InvokeCredentials]): 调用凭证上下文
    """
    credentials: Optional[InvokeCredentials] = Field(
        default_factory=InvokeCredentials,
        description="Credentials context for the plugin invocation or backward invocation.",
    )


class RequestInvokeTool(BaseModel):
    """
    请求调用工具类

    用于封装调用工具所需的各种参数信息。

    属性:
        tool_type (Literal["builtin", "workflow", "api", "mcp"]): 工具类型
        provider (str): 工具提供者
        tool (str): 工具名称
        tool_parameters (dict): 工具参数
        credential_id (Optional[str]): 凭证ID
    """

    tool_type: Literal["builtin", "workflow", "api", "mcp"]
    provider: str
    tool: str
    tool_parameters: dict
    credential_id: Optional[str] = None


class BaseRequestInvokeModel(BaseModel):
    """
    基础模型调用请求类

    作为所有模型调用请求类的基类，定义了通用的模型调用参数。

    属性:
        provider (str): 模型提供者
        model (str): 模型名称
        model_type (ModelType): 模型类型
    """
    provider: str
    model: str
    model_type: ModelType

    model_config = ConfigDict(protected_namespaces=())


class RequestInvokeLLM(BaseRequestInvokeModel):
    """
    请求调用大语言模型类

    用于封装调用大语言模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为LLM
        mode (str): 调用模式
        completion_params (dict[str, Any]): 完成参数
        prompt_messages (list[PromptMessage]): 提示消息列表
        tools (Optional[list[PromptMessageTool]]): 工具列表
        stop (Optional[list[str]]): 停止词列表
        stream (Optional[bool]): 是否流式输出
    """

    model_type: ModelType = ModelType.LLM
    mode: str
    completion_params: dict[str, Any] = Field(default_factory=dict)
    prompt_messages: list[PromptMessage] = Field(default_factory=list)
    tools: Optional[list[PromptMessageTool]] = Field(default_factory=list[PromptMessageTool])
    stop: Optional[list[str]] = Field(default_factory=list[str])
    stream: Optional[bool] = False

    model_config = ConfigDict(protected_namespaces=())

    @field_validator("prompt_messages", mode="before")
    @classmethod
    def convert_prompt_messages(cls, v):
        """
        转换提示消息格式的验证器

        将原始的提示消息字典列表转换为对应的提示消息对象列表。

        参数:
            v: 原始提示消息数据

        返回:
            转换后的提示消息对象列表

        异常:
            ValueError: 当提示消息不是列表格式时抛出
        """
        if not isinstance(v, list):
            raise ValueError("prompt_messages must be a list")

        for i in range(len(v)):
            if v[i]["role"] == PromptMessageRole.USER.value:
                v[i] = UserPromptMessage(**v[i])
            elif v[i]["role"] == PromptMessageRole.ASSISTANT.value:
                v[i] = AssistantPromptMessage(**v[i])
            elif v[i]["role"] == PromptMessageRole.SYSTEM.value:
                v[i] = SystemPromptMessage(**v[i])
            elif v[i]["role"] == PromptMessageRole.TOOL.value:
                v[i] = ToolPromptMessage(**v[i])
            else:
                v[i] = PromptMessage(**v[i])

        return v


class RequestInvokeLLMWithStructuredOutput(RequestInvokeLLM):
    """
    请求调用具有结构化输出的大语言模型类

    扩展自RequestInvokeLLM，增加了结构化输出的schema定义。

    属性:
        structured_output_schema (dict[str, Any]): 结构化输出的JSON schema格式定义
    """

    structured_output_schema: dict[str, Any] = Field(
        default_factory=dict, description="The schema of the structured output in JSON schema format"
    )


class RequestInvokeTextEmbedding(BaseRequestInvokeModel):
    """
    请求调用文本嵌入模型类

    用于封装调用文本嵌入模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为TEXT_EMBEDDING
        texts (list[str]): 待嵌入的文本列表
    """

    model_type: ModelType = ModelType.TEXT_EMBEDDING
    texts: list[str]


class RequestInvokeRerank(BaseRequestInvokeModel):
    """
    请求调用重排序模型类

    用于封装调用重排序模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为RERANK
        query (str): 查询语句
        docs (list[str]): 文档列表
        score_threshold (float): 分数阈值
        top_n (int): 返回前N个结果
    """

    model_type: ModelType = ModelType.RERANK
    query: str
    docs: list[str]
    score_threshold: float
    top_n: int


class RequestInvokeTTS(BaseRequestInvokeModel):
    """
    请求调用文本转语音模型类

    用于封装调用文本转语音模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为TTS
        content_text (str): 待转换的文本内容
        voice (str): 语音类型
    """

    model_type: ModelType = ModelType.TTS
    content_text: str
    voice: str


class RequestInvokeSpeech2Text(BaseRequestInvokeModel):
    """
    请求调用语音转文本模型类

    用于封装调用语音转文本模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为SPEECH2TEXT
        file (bytes): 音频文件数据
    """

    model_type: ModelType = ModelType.SPEECH2TEXT
    file: bytes

    @field_validator("file", mode="before")
    @classmethod
    def convert_file(cls, v):
        """
        转换文件数据格式的验证器

        将十六进制字符串转换为字节数据。

        参数:
            v: 原始文件数据（应为十六进制字符串）

        返回:
            转换后的字节数据

        异常:
            ValueError: 当文件数据不是十六进制字符串时抛出
        """
        # hex string to bytes
        if isinstance(v, str):
            return bytes.fromhex(v)
        else:
            raise ValueError("file must be a hex string")


class RequestInvokeModeration(BaseRequestInvokeModel):
    """
    请求调用内容审核模型类

    用于封装调用内容审核模型所需的各种参数信息。

    属性:
        model_type (ModelType): 模型类型，固定为MODERATION
        text (str): 待审核的文本内容
    """

    model_type: ModelType = ModelType.MODERATION
    text: str


class RequestInvokeParameterExtractorNode(BaseModel):
    """
    请求调用参数提取器节点类

    用于封装调用参数提取器节点所需的各种参数信息。

    属性:
        parameters (list[ParameterConfig]): 参数配置列表
        model (ParameterExtractorModelConfig): 模型配置
        instruction (str): 指令信息
        query (str): 查询语句
    """

    parameters: list[ParameterConfig]
    model: ParameterExtractorModelConfig
    instruction: str
    query: str


class RequestInvokeQuestionClassifierNode(BaseModel):
    """
    请求调用问题分类器节点类

    用于封装调用问题分类器节点所需的各种参数信息。

    属性:
        query (str): 查询语句
        model (QuestionClassifierModelConfig): 模型配置
        classes (list[ClassConfig]): 分类配置列表
        instruction (str): 指令信息
    """

    query: str
    model: QuestionClassifierModelConfig
    classes: list[ClassConfig]
    instruction: str


class RequestInvokeApp(BaseModel):
    """
    请求调用应用类

    用于封装调用应用所需的各种参数信息。

    属性:
        app_id (str): 应用ID
        inputs (dict[str, Any]): 输入参数
        query (Optional[str]): 查询语句
        response_mode (Literal["blocking", "streaming"]): 响应模式
        conversation_id (Optional[str]): 对话ID
        user (Optional[str]): 用户ID
        files (list[dict]): 文件列表
    """

    app_id: str
    inputs: dict[str, Any]
    query: Optional[str] = None
    response_mode: Literal["blocking", "streaming"]
    conversation_id: Optional[str] = None
    user: Optional[str] = None
    files: list[dict] = Field(default_factory=list)


class RequestInvokeEncrypt(BaseModel):
    """
    请求加密类

    用于封装加密/解密操作所需的各种参数信息。

    属性:
        opt (Literal["encrypt", "decrypt", "clear"]): 操作类型
        namespace (Literal["endpoint"]): 命名空间
        identity (str): 身份标识
        data (dict): 数据内容
        config (list[BasicProviderConfig]): 配置列表
    """

    opt: Literal["encrypt", "decrypt", "clear"]
    namespace: Literal["endpoint"]
    identity: str
    data: dict = Field(default_factory=dict)
    config: list[BasicProviderConfig] = Field(default_factory=list)


class RequestInvokeSummary(BaseModel):
    """
    请求摘要类

    用于封装文本摘要操作所需的各种参数信息。

    属性:
        text (str): 待摘要的文本
        instruction (str): 摘要指令
    """

    text: str
    instruction: str


class RequestRequestUploadFile(BaseModel):
    """
    请求上传文件类

    用于封装文件上传请求所需的各种参数信息。

    属性:
        filename (str): 文件名
        mimetype (str): 文件MIME类型
    """

    filename: str
    mimetype: str


class RequestFetchAppInfo(BaseModel):
    """
    请求获取应用信息类

    用于封装获取应用信息请求所需的各种参数信息。

    属性:
        app_id (str): 应用ID
    """

    app_id: str
