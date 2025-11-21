
from flask_restx import Resource

from controllers.console.wraps import setup_required
from controllers.inner_api import inner_api_ns
from controllers.inner_api.plugin.wraps import get_user_tenant, plugin_data
from controllers.inner_api.wraps import plugin_inner_api_only
from core.file.helpers import get_signed_file_url_for_plugin
from core.model_runtime.utils.encoders import jsonable_encoder
from core.plugin.backwards_invocation.app import PluginAppBackwardsInvocation
from core.plugin.backwards_invocation.base import BaseBackwardsInvocationResponse
from core.plugin.backwards_invocation.encrypt import PluginEncrypter
from core.plugin.backwards_invocation.model import PluginModelBackwardsInvocation
from core.plugin.backwards_invocation.node import PluginNodeBackwardsInvocation
from core.plugin.backwards_invocation.tool import PluginToolBackwardsInvocation
from core.plugin.entities.request import (
    RequestFetchAppInfo,
    RequestInvokeApp,
    RequestInvokeEncrypt,
    RequestInvokeLLM,
    RequestInvokeLLMWithStructuredOutput,
    RequestInvokeModeration,
    RequestInvokeParameterExtractorNode,
    RequestInvokeQuestionClassifierNode,
    RequestInvokeRerank,
    RequestInvokeSpeech2Text,
    RequestInvokeSummary,
    RequestInvokeTextEmbedding,
    RequestInvokeTool,
    RequestInvokeTTS,
    RequestRequestUploadFile,
)
from core.tools.entities.tool_entities import ToolProviderType
from libs.helper import length_prefixed_response
from models.account import Account, Tenant
from models.model import EndUser


@inner_api_ns.route("/invoke/llm")
class PluginInvokeLLMApi(Resource):
    """
    插件接口：调用大语言模型（LLM）功能。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeLLM): 调用LLM所需的请求参数。

    返回:
        Streaming response: LLM调用结果的流式响应。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeLLM)
    @inner_api_ns.doc("plugin_invoke_llm")
    @inner_api_ns.doc(description="Invoke LLM models through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "LLM invocation successful (streaming response)",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeLLM):
        def generator():
            response = PluginModelBackwardsInvocation.invoke_llm(user_model.id, tenant_model, payload)
            return PluginModelBackwardsInvocation.convert_to_event_stream(response)

        return length_prefixed_response(0xF, generator())


@inner_api_ns.route("/invoke/llm/structured-output")
class PluginInvokeLLMWithStructuredOutputApi(Resource):
    """
    插件接口：调用支持结构化输出的大语言模型（LLM）功能。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeLLMWithStructuredOutput): 结构化输出调用所需的请求参数。

    返回:
        Streaming response: LLM结构化输出调用结果的流式响应。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeLLMWithStructuredOutput)
    @inner_api_ns.doc("plugin_invoke_llm_structured")
    @inner_api_ns.doc(description="Invoke LLM models with structured output through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "LLM structured output invocation successful (streaming response)",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeLLMWithStructuredOutput):
        def generator():
            response = PluginModelBackwardsInvocation.invoke_llm_with_structured_output(
                user_model.id, tenant_model, payload
            )
            return PluginModelBackwardsInvocation.convert_to_event_stream(response)

        return length_prefixed_response(0xF, generator())


@inner_api_ns.route("/invoke/text-embedding")
class PluginInvokeTextEmbeddingApi(Resource):
    """
    插件接口：调用文本嵌入（Text Embedding）模型。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeTextEmbedding): 文本嵌入调用所需的请求参数。

    返回:
        JSON response: 包含嵌入向量的结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeTextEmbedding)
    @inner_api_ns.doc("plugin_invoke_text_embedding")
    @inner_api_ns.doc(description="Invoke text embedding models through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Text embedding successful",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeTextEmbedding):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginModelBackwardsInvocation.invoke_text_embedding(
                        user_id=user_model.id,
                        tenant=tenant_model,
                        payload=payload,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/rerank")
class PluginInvokeRerankApi(Resource):
    """
    插件接口：调用重排序（Rerank）模型。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeRerank): 重排序调用所需的请求参数。

    返回:
        JSON response: 包含重排序结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeRerank)
    @inner_api_ns.doc("plugin_invoke_rerank")
    @inner_api_ns.doc(description="Invoke rerank models through plugin interface")
    @inner_api_ns.doc(
        responses={200: "Rerank successful", 401: "Unauthorized - invalid API key", 404: "Service not available"}
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeRerank):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginModelBackwardsInvocation.invoke_rerank(
                        user_id=user_model.id,
                        tenant=tenant_model,
                        payload=payload,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/tts")
class PluginInvokeTTSApi(Resource):
    """
    插件接口：调用文本转语音（TTS）模型。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeTTS): TTS调用所需的请求参数。

    返回:
        Streaming response: TTS调用结果的流式响应。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeTTS)
    @inner_api_ns.doc("plugin_invoke_tts")
    @inner_api_ns.doc(description="Invoke text-to-speech models through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "TTS invocation successful (streaming response)",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeTTS):
        def generator():
            response = PluginModelBackwardsInvocation.invoke_tts(
                user_id=user_model.id,
                tenant=tenant_model,
                payload=payload,
            )
            return PluginModelBackwardsInvocation.convert_to_event_stream(response)

        return length_prefixed_response(0xF, generator())


@inner_api_ns.route("/invoke/speech2text")
class PluginInvokeSpeech2TextApi(Resource):
    """
    插件接口：调用语音转文本（Speech-to-Text）模型。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeSpeech2Text): 语音转文本调用所需的请求参数。

    返回:
        JSON response: 包含识别结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeSpeech2Text)
    @inner_api_ns.doc("plugin_invoke_speech2text")
    @inner_api_ns.doc(description="Invoke speech-to-text models through plugin interface")
    @inner_api_ns.doc(
        responses={200: "Speech2Text successful", 401: "Unauthorized - invalid API key", 404: "Service not available"}
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeSpeech2Text):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginModelBackwardsInvocation.invoke_speech2text(
                        user_id=user_model.id,
                        tenant=tenant_model,
                        payload=payload,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/moderation")
class PluginInvokeModerationApi(Resource):
    """
    插件接口：调用内容审核（Moderation）模型。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeModeration): 内容审核调用所需的请求参数。

    返回:
        JSON response: 包含审核结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeModeration)
    @inner_api_ns.doc("plugin_invoke_moderation")
    @inner_api_ns.doc(description="Invoke moderation models through plugin interface")
    @inner_api_ns.doc(
        responses={200: "Moderation successful", 401: "Unauthorized - invalid API key", 404: "Service not available"}
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeModeration):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginModelBackwardsInvocation.invoke_moderation(
                        user_id=user_model.id,
                        tenant=tenant_model,
                        payload=payload,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/tool")
class PluginInvokeToolApi(Resource):
    """
    插件接口：调用工具（Tool）功能。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeTool): 工具调用所需的请求参数。

    返回:
        Streaming response: 工具调用结果的流式响应。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeTool)
    @inner_api_ns.doc("plugin_invoke_tool")
    @inner_api_ns.doc(description="Invoke tools through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Tool invocation successful (streaming response)",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeTool):
        def generator():
            return PluginToolBackwardsInvocation.convert_to_event_stream(
                PluginToolBackwardsInvocation.invoke_tool(
                    tenant_id=tenant_model.id,
                    user_id=user_model.id,
                    tool_type=ToolProviderType.value_of(payload.tool_type),
                    provider=payload.provider,
                    tool_name=payload.tool,
                    tool_parameters=payload.tool_parameters,
                    credential_id=payload.credential_id,
                ),
            )

        return length_prefixed_response(0xF, generator())


@inner_api_ns.route("/invoke/parameter-extractor")
class PluginInvokeParameterExtractorNodeApi(Resource):
    """
    插件接口：调用参数提取节点（Parameter Extractor Node）。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeParameterExtractorNode): 参数提取调用所需的请求参数。

    返回:
        JSON response: 包含提取结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeParameterExtractorNode)
    @inner_api_ns.doc("plugin_invoke_parameter_extractor")
    @inner_api_ns.doc(description="Invoke parameter extractor node through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Parameter extraction successful",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeParameterExtractorNode):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginNodeBackwardsInvocation.invoke_parameter_extractor(
                        tenant_id=tenant_model.id,
                        user_id=user_model.id,
                        parameters=payload.parameters,
                        model_config=payload.model,
                        instruction=payload.instruction,
                        query=payload.query,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/question-classifier")
class PluginInvokeQuestionClassifierNodeApi(Resource):
    """
    插件接口：调用问题分类节点（Question Classifier Node）。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeQuestionClassifierNode): 问题分类调用所需的请求参数。

    返回:
        JSON response: 包含分类结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeQuestionClassifierNode)
    @inner_api_ns.doc("plugin_invoke_question_classifier")
    @inner_api_ns.doc(description="Invoke question classifier node through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Question classification successful",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeQuestionClassifierNode):
        try:
            return jsonable_encoder(
                BaseBackwardsInvocationResponse(
                    data=PluginNodeBackwardsInvocation.invoke_question_classifier(
                        tenant_id=tenant_model.id,
                        user_id=user_model.id,
                        query=payload.query,
                        model_config=payload.model,
                        classes=payload.classes,
                        instruction=payload.instruction,
                    )
                )
            )
        except Exception as e:
            return jsonable_encoder(BaseBackwardsInvocationResponse(error=str(e)))


@inner_api_ns.route("/invoke/app")
class PluginInvokeAppApi(Resource):
    """
    插件接口：调用应用（Application）功能。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeApp): 应用调用所需的请求参数。

    返回:
        Streaming response: 应用调用结果的流式响应。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeApp)
    @inner_api_ns.doc("plugin_invoke_app")
    @inner_api_ns.doc(description="Invoke application through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "App invocation successful (streaming response)",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeApp):
        response = PluginAppBackwardsInvocation.invoke_app(
            app_id=payload.app_id,
            user_id=user_model.id,
            tenant_id=tenant_model.id,
            conversation_id=payload.conversation_id,
            query=payload.query,
            stream=payload.response_mode == "streaming",
            inputs=payload.inputs,
            files=payload.files,
        )

        return length_prefixed_response(0xF, PluginAppBackwardsInvocation.convert_to_event_stream(response))


@inner_api_ns.route("/invoke/encrypt")
class PluginInvokeEncryptApi(Resource):
    """
    插件接口：加密或解密数据。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeEncrypt): 加密/解密调用所需的请求参数。

    返回:
        JSON response: 包含加密/解密结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeEncrypt)
    @inner_api_ns.doc("plugin_invoke_encrypt")
    @inner_api_ns.doc(description="Encrypt or decrypt data through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Encryption/decryption successful",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeEncrypt):
        """
        encrypt or decrypt data
        """
        try:
            return BaseBackwardsInvocationResponse(
                data=PluginEncrypter.invoke_encrypt(tenant_model, payload)
            ).model_dump()
        except Exception as e:
            return BaseBackwardsInvocationResponse(error=str(e)).model_dump()


@inner_api_ns.route("/invoke/summary")
class PluginInvokeSummaryApi(Resource):
    """
    插件接口：调用摘要生成功能。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestInvokeSummary): 摘要生成调用所需的请求参数。

    返回:
        JSON response: 包含摘要结果或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestInvokeSummary)
    @inner_api_ns.doc("plugin_invoke_summary")
    @inner_api_ns.doc(description="Invoke summary functionality through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Summary generation successful",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestInvokeSummary):
        try:
            return BaseBackwardsInvocationResponse(
                data={
                    "summary": PluginModelBackwardsInvocation.invoke_summary(
                        user_id=user_model.id,
                        tenant=tenant_model,
                        payload=payload,
                    )
                }
            ).model_dump()
        except Exception as e:
            return BaseBackwardsInvocationResponse(error=str(e)).model_dump()


@inner_api_ns.route("/upload/file/request")
class PluginUploadFileRequestApi(Resource):
    """
    插件接口：请求文件上传的签名URL。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestRequestUploadFile): 文件上传请求参数。

    返回:
        JSON response: 包含签名URL或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestRequestUploadFile)
    @inner_api_ns.doc("plugin_upload_file_request")
    @inner_api_ns.doc(description="Request signed URL for file upload through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "Signed URL generated successfully",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestRequestUploadFile):
        # generate signed url
        url = get_signed_file_url_for_plugin(payload.filename, payload.mimetype, tenant_model.id, user_model.id)
        return BaseBackwardsInvocationResponse(data={"url": url}).model_dump()


@inner_api_ns.route("/fetch/app/info")
class PluginFetchAppInfoApi(Resource):
    """
    插件接口：获取应用信息。

    参数:
        user_model (Account | EndUser): 当前请求的用户模型。
        tenant_model (Tenant): 当前租户模型。
        payload (RequestFetchAppInfo): 获取应用信息的请求参数。

    返回:
        JSON response: 包含应用信息或错误信息。
    """

    @setup_required
    @plugin_inner_api_only
    @get_user_tenant
    @plugin_data(payload_type=RequestFetchAppInfo)
    @inner_api_ns.doc("plugin_fetch_app_info")
    @inner_api_ns.doc(description="Fetch application information through plugin interface")
    @inner_api_ns.doc(
        responses={
            200: "App information retrieved successfully",
            401: "Unauthorized - invalid API key",
            404: "Service not available",
        }
    )
    def post(self, user_model: Account | EndUser, tenant_model: Tenant, payload: RequestFetchAppInfo):
        return BaseBackwardsInvocationResponse(
            data=PluginAppBackwardsInvocation.fetch_app_info(payload.app_id, tenant_model.id)
        ).model_dump()
