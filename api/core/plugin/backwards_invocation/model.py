import tempfile
from binascii import hexlify, unhexlify
from collections.abc import Generator

from core.llm_generator.output_parser.structured_output import invoke_llm_with_structured_output
from core.model_manager import ModelManager
from core.model_runtime.entities.llm_entities import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMResultChunkWithStructuredOutput,
    LLMResultWithStructuredOutput,
)
from core.model_runtime.entities.message_entities import (
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from core.plugin.backwards_invocation.base import BaseBackwardsInvocation
from core.plugin.entities.request import (
    RequestInvokeLLM,
    RequestInvokeLLMWithStructuredOutput,
    RequestInvokeModeration,
    RequestInvokeRerank,
    RequestInvokeSpeech2Text,
    RequestInvokeSummary,
    RequestInvokeTextEmbedding,
    RequestInvokeTTS,
)
from core.tools.entities.tool_entities import ToolProviderType
from core.tools.utils.model_invocation_utils import ModelInvocationUtils
from core.workflow.nodes.llm import llm_utils
from models.account import Tenant


class PluginModelBackwardsInvocation(BaseBackwardsInvocation):
    """
    插件模型反向调用类

    该类提供了各种AI模型服务的调用接口，包括大语言模型(LLM)、文本嵌入、重排序、文本转语音(TTS)、
    语音转文本(Speech2Text)、内容审核(Moderation)等功能。所有方法都是类方法，可以直接通过类调用。
    """

    @classmethod
    def invoke_llm(
        cls, user_id: str, tenant: Tenant, payload: RequestInvokeLLM
    ) -> Generator[LLMResultChunk, None, None] | LLMResult:
        """
        调用大语言模型

        根据请求参数获取对应的模型实例并调用大语言模型，支持流式和非流式响应两种模式。
        对于流式响应，会逐个处理结果块并扣除相应的配额；对于非流式响应，直接返回完整结果并扣除配额。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeLLM): LLM调用请求参数对象，包含提供者、模型类型、模型名称等信息

        返回:
            Generator[LLMResultChunk, None, None] | LLMResult:
            LLM调用结果，根据stream参数决定是流式响应(Generator)还是非流式响应(LLMResult)
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 调用模型执行推理
        response = model_instance.invoke_llm(
            prompt_messages=payload.prompt_messages,
            model_parameters=payload.completion_params,
            tools=payload.tools,
            stop=payload.stop,
            stream=True if payload.stream is None else payload.stream,
            user=user_id,
        )

        if isinstance(response, Generator):
            # 处理流式响应：逐个处理结果块并扣除配额

            def handle() -> Generator[LLMResultChunk, None, None]:
                for chunk in response:
                    # 如果当前块包含使用量信息，则扣除相应配额
                    if chunk.delta.usage:
                        llm_utils.deduct_llm_quota(
                            tenant_id=tenant.id, model_instance=model_instance, usage=chunk.delta.usage
                        )
                    # 清空提示消息以减少传输数据量
                    chunk.prompt_messages = []
                    yield chunk

            return handle()
        else:
            # 处理非流式响应：直接扣除配额并返回结果
            if response.usage:
                llm_utils.deduct_llm_quota(tenant_id=tenant.id, model_instance=model_instance, usage=response.usage)

            def handle_non_streaming(response: LLMResult) -> Generator[LLMResultChunk, None, None]:
                # 将非流式响应转换为流式响应格式
                yield LLMResultChunk(
                    model=response.model,
                    prompt_messages=[],
                    system_fingerprint=response.system_fingerprint,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=response.message,
                        usage=response.usage,
                        finish_reason="",
                    ),
                )

            return handle_non_streaming(response)

    @classmethod
    def invoke_llm_with_structured_output(
        cls, user_id: str, tenant: Tenant, payload: RequestInvokeLLMWithStructuredOutput
    ):
        """
        调用具有结构化输出的大语言模型

        调用支持结构化输出的LLM，可以按照指定的JSON Schema格式返回结果，确保输出符合预定义的结构。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeLLMWithStructuredOutput): 结构化输出LLM调用请求参数对象

        返回:
            结构化输出的LLM调用结果，支持流式和非流式响应
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 获取模型Schema信息
        model_schema = model_instance.model_type_instance.get_model_schema(payload.model, model_instance.credentials)

        if not model_schema:
            raise ValueError(f"Model schema not found for {payload.model}")

        # 调用支持结构化输出的LLM
        response = invoke_llm_with_structured_output(
            provider=payload.provider,
            model_schema=model_schema,
            model_instance=model_instance,
            prompt_messages=payload.prompt_messages,
            json_schema=payload.structured_output_schema,
            tools=payload.tools,
            stop=payload.stop,
            stream=True if payload.stream is None else payload.stream,
            user=user_id,
            model_parameters=payload.completion_params,
        )

        if isinstance(response, Generator):
            # 处理流式响应

            def handle() -> Generator[LLMResultChunkWithStructuredOutput, None, None]:
                for chunk in response:
                    # 如果当前块包含使用量信息，则扣除相应配额
                    if chunk.delta.usage:
                        llm_utils.deduct_llm_quota(
                            tenant_id=tenant.id, model_instance=model_instance, usage=chunk.delta.usage
                        )
                    # 清空提示消息以减少传输数据量
                    chunk.prompt_messages = []
                    yield chunk

            return handle()
        else:
            # 处理非流式响应
            if response.usage:
                llm_utils.deduct_llm_quota(tenant_id=tenant.id, model_instance=model_instance, usage=response.usage)

            def handle_non_streaming(
                response: LLMResultWithStructuredOutput,
            ) -> Generator[LLMResultChunkWithStructuredOutput, None, None]:
                # 将非流式响应转换为流式响应格式
                yield LLMResultChunkWithStructuredOutput(
                    model=response.model,
                    prompt_messages=[],
                    system_fingerprint=response.system_fingerprint,
                    structured_output=response.structured_output,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=response.message,
                        usage=response.usage,
                        finish_reason="",
                    ),
                )

            return handle_non_streaming(response)

    @classmethod
    def invoke_text_embedding(cls, user_id: str, tenant: Tenant, payload: RequestInvokeTextEmbedding):
        """
        调用文本嵌入模型

        将文本转换为向量表示，用于语义相似度计算、聚类、推荐系统等任务。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeTextEmbedding): 文本嵌入调用请求参数对象，包含待转换的文本列表

        返回:
            文本嵌入结果，包含每个文本对应的向量表示
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 调用文本嵌入模型
        response = model_instance.invoke_text_embedding(
            texts=payload.texts,
            user=user_id,
        )

        return response

    @classmethod
    def invoke_rerank(cls, user_id: str, tenant: Tenant, payload: RequestInvokeRerank):
        """
        调用重排序模型

        对文档列表进行重新排序，提高与查询相关性高的文档的排名，常用于搜索结果优化。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeRerank): 重排序调用请求参数对象，包含查询语句、文档列表等信息

        返回:
            重排序结果，包含每个文档的相关性得分和排序后的位置
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 调用重排序模型
        response = model_instance.invoke_rerank(
            query=payload.query,
            docs=payload.docs,
            score_threshold=payload.score_threshold,
            top_n=payload.top_n,
            user=user_id,
        )

        return response

    @classmethod
    def invoke_tts(cls, user_id: str, tenant: Tenant, payload: RequestInvokeTTS):
        """
        调用文本转语音模型

        将文本转换为语音输出，支持不同的语音风格和语言。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeTTS): TTS调用请求参数对象，包含待转换的文本和语音参数

        返回:
            Generator[dict, None, None]: 语音数据生成器，数据以十六进制字符串形式返回
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 调用文本转语音模型
        response = model_instance.invoke_tts(
            content_text=payload.content_text,
            tenant_id=tenant.id,
            voice=payload.voice,
            user=user_id,
        )

        # 将语音数据转换为十六进制字符串格式
        def handle() -> Generator[dict, None, None]:
            for chunk in response:
                yield {"result": hexlify(chunk).decode("utf-8")}

        return handle()

    @classmethod
    def invoke_speech2text(cls, user_id: str, tenant: Tenant, payload: RequestInvokeSpeech2Text):
        """
        调用语音转文本模型

        将语音文件转换为文本内容，支持多种音频格式。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeSpeech2Text): 语音转文本调用请求参数对象，包含音频文件数据

        返回:
            dict: 包含转换结果的字典，键为"result"，值为转换后的文本
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 创建临时文件存储音频数据
        with tempfile.NamedTemporaryFile(suffix=".mp3", mode="wb", delete=True) as temp:
            # 将十六进制字符串转换为二进制数据并写入临时文件
            temp.write(unhexlify(payload.file))
            temp.flush()
            temp.seek(0)

            # 调用语音转文本模型
            response = model_instance.invoke_speech2text(
                file=temp,
                user=user_id,
            )

            return {
                "result": response,
            }

    @classmethod
    def invoke_moderation(cls, user_id: str, tenant: Tenant, payload: RequestInvokeModeration):
        """
        调用内容审核模型

        对文本内容进行审核，检测是否包含违规内容，如暴力、色情、敏感信息等。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeModeration): 内容审核调用请求参数对象，包含待审核的文本

        返回:
            dict: 包含审核结果的字典，键为"result"，值为审核结果对象
        """
        # 根据请求参数获取模型实例
        model_instance = ModelManager().get_model_instance(
            tenant_id=tenant.id,
            provider=payload.provider,
            model_type=payload.model_type,
            model=payload.model,
        )

        # 调用内容审核模型
        response = model_instance.invoke_moderation(
            text=payload.text,
            user=user_id,
        )

        return {
            "result": response,
        }

    @classmethod
    def get_system_model_max_tokens(cls, tenant_id: str) -> int:
        """
        获取系统模型最大token数

        获取指定租户的系统模型支持的最大上下文token数量，用于控制输入文本长度。

        参数:
            tenant_id (str): 租户ID

        返回:
            int: 系统模型支持的最大token数
        """
        return ModelInvocationUtils.get_max_llm_context_tokens(tenant_id=tenant_id)

    @classmethod
    def get_prompt_tokens(cls, tenant_id: str, prompt_messages: list[PromptMessage]) -> int:
        """
        计算提示消息的token数量

        根据模型的token化规则计算提示消息列表的token数量，用于控制输入长度。

        参数:
            tenant_id (str): 租户ID
            prompt_messages (list[PromptMessage]): 提示消息列表

        返回:
            int: 提示消息的token数量
        """
        return ModelInvocationUtils.calculate_tokens(tenant_id=tenant_id, prompt_messages=prompt_messages)

    @classmethod
    def invoke_system_model(
        cls,
        user_id: str,
        tenant: Tenant,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """
        调用系统模型

        调用系统默认的LLM模型执行推理任务，用于系统内部处理。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            prompt_messages (list[PromptMessage]): 提示消息列表

        返回:
            LLMResult: LLM调用结果
        """
        return ModelInvocationUtils.invoke(
            user_id=user_id,
            tenant_id=tenant.id,
            tool_type=ToolProviderType.PLUGIN,
            tool_name="plugin",
            prompt_messages=prompt_messages,
        )

    @classmethod
    def invoke_summary(cls, user_id: str, tenant: Tenant, payload: RequestInvokeSummary):
        """
        调用文本摘要功能

        对长文本进行摘要处理，如果文本过长会自动分段处理并合并结果，确保摘要的完整性和连贯性。

        参数:
            user_id (str): 用户ID，用于标识调用用户
            tenant (Tenant): 租户对象，包含租户相关信息
            payload (RequestInvokeSummary): 文本摘要调用请求参数对象，包含待摘要的文本和额外指令

        返回:
            str: 摘要文本，如果输入文本较短则直接返回原文
        """
        # 获取系统模型最大token数
        max_tokens = cls.get_system_model_max_tokens(tenant_id=tenant.id)
        content = payload.text

        # 定义摘要提示模板
        SUMMARY_PROMPT = """You are a professional language researcher, you are interested in the language
and you can quickly aimed at the main point of an webpage and reproduce it in your own words but
retain the original meaning and keep the key points.
however, the text you got is too long, what you got is possible a part of the text.
Please summarize the text you got.

Here is the extra instruction you need to follow:
<extra_instruction>
{payload.instruction}
</extra_instruction>
"""

        # 如果文本长度小于最大token数的60%，则直接返回原文
        if (
            cls.get_prompt_tokens(
                tenant_id=tenant.id,
                prompt_messages=[UserPromptMessage(content=content)],
            )
            < max_tokens * 0.6
        ):
            return content

        # 计算提示token数的内部函数
        def get_prompt_tokens(content: str) -> int:
            return cls.get_prompt_tokens(
                tenant_id=tenant.id,
                prompt_messages=[
                    SystemPromptMessage(content=SUMMARY_PROMPT.replace("{payload.instruction}", payload.instruction)),
                    UserPromptMessage(content=content),
                ],
            )

        # 执行摘要的内部函数
        def summarize(content: str) -> str:
            summary = cls.invoke_system_model(
                user_id=user_id,
                tenant=tenant,
                prompt_messages=[
                    SystemPromptMessage(content=SUMMARY_PROMPT.replace("{payload.instruction}", payload.instruction)),
                    UserPromptMessage(content=content),
                ],
            )

            assert isinstance(summary.message.content, str)
            return summary.message.content

        # 按行分割文本
        lines = content.split("\n")
        new_lines: list[str] = []
        # 将长行分割成多个较短的行
        for i in range(len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            if len(line) < max_tokens * 0.5:
                new_lines.append(line)
            elif get_prompt_tokens(line) > max_tokens * 0.7:
                # 如果单行过长，继续分割直到满足token限制
                while get_prompt_tokens(line) > max_tokens * 0.7:
                    new_lines.append(line[: int(max_tokens * 0.5)])
                    line = line[int(max_tokens * 0.5) :]
                new_lines.append(line)
            else:
                new_lines.append(line)

        # 将行合并成满足token限制的消息块
        messages: list[str] = []
        for line in new_lines:
            if len(messages) == 0:
                messages.append(line)
            else:
                if len(messages[-1]) + len(line) < max_tokens * 0.5:
                    messages[-1] += line
                if get_prompt_tokens(messages[-1] + line) > max_tokens * 0.7:
                    messages.append(line)
                else:
                    messages[-1] += line

        # 对每个消息块执行摘要
        summaries = []
        for i in range(len(messages)):
            message = messages[i]
            summary = summarize(message)
            summaries.append(summary)

        # 合并所有摘要结果
        result = "\n".join(summaries)

        # 如果合并后的摘要仍然过长，则递归调用摘要功能
        if (
            cls.get_prompt_tokens(
                tenant_id=tenant.id,
                prompt_messages=[UserPromptMessage(content=result)],
            )
            > max_tokens * 0.7
        ):
            return cls.invoke_summary(
                user_id=user_id,
                tenant=tenant,
                payload=RequestInvokeSummary(text=result, instruction=payload.instruction),
            )

        return result
