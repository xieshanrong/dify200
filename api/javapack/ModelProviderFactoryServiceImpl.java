package com.cmic.agentstudio.service.impl;

import com.cmic.agentstudio.common.req.console.agent.AnthropicReq;
import com.cmic.agentstudio.common.req.console.agent.OpenAIEmbeddingReq;
import com.cmic.agentstudio.common.req.console.agent.OpenAIModerationReq;
import com.cmic.agentstudio.common.req.console.agent.OpenAIReq;
import com.cmic.agentstudio.common.req.console.agent.OpenAITTSReq;
import com.cmic.agentstudio.common.req.console.agent.OpenAIWhisperReq;
import com.cmic.agentstudio.common.resp.console.agent.AnthropicResp;
import com.cmic.agentstudio.common.resp.console.agent.OpenAIEmbeddingResp;
import com.cmic.agentstudio.common.resp.console.agent.OpenAIModerationResp;
import com.cmic.agentstudio.common.resp.console.agent.OpenAIResp;
import com.cmic.agentstudio.common.resp.console.agent.OpenAIWhisperResp;
import com.cmic.agentstudio.constant.DefaultConstants;
import com.cmic.agentstudio.constant.ModelProviderConstants;
import com.cmic.agentstudio.constant.ModelTypeConstants;
import com.cmic.agentstudio.core.agent.entities.AnthropicMessage;
import com.cmic.agentstudio.core.agent.entities.AnthropicTool;
import com.cmic.agentstudio.core.agent.entities.OpenAIMessage;
import com.cmic.agentstudio.core.agent.entities.OpenAIModerationResult;
import com.cmic.agentstudio.core.agent.entities.OpenAITool;
import com.cmic.agentstudio.core.entities.RerankResult;
import com.cmic.agentstudio.core.entities.RerankResultItem;
import com.cmic.agentstudio.core.entities.TTSVoice;
import com.cmic.agentstudio.core.entities.TextEmbeddingResult;
import com.cmic.agentstudio.core.modelruntime.callbacks.Callback;
import com.cmic.agentstudio.core.modelruntime.entities.message.PromptMessage;
import com.cmic.agentstudio.core.modelruntime.entities.message.PromptMessageTool;
import com.cmic.agentstudio.service.AnthropicClientService;
import com.cmic.agentstudio.service.ModelProviderFactoryService;
import com.cmic.agentstudio.service.OpenAIClientService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 模型提供者工厂服务实现类
 * 该类负责根据不同的提供商名称路由到相应的模型服务实现
 *
 * @author: xieshanrong
 * @version: 1.0
 */
@Service
@Slf4j
public class ModelProviderFactoryServiceImpl implements ModelProviderFactoryService {

    @Autowired
    private OpenAIClientService openAIClientService;

    @Autowired
    private AnthropicClientService anthropicClientService;

    /**
     * 调用大语言模型的流式接口
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName    模型提供商名称 (如: openai, anthropic, google)
     * @param modelType       模型类型
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return 流式响应对象
     */
    @Override
    public Object invokeLLMStream(String providerName, String modelType, List<PromptMessage> promptMessages, Object modelParameters, List<PromptMessageTool> tools, List<String> stop, String userId, List<Callback> callbacks) {
        try {
            log.debug("Factory invoke LLM stream: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用具体的LLM流式实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI ->
                        invokeOpenAILLMStream(promptMessages, modelParameters, tools, stop, userId, callbacks);
                case ModelProviderConstants.PROVIDER_ANTHROPIC ->
                        invokeAnthropicLLMStream(promptMessages, modelParameters, tools, stop, userId, callbacks);
                case ModelProviderConstants.PROVIDER_GOOGLE ->
                        invokeGoogleLLMStream(promptMessages, modelParameters, tools, stop, userId, callbacks);
                default -> throw new RuntimeException("Unsupported provider: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory LLM stream invoke failed", e);
            throw new RuntimeException("LLM stream invoke failed", e);
        }
    }

    /**
     * 调用大语言模型的阻塞式接口
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName    模型提供商名称 (如: openai, anthropic, google)
     * @param modelType       模型类型
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return 模型响应对象
     */
    @Override
    public Object invokeLLM(String providerName, String modelType, List<PromptMessage> promptMessages, Object modelParameters, List<PromptMessageTool> tools, List<String> stop, String userId, List<Callback> callbacks) {
        try {
            log.debug("Factory invoke LLM: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用具体的LLM实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI ->
                        invokeOpenAILLM(promptMessages, modelParameters, tools, stop, userId, callbacks);
                case ModelProviderConstants.PROVIDER_ANTHROPIC ->
                        invokeAnthropicLLM(promptMessages, modelParameters, tools, stop, userId, callbacks);
                case ModelProviderConstants.PROVIDER_GOOGLE ->
                        invokeGoogleLLM(promptMessages, modelParameters, tools, stop, userId, callbacks);
                default -> throw new RuntimeException("Unsupported provider: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory LLM invoke failed", e);
            throw new RuntimeException("LLM invoke failed", e);
        }
    }

    /**
     * 计算提示消息的Token数量
     * 根据提供商名称路由到对应的Token计算方法
     *
     * @param providerName   模型提供商名称
     * @param modelType      模型类型
     * @param promptMessages 提示消息列表
     * @param tools          工具列表
     * @return Token数量
     */
    @Override
    public int getNumTokens(String providerName, String modelType, List<PromptMessage> promptMessages, List<PromptMessageTool> tools) {
        try {
            log.debug("Factory get num tokens: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的Token计算方法
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> getOpenAINumTokens(promptMessages, tools);
                case ModelProviderConstants.PROVIDER_ANTHROPIC -> getAnthropicNumTokens(promptMessages, tools);
                case ModelProviderConstants.PROVIDER_GOOGLE -> getGoogleNumTokens(promptMessages, tools);
                // 通用计算方法
                default -> calculateGenericNumTokens(promptMessages, tools);
            };

        } catch (Exception e) {
            log.error("Factory get num tokens failed", e);
            return 0;
        }
    }

    /**
     * 调用文本嵌入模型
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: openai, huggingface)
     * @param modelType    模型类型
     * @param texts        待嵌入的文本列表
     * @param user         用户ID
     * @return 文本嵌入结果
     */
    @Override
    public TextEmbeddingResult invokeTextEmbedding(String providerName, String modelType, List<String> texts, String user) {
        try {
            log.debug("Factory invoke text embedding: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的文本嵌入实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> invokeOpenAITextEmbedding(texts, user);
                case ModelProviderConstants.PROVIDER_HUGGINGFACE -> invokeHuggingFaceTextEmbedding(texts, user);
                default -> throw new RuntimeException("Unsupported provider for text embedding: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory text embedding invoke failed", e);
            throw new RuntimeException("Text embedding invoke failed", e);
        }
    }

    /**
     * 调用重排序模型
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: cohere, bge)
     * @param modelType    模型类型
     * @param query        查询文本
     * @param documents    待重排序的文档列表
     * @param user         用户ID
     * @return 重排序结果
     */
    @Override
    public RerankResult invokeRerank(String providerName, String modelType, String query, List<String> documents, String user) {
        try {
            log.debug("Factory invoke rerank: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的重排序实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_COHERE -> invokeCohereRerank(query, documents, user);
                case ModelProviderConstants.PROVIDER_BGE -> invokeBGERerank(query, documents, user);
                default -> throw new RuntimeException("Unsupported provider for rerank: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory rerank invoke failed", e);
            throw new RuntimeException("Rerank invoke failed", e);
        }
    }

    /**
     * 调用文本转语音(TTS)模型
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: openai, azure)
     * @param modelType    模型类型
     * @param contentText  待转换的文本内容
     * @param user         用户ID
     * @return 音频数据字节流
     */
    @Override
    public Iterable<byte[]> invokeTTS(String providerName, String modelType, String contentText, String user) {
        try {
            log.debug("Factory invoke TTS: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的TTS实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> invokeOpenAITTS(contentText, user);
                case ModelProviderConstants.PROVIDER_AZURE -> invokeAzureTTS(contentText, user);
                default -> throw new RuntimeException("Unsupported provider for TTS: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory TTS invoke failed", e);
            throw new RuntimeException("TTS invoke failed", e);
        }
    }

    /**
     * 调用语音转文本(Speech2Text)模型
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: openai, azure)
     * @param modelType    模型类型
     * @param file         音频文件对象
     * @param user         用户ID
     * @return 识别出的文本内容
     */
    @Override
    public String invokeSpeech2Text(String providerName, String modelType, Object file, String user) {
        try {
            log.debug("Factory invoke speech2text: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的语音转文本实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> invokeOpenAISpeech2Text(file, user);
                case ModelProviderConstants.PROVIDER_AZURE -> invokeAzureSpeech2Text(file, user);
                default -> throw new RuntimeException("Unsupported provider for speech2text: " + providerName);
            };

        } catch (Exception e) {
            log.error("Factory speech2text invoke failed", e);
            throw new RuntimeException("Speech2text invoke failed", e);
        }
    }

    /**
     * 获取TTS模型的可用语音列表
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: openai, azure)
     * @param modelType    模型类型
     * @param language     语言代码 (如: en-US)
     * @return 语音列表
     */
    @Override
    public List<TTSVoice> getTTSModelVoices(String providerName, String modelType, String language) {
        try {
            log.debug("Factory get TTS voices: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的获取语音列表实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> getOpenAITTSVoices(language);
                case ModelProviderConstants.PROVIDER_AZURE -> getAzureTTSVoices(language);
                default -> new ArrayList<>();
            };

        } catch (Exception e) {
            log.error("Factory get TTS voices failed", e);
            return new ArrayList<>();
        }
    }

    /**
     * 计算文本嵌入所需的Token数量
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称
     * @param modelType    模型类型
     * @param texts        待计算的文本列表
     * @return 每个文本对应的Token数量列表
     */
    @Override
    public List<Integer> getTextEmbeddingNumTokens(String providerName, String modelType, List<String> texts) {
        try {
            log.debug("Factory get text embedding num tokens: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的Token计算实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> getOpenAITextEmbeddingNumTokens(texts);
                case ModelProviderConstants.PROVIDER_HUGGINGFACE -> getHuggingFaceTextEmbeddingNumTokens(texts);
                // 默认计算方法
                default -> texts.stream().map(text -> text.length() / 4).toList();
            };

        } catch (Exception e) {
            log.error("Factory get text embedding num tokens failed", e);
            // 出错时返回0
            return texts.stream().map(text -> 0).toList();
        }
    }

    /**
     * 调用内容审核模型
     * 根据提供商名称路由到对应的实现方法
     *
     * @param providerName 模型提供商名称 (如: openai, azure)
     * @param modelType    模型类型
     * @param text         待审核的文本内容
     * @param user         用户ID
     * @return 是否通过审核 (true: 不合规, false: 合规)
     */
    @Override
    public boolean invokeModeration(String providerName, String modelType, String text, String user) {
        try {
            log.debug("Factory invoke moderation: provider={}, model={}", providerName, modelType);

            // 根据提供商名称调用对应的内容审核实现
            return switch (providerName) {
                case ModelProviderConstants.PROVIDER_OPENAI -> invokeOpenAIModeration(text, user);
                case ModelProviderConstants.PROVIDER_AZURE -> invokeAzureModeration(text, user);
                // 默认不审核，认为内容合规
                default -> false;
            };

        } catch (Exception e) {
            log.error("Factory moderation invoke failed", e);
            // 出错时默认认为内容合规
            return false;
        }
    }

    /**
     * OpenAI 流式LLM调用实现
     * 对应 Python: openai_model.invoke_llm_stream()
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return OpenAI流式响应对象
     */
    private Object invokeOpenAILLMStream(List<PromptMessage> promptMessages, Object modelParameters,
                                         List<PromptMessageTool> tools, List<String> stop,
                                         String userId, List<Callback> callbacks) {
        try {
            log.debug("Invoking OpenAI LLM stream: model={}, prompts={}",
                    extractModelName(modelParameters), promptMessages.size());

            // 构建OpenAI请求对象
            OpenAIReq request = buildOpenAIRequest(promptMessages, modelParameters, tools, stop, true);

            // 调用OpenAI流式API
            return openAIClientService.streamChatCompletion(request, callbacks);

        } catch (Exception e) {
            log.error("OpenAI LLM stream invocation failed", e);
            throw new RuntimeException("OpenAI LLM stream failed", e);
        }
    }

    /**
     * OpenAI 阻塞式LLM调用实现
     * 对应 Python: openai_model.invoke_llm()
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return OpenAI响应对象
     */
    private Object invokeOpenAILLM(List<PromptMessage> promptMessages, Object modelParameters,
                                   List<PromptMessageTool> tools, List<String> stop,
                                   String userId, List<Callback> callbacks) {
        try {
            log.debug("Invoking OpenAI LLM: model={}, prompts={}",
                    extractModelName(modelParameters), promptMessages.size());

            // 构建OpenAI请求对象
            OpenAIReq request = buildOpenAIRequest(promptMessages, modelParameters, tools, stop, false);

            // 调用OpenAI API
            OpenAIResp response = openAIClientService.chatCompletion(request);

            // 转换为通用响应格式
            return convertOpenAIResponse(response);

        } catch (Exception e) {
            log.error("OpenAI LLM invocation failed", e);
            throw new RuntimeException("OpenAI LLM failed", e);
        }
    }

    /**
     * 计算OpenAI模型的Token数量
     *
     * @param promptMessages 提示消息列表
     * @param tools          工具列表
     * @return Token数量
     */
    private int getOpenAINumTokens(List<PromptMessage> promptMessages, List<PromptMessageTool> tools) {
        // OpenAI token计算实现 - 简化的字符长度估算
        return promptMessages.stream().mapToInt(msg -> msg.getContent().toString().length() / 4).sum();
    }

    /**
     * Anthropic 流式LLM调用实现
     * 对应 Python: anthropic_model.invoke_llm_stream()
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return Anthropic流式响应对象
     */
    private Object invokeAnthropicLLMStream(List<PromptMessage> promptMessages, Object modelParameters,
                                            List<PromptMessageTool> tools, List<String> stop,
                                            String userId, List<Callback> callbacks) {
        // Anthropic流式LLM实现
        try {
            log.debug("Invoking Anthropic LLM stream: model={}, prompts={}",
                    extractModelName(modelParameters), promptMessages.size());

            // 构建Anthropic请求对象
            AnthropicReq request = buildAnthropicRequest(promptMessages, modelParameters, tools, stop, true);

            // 调用Anthropic流式API
            return anthropicClientService.streamMessages(request, callbacks);

        } catch (Exception e) {
            log.error("Anthropic LLM stream invocation failed", e);
            throw new RuntimeException("Anthropic LLM stream failed", e);
        }
    }

    /**
     * Anthropic 阻塞式LLM调用实现
     * 对应 Python: anthropic_model.invoke_llm()
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return Anthropic响应对象
     */
    private Object invokeAnthropicLLM(List<PromptMessage> promptMessages, Object modelParameters,
                                      List<PromptMessageTool> tools, List<String> stop,
                                      String userId, List<Callback> callbacks) {
        // Anthropic阻塞式LLM实现
        try {
            log.debug("Invoking Anthropic LLM: model={}, prompts={}",
                    extractModelName(modelParameters), promptMessages.size());

            // 构建Anthropic请求对象
            AnthropicReq request = buildAnthropicRequest(promptMessages, modelParameters, tools, stop, false);

            // 调用Anthropic API
            AnthropicResp response = anthropicClientService.messages(request);

            // 转换为通用响应格式
            return convertAnthropicResponse(response);

        } catch (Exception e) {
            log.error("Anthropic LLM invocation failed", e);
            throw new RuntimeException("Anthropic LLM failed", e);
        }
    }

    /**
     * 计算Anthropic模型的Token数量
     *
     * @param promptMessages 提示消息列表
     * @param tools          工具列表
     * @return Token数量
     */
    private int getAnthropicNumTokens(List<PromptMessage> promptMessages, List<PromptMessageTool> tools) {
        // Anthropic token计算实现 - 简化的字符长度估算
        return promptMessages.stream().mapToInt(msg -> msg.getContent().toString().length() / 4).sum();
    }

    /**
     * Google 流式LLM调用实现占位符
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return Google流式响应对象占位符
     */
    private Object invokeGoogleLLMStream(List<PromptMessage> promptMessages, Object modelParameters,
                                         List<PromptMessageTool> tools, List<String> stop,
                                         String userId, List<Callback> callbacks) {
        // Google流式LLM实现 - 占位符
        return "Google stream response";
    }

    /**
     * Google 阻塞式LLM调用实现占位符
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param userId          用户ID
     * @param callbacks       回调函数列表
     * @return Google响应对象占位符
     */
    private Object invokeGoogleLLM(List<PromptMessage> promptMessages, Object modelParameters,
                                   List<PromptMessageTool> tools, List<String> stop,
                                   String userId, List<Callback> callbacks) {
        // Google阻塞式LLM实现 - 占位符
        return "Google response";
    }

    /**
     * 计算Google模型的Token数量
     *
     * @param promptMessages 提示消息列表
     * @param tools          工具列表
     * @return Token数量
     */
    private int getGoogleNumTokens(List<PromptMessage> promptMessages, List<PromptMessageTool> tools) {
        // Google token计算实现 - 简化的字符长度估算
        return promptMessages.stream().mapToInt(msg -> msg.getContent().toString().length() / 4).sum();
    }

    /**
     * 通用Token计算方法
     *
     * @param promptMessages 提示消息列表
     * @param tools          工具列表
     * @return Token数量
     */
    private int calculateGenericNumTokens(List<PromptMessage> promptMessages, List<PromptMessageTool> tools) {
        // 通用token计算实现 - 简化的字符长度估算
        return promptMessages.stream().mapToInt(msg -> msg.getContent().toString().length() / 4).sum();
    }

    /**
     * OpenAI 文本嵌入调用实现
     *
     * @param texts 待嵌入的文本列表
     * @param user  用户ID
     * @return 文本嵌入结果
     */
    private TextEmbeddingResult invokeOpenAITextEmbedding(List<String> texts, String user) {
        try {
            log.debug("Invoking OpenAI text embedding: texts={}", texts.size());

            // 构建OpenAI嵌入请求
            OpenAIEmbeddingReq request = new OpenAIEmbeddingReq();
            request.setModel("text-embedding-ada-002");
            request.setInput(texts);

            // 调用OpenAI嵌入API
            OpenAIEmbeddingResp response = openAIClientService.embeddings(request);

            // 转换为通用格式
            TextEmbeddingResult result = new TextEmbeddingResult();
            result.setEmbeddings(response.getData().stream()
                    .map(item -> item.getEmbedding())
                    .toList());
            result.setModel(response.getModel());
            result.setUsagePromptTokens(response.getUsage().getPromptTokens());
            result.setUsageTotalTokens(response.getUsage().getTotalTokens());

            return result;

        } catch (Exception e) {
            log.error("OpenAI text embedding invocation failed", e);
            throw new RuntimeException("OpenAI text embedding failed", e);
        }
    }

    /**
     * HuggingFace 文本嵌入调用实现
     *
     * @param texts 待嵌入的文本列表
     * @param user  用户ID
     * @return 文本嵌入结果
     */
    private TextEmbeddingResult invokeHuggingFaceTextEmbedding(List<String> texts, String user) {
        // HuggingFace文本嵌入实现 - 示例数据
        TextEmbeddingResult result = new TextEmbeddingResult();
        result.setEmbeddings(texts.stream().map(text -> Arrays.asList(0.4f, 0.5f, 0.6f)).toList());
        result.setModel(ModelTypeConstants.MODEL_TYPE_ALL_MINILM_L6_V2);
        result.setUsagePromptTokens(texts.stream().mapToInt(text -> text.length() / 4).sum());
        result.setUsageTotalTokens(texts.stream().mapToInt(text -> text.length() / 4).sum());
        return result;
    }

    /**
     * Cohere 重排序调用实现
     *
     * @param query     查询文本
     * @param documents 待重排序的文档列表
     * @param user      用户ID
     * @return 重排序结果
     */
    private RerankResult invokeCohereRerank(String query, List<String> documents, String user) {
        // Cohere重排序实现 - 示例数据
        List<RerankResultItem> results = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            results.add(new RerankResultItem(i, (double) (0.8f - (i * 0.1f)), documents.get(i)));
        }
        return new RerankResult(results, ModelTypeConstants.MODEL_TYPE_RERANK_ENGLISH_V2);
    }

    /**
     * BGE 重排序调用实现
     *
     * @param query     查询文本
     * @param documents 待重排序的文档列表
     * @param user      用户ID
     * @return 重排序结果
     */
    private RerankResult invokeBGERerank(String query, List<String> documents, String user) {
        // BGE重排序实现 - 示例数据
        List<RerankResultItem> results = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            results.add(new RerankResultItem(i, (double) (0.9f - (i * 0.05f)), documents.get(i)));
        }
        return new RerankResult(results, ModelTypeConstants.MODEL_TYPE_BGE_RERANKER_V2_M3);
    }

    /**
     * OpenAI TTS调用
     *
     * @param contentText 待转换的文本内容
     * @param user        用户ID
     * @return 音频数据字节流
     */
    private Iterable<byte[]> invokeOpenAITTS(String contentText, String user) {
        try {
            log.debug("Invoking OpenAI TTS: text_length={}", contentText.length());

            // 构建OpenAI TTS请求
            OpenAITTSReq request = new OpenAITTSReq();
            request.setModel("tts-1");
            request.setInput(contentText);
            request.setVoice("alloy");
            request.setResponseFormat("mp3");
            request.setSpeed(1.0);

            // 调用OpenAI TTS API
            byte[] audioData = openAIClientService.textToSpeech(request);

            // 返回音频流
            return Collections.singletonList(audioData);

        } catch (Exception e) {
            log.error("OpenAI TTS invocation failed", e);
            throw new RuntimeException("OpenAI TTS failed", e);
        }
    }

    /**
     * Azure TTS调用实现
     *
     * @param contentText 待转换的文本内容
     * @param user        用户ID
     * @return 音频数据字节流
     */
    private Iterable<byte[]> invokeAzureTTS(String contentText, String user) {
        try {
            log.debug("Invoking Azure TTS: text_length={}", contentText.length());

            // 构建Azure TTS请求
            AzureTTSReq request = new AzureTTSReq();
            request.setText(contentText);
            request.setVoice("en-US-JennyNeural");
            request.setLanguage("en-US");
            request.setOutputFormat("audio-16khz-128kbitrate-mono-mp3");

            // 调用Azure TTS API
            byte[] audioData = azureClientService.textToSpeech(request);

            // 返回音频流
            return Collections.singletonList(audioData);

        } catch (Exception e) {
            log.error("Azure TTS invocation failed", e);
            throw new RuntimeException("Azure TTS failed", e);
        }
    }

    /**
     * OpenAI 语音转文本调用实现
     *
     * @param file 音频文件对象
     * @param user 用户ID
     * @return 识别出的文本内容
     */
    private String invokeOpenAISpeech2Text(Object file, String user) {
        try {
            log.debug("Invoking OpenAI Speech2Text: user={}", user);

            // 转换文件为字节数组
            byte[] audioData = convertToByteArray(file);

            // 构建OpenAI Whisper请求
            OpenAIWhisperReq request = new OpenAIWhisperReq();
            request.setFile(audioData);
            request.setModel("whisper-1");
            request.setLanguage("en");
            request.setResponseFormat("json");

            // 调用OpenAI Whisper API
            OpenAIWhisperResp response = openAIClientService.speechToText(request);

            return response.getText();

        } catch (Exception e) {
            log.error("OpenAI Speech2Text invocation failed", e);
            throw new RuntimeException("OpenAI Speech2Text failed", e);
        }
    }

    /**
     * Azure 语音转文本调用实现
     *
     * @param file 音频文件对象
     * @param user 用户ID
     * @return 识别出的文本内容
     */
    private String invokeAzureSpeech2Text(Object file, String user) {
        // Azure语音转文本实现 - 示例数据
        return "Azure speech to text result";
    }

    /**
     * 获取OpenAI TTS模型的可用语音列表
     *
     * @param language 语言代码
     * @return 语音列表
     */
    private List<TTSVoice> getOpenAITTSVoices(String language) {
        // OpenAI TTS语音列表实现 - 示例数据
        List<TTSVoice> voices = new ArrayList<>();
        voices.add(new TTSVoice(ModelProviderConstants.TTS_VOICE_ALLOY, "Alloy", ModelProviderConstants.DEFAULT_LANGUAGE));
        voices.add(new TTSVoice(ModelProviderConstants.TTS_VOICE_ECHO, "Echo", ModelProviderConstants.DEFAULT_LANGUAGE));
        voices.add(new TTSVoice(ModelProviderConstants.TTS_VOICE_FABLE, "Fable", ModelProviderConstants.DEFAULT_LANGUAGE));
        return voices;
    }

    /**
     * 获取Azure TTS模型的可用语音列表
     *
     * @param language 语言代码
     * @return 语音列表
     */
    private List<TTSVoice> getAzureTTSVoices(String language) {
        // Azure TTS语音列表实现 - 示例数据
        List<TTSVoice> voices = new ArrayList<>();
        voices.add(new TTSVoice(ModelProviderConstants.TTS_VOICE_JENNY_NEURAL, "Jenny", ModelProviderConstants.DEFAULT_LANGUAGE));
        voices.add(new TTSVoice(ModelProviderConstants.TTS_VOICE_GUY_NEURAL, "Guy", ModelProviderConstants.DEFAULT_LANGUAGE));
        return voices;
    }

    /**
     * 计算OpenAI文本嵌入所需的Token数量
     *
     * @param texts 待计算的文本列表
     * @return 每个文本对应的Token数量列表
     */
    private List<Integer> getOpenAITextEmbeddingNumTokens(List<String> texts) {
        // OpenAI文本嵌入token计算 - 简化的字符长度估算
        return texts.stream().map(text -> text.length() / 4).toList();
    }

    /**
     * 计算HuggingFace文本嵌入所需的Token数量
     *
     * @param texts 待计算的文本列表
     * @return 每个文本对应的Token数量列表
     */
    private List<Integer> getHuggingFaceTextEmbeddingNumTokens(List<String> texts) {
        // HuggingFace文本嵌入token计算 - 简化的字符长度估算
        return texts.stream().map(text -> text.length() / 4).toList();
    }

    /**
     * OpenAI 内容审核调用实现
     *
     * @param text 待审核的文本内容
     * @param user 用户ID
     * @return 是否通过审核 (true: 不合规, false: 合规)
     */
    private boolean invokeOpenAIModeration(String text, String user) {
        try {
            log.debug("Invoking OpenAI moderation: text_length={}", text.length());

            // 构建OpenAI审核请求
            OpenAIModerationReq request = new OpenAIModerationReq();
            request.setInput(List.of(text));
            request.setModel("text-moderation-latest");

            // 调用OpenAI审核API
            OpenAIModerationResp response = openAIClientService.moderation(request);

            // 检查是否有违规内容
            if (response.getResults() != null && !response.getResults().isEmpty()) {
                OpenAIModerationResult result = response.getResults().get(0);
                return result.getFlagged();
            }

            return false;

        } catch (Exception e) {
            log.error("OpenAI moderation invocation failed", e);
            // 默认不违规
            return false;
        }
    }

    /**
     * Azure 内容审核调用实现
     *
     * @param text 待审核的文本内容
     * @param user 用户ID
     * @return 是否通过审核 (true: 不合规, false: 合规)
     */
    private boolean invokeAzureModeration(String text, String user) {
        // Azure内容审核实现 - 简化实现
        return text.toLowerCase().contains("inappropriate");
    }

    /**
     * 转换OpenAI响应为通用格式
     *
     * @param response OpenAI原始响应对象
     * @return 通用格式的响应对象
     */
    private Object convertOpenAIResponse(OpenAIResp response) {
        // 转换OpenAI响应为通用格式
        Map<String, Object> result = new HashMap<>();
        if (response.getChoices() != null && !response.getChoices().isEmpty()) {
            result.put("content", response.getChoices().get(0).getMessage().getContent());
            result.put("finishReason", response.getChoices().get(0).getFinishReason());
        }
        result.put("usage", response.getUsage());
        result.put("model", response.getModel());
        result.put("id", response.getId());
        return result;
    }

    /**
     * 构建Anthropic请求对象
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param stream          是否为流式请求
     * @return Anthropic请求对象
     */
    private AnthropicReq buildAnthropicRequest(List<PromptMessage> promptMessages, Object modelParameters,
                                               List<PromptMessageTool> tools, List<String> stop, boolean stream) {
        AnthropicReq request = new AnthropicReq();
        request.setModel(extractModelName(modelParameters));
        request.setMessages(convertToAnthropicMessages(promptMessages));
        request.setTools(convertToAnthropicTools(tools));
        request.setMaxTokens(DefaultConstants.DEFAULT_MAX_TOKENS);
        request.setStream(stream);

        // 设置其他参数
//        if (modelParameters instanceof Map) {
//            Map<?, ?> params = (Map<?, ?>) modelParameters;
//            request.setTemperature((Double) params.getOrDefault("temperature", DEFAULT_TEMPERATURE));
//            request.setTopP((Double) params.getOrDefault("top_p", DEFAULT_TOP_P));
//            request.setTopK((Integer) params.getOrDefault("top_k", DEFAULT_TOP_K));
//        }

        return request;
    }

    /**
     * 从模型参数中提取模型名称
     *
     * @param modelParameters 模型参数
     * @return 模型名称
     */
    private String extractModelName(Object modelParameters) {
        if (modelParameters instanceof Map) {
            Map<?, ?> params = (Map<?, ?>) modelParameters;
            Object model = params.get("model");
            return model != null ? model.toString() : ModelProviderConstants.DEFAULT_OPENAI_MODEL;
        }
        return ModelProviderConstants.DEFAULT_OPENAI_MODEL;
    }

    /**
     * 转换提示消息为Anthropic消息格式
     *
     * @param messages 提示消息列表
     * @return Anthropic消息列表
     */
    private List<AnthropicMessage> convertToAnthropicMessages(List<PromptMessage> messages) {
        return messages.stream().map(msg -> {
            AnthropicMessage anthropicMsg = new AnthropicMessage();
            anthropicMsg.setRole(msg.getRole().getValue());
            anthropicMsg.setContent(msg.getContent().toString());
            return anthropicMsg;
        }).toList();
    }

    /**
     * 转换工具为Anthropic工具格式
     *
     * @param tools 工具列表
     * @return Anthropic工具列表
     */
    private List<AnthropicTool> convertToAnthropicTools(List<PromptMessageTool> tools) {
        if (tools == null) {
            return new ArrayList<>();
        }
        return tools.stream().map(tool -> {
            AnthropicTool anthropicTool = new AnthropicTool();
            anthropicTool.setName(tool.getName());
            anthropicTool.setDescription(tool.getDescription());
            anthropicTool.setInputSchema(tool.getParameters());
            return anthropicTool;
        }).toList();
    }

    /**
     * 构建OpenAI请求对象
     *
     * @param promptMessages  提示消息列表
     * @param modelParameters 模型参数
     * @param tools           工具列表
     * @param stop            停止词列表
     * @param stream          是否为流式请求
     * @return OpenAI请求对象
     */
    private OpenAIReq buildOpenAIRequest(List<PromptMessage> promptMessages, Object modelParameters,
                                         List<PromptMessageTool> tools, List<String> stop, boolean stream) {
        OpenAIReq request = new OpenAIReq();
        request.setModel(extractModelName(modelParameters));
        request.setMessages(convertToOpenAIMessages(promptMessages));
        request.setTools(convertToOpenAITools(tools));
        request.setStop(stop);
        request.setStream(stream);


//        // 设置其他参数
//        if (modelParameters instanceof Map) {
//            Map<?, ?> params = (Map<?, ?>) modelParameters;
//
//            Object temperatureObj = params.getOrDefault("temperature", DEFAULT_TEMPERATURE);
//            Object maxTokensObj = params.getOrDefault("max_tokens", DefaultConstants.DEFAULT_MAX_TOKENS);
//            Object topPObj = params.getOrDefault("top_p", DEFAULT_TOP_P);
//            Object frequencyPenaltyObj = params.getOrDefault("frequency_penalty", DEFAULT_FREQUENCY_PENALTY);
//            Object presencePenaltyObj = params.getOrDefault("presence_penalty", DEFAULT_PRESENCE_PENALTY);
//
//            // 安全地转换类型
//            if (temperatureObj instanceof Number) {
//                request.setTemperature(((Number) temperatureObj).doubleValue());
//            }
//            if (maxTokensObj instanceof Number) {
//                request.setMaxTokens(((Number) maxTokensObj).intValue());
//            }
//            if (topPObj instanceof Number) {
//                request.setTopP(((Number) topPObj).doubleValue());
//            }
//            if (frequencyPenaltyObj instanceof Number) {
//                request.setFrequencyPenalty(((Number) frequencyPenaltyObj).doubleValue());
//            }
//            if (presencePenaltyObj instanceof Number) {
//                request.setPresencePenalty(((Number) presencePenaltyObj).doubleValue());
//            }

        return request;
    }

    /**
     * 转换提示消息为OpenAI消息格式
     *
     * @param messages 提示消息列表
     * @return OpenAI消息列表
     */
    private List<OpenAIMessage> convertToOpenAIMessages(List<PromptMessage> messages) {
        return messages.stream().map(msg -> {
            OpenAIMessage openaiMsg = new OpenAIMessage();
            openaiMsg.setRole(msg.getRole().getValue());
            openaiMsg.setContent(msg.getContent().toString());
            return openaiMsg;
        }).toList();
    }

    /**
     * 转换工具为OpenAI工具格式
     *
     * @param tools 工具列表
     * @return OpenAI工具列表
     */
    private List<OpenAITool> convertToOpenAITools(List<PromptMessageTool> tools) {
        if (tools == null) return new ArrayList<>();
        return tools.stream().map(tool -> {
            OpenAITool openaiTool = new OpenAITool();
//            openaiTool.setType(tool.getType());
//            openaiTool.setFunction(tool.getFunction());
            return openaiTool;
        }).toList();
    }

    /**
     * 转换Anthropic响应为通用格式
     *
     * @param response Anthropic原始响应对象
     * @return 通用格式的响应对象
     */
    private Object convertAnthropicResponse(AnthropicResp response) {
        // 转换Anthropic响应为通用格式
        Map<String, Object> result = new HashMap<>();
        if (response.getContent() != null && !response.getContent().isEmpty()) {
            result.put("content", response.getContent().get(0).getText());
            result.put("stopReason", response.getStopReason());
        }
        result.put("usage", response.getUsage());
        result.put("model", response.getModel());
        result.put("id", response.getId());
        return result;
    }

    /**
     * 转换为字节数组
     */
    private byte[] convertToByteArray(Object file) {
        if (file instanceof byte[]) {
            return (byte[]) file;
        } else if (file instanceof String) {
            return ((String) file).getBytes();
        } else {
            throw new RuntimeException("Unsupported file type: " + file.getClass());
        }
    }
}
