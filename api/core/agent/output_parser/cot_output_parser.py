import json
import re
from collections.abc import Generator
from typing import Union

from core.agent.entities import AgentScratchpadUnit
from core.model_runtime.entities.llm_entities import LLMResultChunk


class CotAgentOutputParser:
    """
    思维链(Chain-of-Thought)代理输出解析器类。

    该类用于解析基于ReAct框架的思维链代理的流式输出，能够识别和提取
    模型生成的思考过程(Thought)和行动(Action)信息，并将其转换为结构化数据。
    """

    @classmethod
    def handle_react_stream_output(
        cls, llm_response: Generator[LLMResultChunk, None, None], usage_dict: dict
    ) -> Generator[Union[str, AgentScratchpadUnit.Action], None, None]:
        """
        处理ReAct框架的流式输出，解析其中的思考和行动信息。

        该方法能够实时解析大语言模型的流式输出，识别其中的代码块、JSON格式数据，
        以及普通文本内容，并根据内容类型生成相应的字符串或Action对象。

        Args:
            llm_response (Generator[LLMResultChunk, None, None]): 大语言模型的流式响应生成器
            usage_dict (dict): 用于存储模型使用量信息的字典

        Yields:
            Generator[Union[str, AgentScratchpadUnit.Action], None, None]: 解析后的字符串内容或Action对象生成器
        """

        def parse_action(action) -> Union[str, AgentScratchpadUnit.Action]:
            """
            解析行动信息，将其转换为AgentScratchpadUnit.Action对象。

            该函数能够处理不同格式的行动输入，包括字符串、字典和列表格式，
            并从中提取行动名称和行动参数。

            Args:
                action: 行动信息，可以是字符串、字典或列表格式

            Returns:
                Union[str, AgentScratchpadUnit.Action]: 解析后的字符串或Action对象
            """
            action_name = None
            action_input = None

            # 如果是字符串格式，尝试解析为JSON
            if isinstance(action, str):
                try:
                    action = json.loads(action, strict=False)
                except json.JSONDecodeError:
                    return action or ""

            # Cohere模型总是返回列表格式
            if isinstance(action, list) and len(action) == 1:
                action = action[0]

            # 遍历键值对，识别行动名称和输入参数
            for key, value in action.items():
                if "input" in key.lower():
                    action_input = value
                else:
                    action_name = value

            # 如果同时存在行动名称和输入参数，则创建Action对象
            if action_name is not None and action_input is not None:
                return AgentScratchpadUnit.Action(
                    action_name=action_name,
                    action_input=action_input,
                )
            else:
                return json.dumps(action)

        def extra_json_from_code_block(code_block) -> list[Union[list, dict]]:
            """
            从代码块中提取JSON数据。

            使用正则表达式从代码块中识别和提取JSON格式的数据，支持多种代码块格式。

            Args:
                code_block (str): 包含代码块内容的字符串

            Returns:
                list[Union[list, dict]]: 提取出的JSON对象列表
            """
            # 使用正则表达式查找代码块中的JSON内容
            blocks = re.findall(r"```[json]*\s*([\[{].*[]}])\s*```", code_block, re.DOTALL | re.IGNORECASE)
            if not blocks:
                return []
            try:
                json_blocks = []
                for block in blocks:
                    # 移除可能存在的前缀文字
                    json_text = re.sub(r"^[a-zA-Z]+\n", "", block.strip(), flags=re.MULTILINE)
                    json_blocks.append(json.loads(json_text, strict=False))
                return json_blocks
            except:
                return []

        # 初始化解析状态和缓存变量
        code_block_cache = ""           # 代码块内容缓存
        code_block_delimiter_count = 0  # 代码块分隔符计数
        in_code_block = False           # 是否在代码块内
        json_cache = ""                 # JSON内容缓存
        json_quote_count = 0            # JSON大括号计数
        in_json = False                 # 是否在JSON对象内
        got_json = False                # 是否已获取完整JSON

        action_cache = ""               # 行动标识缓存
        action_str = "action:"          # 行动标识字符串
        action_idx = 0                  # 行动标识匹配索引

        thought_cache = ""              # 思考标识缓存
        thought_str = "thought:"        # 思考标识字符串
        thought_idx = 0                 # 思考标识匹配索引

        last_character = ""             # 上一个处理的字符

        # 遍历LLM响应流中的每个响应块
        for response in llm_response:
            # 记录使用量信息
            if response.delta.usage:
                usage_dict["usage"] = response.delta.usage
            response_content = response.delta.message.content
            if not isinstance(response_content, str):
                continue

            # 逐字符处理响应内容
            index = 0
            while index < len(response_content):
                steps = 1
                delta = response_content[index : index + steps]
                yield_delta = False

                # 处理代码块分隔符
                if not in_json and delta == "`":
                    last_character = delta
                    code_block_cache += delta
                    code_block_delimiter_count += 1
                else:
                    if not in_code_block:
                        if code_block_delimiter_count > 0:
                            last_character = delta
                            yield code_block_cache
                        code_block_cache = ""
                    else:
                        last_character = delta
                        code_block_cache += delta
                    code_block_delimiter_count = 0

                # 处理不在代码块和JSON中的内容（主要是文本和标识符）
                if not in_code_block and not in_json:
                    # 检查是否匹配"action:"标识（首字符匹配）
                    if delta.lower() == action_str[action_idx] and action_idx == 0:
                        if last_character not in {"\n", " ", ""}:
                            yield_delta = True
                        else:
                            last_character = delta
                            action_cache += delta
                            action_idx += 1
                            if action_idx == len(action_str):
                                action_cache = ""
                                action_idx = 0
                            index += steps
                            continue
                    # 检查是否匹配"action:"标识（后续字符匹配）
                    elif delta.lower() == action_str[action_idx] and action_idx > 0:
                        last_character = delta
                        action_cache += delta
                        action_idx += 1
                        if action_idx == len(action_str):
                            action_cache = ""
                            action_idx = 0
                        index += steps
                        continue
                    else:
                        # 如果之前有缓存的action标识但未完全匹配，则输出缓存内容
                        if action_cache:
                            last_character = delta
                            yield action_cache
                            action_cache = ""
                            action_idx = 0

                    # 检查是否匹配"thought:"标识（首字符匹配）
                    if delta.lower() == thought_str[thought_idx] and thought_idx == 0:
                        if last_character not in {"\n", " ", ""}:
                            yield_delta = True
                        else:
                            last_character = delta
                            thought_cache += delta
                            thought_idx += 1
                            if thought_idx == len(thought_str):
                                thought_cache = ""
                                thought_idx = 0
                            index += steps
                            continue
                    # 检查是否匹配"thought:"标识（后续字符匹配）
                    elif delta.lower() == thought_str[thought_idx] and thought_idx > 0:
                        last_character = delta
                        thought_cache += delta
                        thought_idx += 1
                        if thought_idx == len(thought_str):
                            thought_cache = ""
                            thought_idx = 0
                        index += steps
                        continue
                    else:
                        # 如果之前有缓存的thought标识但未完全匹配，则输出缓存内容
                        if thought_cache:
                            last_character = delta
                            yield thought_cache
                            thought_cache = ""
                            thought_idx = 0

                    # 如果需要输出当前字符
                    if yield_delta:
                        index += steps
                        last_character = delta
                        yield delta
                        continue

                # 处理代码块分隔符计数达到3的情况（即遇到完整的标识）
                if code_block_delimiter_count == 3:
                    if in_code_block:
                        last_character = delta
                        # 从代码块中提取JSON数据
                        action_json_list = extra_json_from_code_block(code_block_cache)
                        if action_json_list:
                            for action_json in action_json_list:
                                yield parse_action(action_json)
                            code_block_cache = ""
                        else:
                            index += steps
                            continue

                    # 切换代码块状态
                    in_code_block = not in_code_block
                    code_block_delimiter_count = 0

                # 处理不在代码块中的情况（主要处理JSON对象）
                if not in_code_block:

                    # 处理单个JSON对象的开始
                    if delta == "{":
                        json_quote_count += 1
                        in_json = True
                        last_character = delta
                        json_cache += delta
                    # 处理单个JSON对象的结束
                    elif delta == "}":
                        last_character = delta
                        json_cache += delta
                        if json_quote_count > 0:
                            json_quote_count -= 1
                            if json_quote_count == 0:
                                in_json = False
                                got_json = True
                                index += steps
                                continue
                    else:
                        # 在JSON对象内部时，缓存字符内容
                        if in_json:
                            last_character = delta
                            json_cache += delta

                    # 处理已获取完整的JSON对象
                    if got_json:
                        got_json = False
                        last_character = delta
                        yield parse_action(json_cache)
                        json_cache = ""
                        json_quote_count = 0
                        in_json = False

                # 处理不在代码块和JSON中的普通字符
                if not in_code_block and not in_json:
                    last_character = delta
                    yield delta.replace("`", "")

                index += steps

        # 处理剩余未处理的代码块缓存
        if code_block_cache:
            yield code_block_cache

        # 处理剩余未处理的JSON缓存
        if json_cache:
            yield parse_action(json_cache)
