
import enum
import json
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator

from core.entities.parameter_entities import CommonParameterType
from core.tools.entities.common_entities import I18nObject


class PluginParameterOption(BaseModel):
    """
    插件参数选项类，用于定义可选参数的值、标签和图标。

    属性:
        value (str): 选项的值。
        label (I18nObject): 显示给用户的国际化标签。
        icon (Optional[str]): 选项的图标，可以是 URL 或 base64 编码的图片。
    """

    value: str = Field(..., description="The value of the option")
    label: I18nObject = Field(..., description="The label of the option")
    icon: Optional[str] = Field(
        default=None, description="The icon of the option, can be a url or a base64 encoded image"
    )

    @field_validator("value", mode="before")
    @classmethod
    def transform_id_to_str(cls, value) -> str:
        """
        将非字符串类型的值转换为字符串。

        参数:
            value: 输入值。

        返回:
            str: 转换后的字符串值。
        """
        if not isinstance(value, str):
            return str(value)
        else:
            return value


class PluginParameterType(enum.StrEnum):
    """
    所有可用的插件参数类型枚举。
    """

    STRING = CommonParameterType.STRING.value
    NUMBER = CommonParameterType.NUMBER.value
    BOOLEAN = CommonParameterType.BOOLEAN.value
    SELECT = CommonParameterType.SELECT.value
    SECRET_INPUT = CommonParameterType.SECRET_INPUT.value
    FILE = CommonParameterType.FILE.value
    FILES = CommonParameterType.FILES.value
    APP_SELECTOR = CommonParameterType.APP_SELECTOR.value
    MODEL_SELECTOR = CommonParameterType.MODEL_SELECTOR.value
    TOOLS_SELECTOR = CommonParameterType.TOOLS_SELECTOR.value
    ANY = CommonParameterType.ANY.value
    DYNAMIC_SELECT = CommonParameterType.DYNAMIC_SELECT.value

    # deprecated, should not use.
    SYSTEM_FILES = CommonParameterType.SYSTEM_FILES.value

    # MCP object and array type parameters
    ARRAY = CommonParameterType.ARRAY.value
    OBJECT = CommonParameterType.OBJECT.value


class MCPServerParameterType(enum.StrEnum):
    """
    MCP 服务器支持的复杂参数类型。
    """

    ARRAY = "array"
    OBJECT = "object"


class PluginParameterAutoGenerate(BaseModel):
    """
    自动生成功能配置类。

    子类:
        Type (enum.StrEnum): 自动生成功能的类型。
    """

    class Type(enum.StrEnum):
        PROMPT_INSTRUCTION = "prompt_instruction"

    type: Type


class PluginParameterTemplate(BaseModel):
    """
    参数模板配置类。

    属性:
        enabled (bool): 是否启用 Jinja 模板功能，默认为 False。
    """

    enabled: bool = Field(default=False, description="Whether the parameter is jinja enabled")


class PluginParameter(BaseModel):
    """
    插件参数定义类，用于描述一个插件参数的各种属性。

    属性:
        name (str): 参数名称。
        label (I18nObject): 显示给用户的标签。
        placeholder (Optional[I18nObject]): 占位符文本。
        scope (str | None): 参数作用域。
        auto_generate (Optional[PluginParameterAutoGenerate]): 自动生成功能配置。
        template (Optional[PluginParameterTemplate]): 模板相关配置。
        required (bool): 是否为必填参数，默认为 False。
        default (Optional[Union[float, int, str]]): 默认值。
        min (Optional[Union[float, int]]): 最小值限制。
        max (Optional[Union[float, int]]): 最大值限制。
        precision (Optional[int]): 数值精度。
        options (list[PluginParameterOption]): 可选值列表。
    """

    name: str = Field(..., description="The name of the parameter")
    label: I18nObject = Field(..., description="The label presented to the user")
    placeholder: Optional[I18nObject] = Field(default=None, description="The placeholder presented to the user")
    scope: str | None = None
    auto_generate: Optional[PluginParameterAutoGenerate] = None
    template: Optional[PluginParameterTemplate] = None
    required: bool = False
    default: Optional[Union[float, int, str]] = None
    min: Optional[Union[float, int]] = None
    max: Optional[Union[float, int]] = None
    precision: Optional[int] = None
    options: list[PluginParameterOption] = Field(default_factory=list)

    @field_validator("options", mode="before")
    @classmethod
    def transform_options(cls, v):
        """
        验证并转换 options 字段为列表格式。

        参数:
            v: 输入值。

        返回:
            list: 转换后的列表。
        """
        if not isinstance(v, list):
            return []
        return v


def as_normal_type(typ: enum.StrEnum):
    """
    将特定类型的参数映射为标准类型字符串。

    参数:
        typ (enum.StrEnum): 参数类型枚举。

    返回:
        str: 标准类型字符串。
    """
    if typ.value in {
        PluginParameterType.SECRET_INPUT,
        PluginParameterType.SELECT,
    }:
        return "string"
    return typ.value


def cast_parameter_value(typ: enum.StrEnum, value: Any, /):
    """
    根据参数类型对值进行类型转换。

    参数:
        typ (enum.StrEnum): 参数类型。
        value (Any): 待转换的原始值。

    返回:
        Any: 转换后的值。

    异常:
        ValueError: 当值无法正确转换时抛出。
    """
    try:
        match typ.value:
            case PluginParameterType.STRING | PluginParameterType.SECRET_INPUT | PluginParameterType.SELECT:
                if value is None:
                    return ""
                else:
                    return value if isinstance(value, str) else str(value)

            case PluginParameterType.BOOLEAN:
                if value is None:
                    return False
                elif isinstance(value, str):
                    # Allowed YAML boolean value strings: https://yaml.org/type/bool.html
                    # and also '0' for False and '1' for True
                    match value.lower():
                        case "true" | "yes" | "y" | "1":
                            return True
                        case "false" | "no" | "n" | "0":
                            return False
                        case _:
                            return bool(value)
                else:
                    return value if isinstance(value, bool) else bool(value)

            case PluginParameterType.NUMBER:
                if isinstance(value, int | float):
                    return value
                elif isinstance(value, str) and value:
                    if "." in value:
                        return float(value)
                    else:
                        return int(value)
            case PluginParameterType.SYSTEM_FILES | PluginParameterType.FILES:
                if not isinstance(value, list):
                    return [value]
                return value
            case PluginParameterType.FILE:
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError("This parameter only accepts one file but got multiple files while invoking.")
                    else:
                        return value[0]
                return value
            case PluginParameterType.MODEL_SELECTOR | PluginParameterType.APP_SELECTOR:
                if not isinstance(value, dict):
                    raise ValueError("The selector must be a dictionary.")
                return value
            case PluginParameterType.TOOLS_SELECTOR:
                if value and not isinstance(value, list):
                    raise ValueError("The tools selector must be a list.")
                return value
            case PluginParameterType.ANY:
                if value and not isinstance(value, str | dict | list | int | float):
                    raise ValueError("The var selector must be a string, dictionary, list or number.")
                return value
            case PluginParameterType.ARRAY:
                if not isinstance(value, list):
                    # Try to parse JSON string for arrays
                    if isinstance(value, str):
                        try:
                            parsed_value = json.loads(value)
                            if isinstance(parsed_value, list):
                                return parsed_value
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return [value]
                return value
            case PluginParameterType.OBJECT:
                if not isinstance(value, dict):
                    # Try to parse JSON string for objects
                    if isinstance(value, str):
                        try:
                            parsed_value = json.loads(value)
                            if isinstance(parsed_value, dict):
                                return parsed_value
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return {}
                return value
            case _:
                return str(value)
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"The tool parameter value {value} is not in correct type of {as_normal_type(typ)}.")


def init_frontend_parameter(rule: PluginParameter, type: enum.StrEnum, value: Any):
    """
    根据规则初始化前端参数值。

    参数:
        rule (PluginParameter): 参数规则定义。
        type (enum.StrEnum): 参数类型。
        value (Any): 实际传入的参数值。

    返回:
        Any: 初始化后的参数值。

    异常:
        ValueError: 当参数值不符合要求时抛出。
    """
    parameter_value = value
    if not parameter_value and parameter_value != 0:
        # get default value
        parameter_value = rule.default
        if not parameter_value and rule.required:
            raise ValueError(f"tool parameter {rule.name} not found in tool config")

    if type == PluginParameterType.SELECT:
        # check if tool_parameter_config in options
        options = [x.value for x in rule.options]
        if parameter_value is not None and parameter_value not in options:
            raise ValueError(f"tool parameter {rule.name} value {parameter_value} not in options {options}")

    return cast_parameter_value(type, parameter_value)
