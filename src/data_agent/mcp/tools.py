"""把 MCP 服务器的工具包装成我们的 Tool：名字加前缀、描述截断、参数按它的 JSON Schema 校验、结果转成 ToolOutput。

外部服务器的描述和结果都是不可信输入：描述会进模型的工具表（等于别人往我们的提示词里写字），
所以标明来源、限制长度；调用前按 schema 校验参数，多传的参数也拒掉（拼错的可选参数会被服务器悄悄忽略）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from ..core.messages import INVALID_JSON_ARGS, Image
from ..core.tools import Tool, ToolOutput
from .client import McpClient, McpError

PREFIX = "mcp__"
MAX_DESCRIPTION = 1500
MAX_NAME = 64                 # OpenAI / Anthropic 对工具名的长度上限


def tool_name(server: str, tool: str) -> str:
    """mcp__服务器__工具（学 Claude Code），只留模型 API 认的字符。"""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{PREFIX}{server}__{tool}")[:MAX_NAME]


class McpTool(Tool):
    Args = BaseModel             # 不用：参数按服务器给的 JSON Schema 校验

    def __init__(self, client: McpClient, spec: dict[str, Any]) -> None:
        self.client = client
        self.server = client.config.name
        self.remote_name = spec["name"]
        self.name = tool_name(self.server, self.remote_name)
        desc = (spec.get("description") or "").strip()
        if len(desc) > MAX_DESCRIPTION:
            desc = desc[:MAX_DESCRIPTION] + "…"
        self.description = f"[外部 MCP 服务器 {self.server} 的工具] {desc}"
        self.input_schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
        # 服务器自称只读才让上下文清理掉旧结果（清掉了要重调）；是不是真只读我们没法验证，审批照样要过
        self.rerunnable = bool((spec.get("annotations") or {}).get("readOnlyHint"))

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.input_schema}

    def run(self, args: Any) -> ToolOutput:          # execute() 整个换掉了，这里用不到
        raise NotImplementedError

    def execute(self, raw_args: dict[str, Any]) -> ToolOutput:
        if INVALID_JSON_ARGS in raw_args:
            return ToolOutput.error(f"参数不是合法的 JSON，请重新调用：{raw_args[INVALID_JSON_ARGS]}")
        if errors := validate(self.input_schema, raw_args):
            return ToolOutput.error("参数不合法：" + "；".join(errors[:5]))
        try:
            result = self.client.call_tool(self.remote_name, raw_args)
        except McpError as exc:
            return ToolOutput.error(str(exc)).capped(self.max_output_chars)
        return to_output(result, f"调用了 {self.server} 的 {self.remote_name}").capped(self.max_output_chars)


def to_output(result: dict[str, Any], summary: str) -> ToolOutput:
    """MCP 的结果是一串内容块：文字拼起来，图片发给模型，资源有文字就给文字。"""
    parts: list[str] = []
    images: list[Image] = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "image" and block.get("data"):
            images.append(Image(block.get("mimeType", "image/png"), block["data"]))
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(res.get("text") or f"[资源 {res.get('uri', '')}，不是文字，没有显示]")
        elif kind == "resource_link":
            parts.append(f"[资源链接] {block.get('name', '')} {block.get('uri', '')}".strip())
        else:
            parts.append(f"[{kind} 内容，没有显示]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts) or "（工具没有返回内容）"
    return ToolOutput(text, summary=summary, is_error=bool(result.get("isError")), images=tuple(images))


# ---------------------------------------------------------------- JSON Schema 校验（常用的那部分）
_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def validate(schema: dict[str, Any], value: Any, path: str = "参数") -> list[str]:
    """支持 type、enum、const、required、properties、additionalProperties、items、anyOf / oneOf、
    数值和长度的上下限；$ref 这类不认识的关键字跳过（不拦，交给服务器）。

    顶层对象没写 additionalProperties 时也拒绝多传的参数：拼错的可选参数（limt）服务器多半悄悄忽略，调用照样「成功」。"""
    if not isinstance(schema, dict):
        return []
    for key in ("anyOf", "oneOf"):
        if key in schema and not any(not validate(s, value, path) for s in schema[key]):
            return [f"{path} 不符合任何一种允许的格式"]
    types = schema.get("type")
    types = [types] if isinstance(types, str) else types or []
    if types and not any(_TYPES.get(t, lambda v: True)(value) for t in types):
        return [f"{path} 应该是 {'/'.join(types)}，传的是 {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} 只能是 {schema['enum']} 之一，传的是 {value!r}"]
    if "const" in schema and value != schema["const"]:
        return [f"{path} 只能是 {schema['const']!r}"]
    errors = _limits(schema, value, path)
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        errors += [f"缺少必填的 {path}.{k}" if path != "参数" else f"缺少必填参数 {k}"
                   for k in schema.get("required") or [] if k not in value]
        extra = schema.get("additionalProperties", False if path == "参数" and props else True)
        for k, v in value.items():
            where = k if path == "参数" else f"{path}.{k}"
            if k in props:
                errors += validate(props[k], v, where)
            elif extra is False:
                errors.append(f"没有参数 {where}，可用的：{'、'.join(props) or '（没有参数）'}")
            elif isinstance(extra, dict):
                errors += validate(extra, v, where)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errors += validate(schema["items"], v, f"{path}[{i}]")
    return errors


def _limits(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    out = []
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path} 不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path} 不能大于 {schema['maximum']}")
    if isinstance(value, (str, list)):
        low, high = ("minLength", "maxLength") if isinstance(value, str) else ("minItems", "maxItems")
        if low in schema and len(value) < schema[low]:
            out.append(f"{path} 太短（至少 {schema[low]}）")
        if high in schema and len(value) > schema[high]:
            out.append(f"{path} 太长（最多 {schema[high]}）")
    return out
