"""统一的消息 / 工具调用结构 —— 整个 Agent 的「通用语」。

为什么要单独一层？
    不同厂商的消息格式差别很大：Anthropic 用 content blocks（text / tool_use /
    tool_result / thinking），OpenAI 用 tool_calls + role="tool"，参数一个是 dict
    一个是 JSON 字符串。如果主循环直接操作厂商格式，换模型就得重写一遍。

    所以这里定义一套中立结构：core / tools 只认它，格式翻译全部关在 llm/ 里面。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """模型发出的一次工具调用请求。注意：它只是「请求」，执行的是我们自己的代码。"""

    id: str                      # 调用 ID，回传结果时必须原样带上
    name: str                    # 工具名
    arguments: dict[str, Any]    # 已解析成 dict 的参数


@dataclass(slots=True)
class Message:
    """一条对话消息。"""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None   # role="tool" 时，对应哪次调用

    # provider 的原生 content（比如 Anthropic 的 content blocks 列表）。
    # 回传历史时优先用它 —— 自己拼 text 回去会丢掉 thinking 块等信息。
    raw: Any = None

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", content=text)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(role="assistant", content=text)

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> "Message":
        return Message(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass(slots=True)
class LLMResponse:
    """一次模型调用的结果（已归一化）。"""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls,
            raw=self.raw_content,
        )
