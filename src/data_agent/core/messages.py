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


@dataclass(frozen=True, slots=True)
class Usage:
    """一次模型调用的 token 用量（已归一化）。

    ⚠️ 各家的「输入 token」口径不一样，这是算错上下文大小最常见的原因：

        Anthropic   input_tokens **不含**缓存部分，完整输入 = input + 读缓存 + 写缓存
        OpenAI 系   prompt_tokens **已含**缓存命中部分

    这里统一成「互不重叠的四块」，各 provider 负责拆分。上层只用这四个字段，
    不用关心厂商叫它什么、包不包含。
    """

    input: int = 0          # 没走缓存、按全价计费的输入
    output: int = 0         # 输出（含思考 token）
    cache_read: int = 0     # 命中缓存的输入
    cache_write: int = 0    # 这次写入缓存的输入（只有 Anthropic 单独报）

    @property
    def prompt_tokens(self) -> int:
        """这次请求的完整输入：系统提示词 + 工具定义 + 全部历史消息。"""
        return self.input + self.cache_read + self.cache_write

    @property
    def context_tokens(self) -> int:
        """这次回复之后，对话一共占多少上下文。

        要加上 output：这次的回复会原样进历史，下一次请求就是输入的一部分。
        """
        return self.prompt_tokens + self.output

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


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

    # 只有模型真实返回的 assistant 消息才带。它是估算上下文大小的「锚点」：
    # 这条消息之前（含它自己）的 token 数是 API 报的精确值，之后的才需要估。
    #
    # 为什么挂在消息上而不是 Agent 上：一轮失败被回滚时，这条消息连同它的
    # usage 一起消失，不会留下一个已经不对的数。
    #
    # ⚠️ 锚点成立的前提是「它之前的消息没被改过」。以后做压缩时，被改写位置
    #    之后的 assistant 消息必须换成 usage=None 的新对象，否则锚点就是错的。
    usage: "Usage | None" = None

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
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls,
            raw=self.raw_content,
            # 全 0 说明厂商没报用量（有些兼容接口会这样），不能当锚点用
            usage=self.usage if self.usage.context_tokens > 0 else None,
        )
