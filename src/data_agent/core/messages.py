"""中立的消息 / 工具调用结构。core 和 tools 只认它，各厂商格式的翻译都关在 llm/ 里。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Usage:
    """一次调用的 token 用量，统一成互不重叠的四块（各家「输入」口径不同，provider 负责拆）。

    Anthropic 的 input_tokens 不含缓存部分；OpenAI 系的 prompt_tokens 已含缓存命中。
    """

    input: int = 0          # 没走缓存、按全价计费的输入
    output: int = 0         # 输出（含思考 token）
    cache_read: int = 0     # 命中缓存的输入
    cache_write: int = 0    # 这次写入缓存的输入（只有 Anthropic 单独报）

    @property
    def prompt_tokens(self) -> int:
        """完整输入：系统提示词 + 工具定义 + 全部消息。"""
        return self.input + self.cache_read + self.cache_write

    @property
    def context_tokens(self) -> int:
        """这次回复之后对话占多少上下文（回复会进历史，所以加上 output）。"""
        return self.prompt_tokens + self.output

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型发出的一次工具调用请求（执行的是我们自己的代码）。"""

    id: str                      # 调用 ID，回传结果时必须原样带上
    name: str                    # 工具名
    arguments: dict[str, Any]    # 已解析成 dict 的参数


@dataclass(frozen=True, slots=True)
class Image:
    """消息里的一张图。数据直接存 base64：会话日志要能原样读回，文件之后被覆盖了也不影响。"""

    media_type: str              # image/png、image/jpeg…
    data: str                    # base64，不带 data: 前缀
    width: int = 0               # 估 token 用，0 = 不知道
    height: int = 0

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


@dataclass(frozen=True, slots=True)
class MessageMeta:
    """只在本地用、不发给模型的信息。要给消息加本地信息就加在这里，别加在 Message 上。"""

    # 模型真实返回的 assistant 消息才带：估算上下文大小的锚点。挂在消息上，回滚时跟着消失
    usage: Usage | None = None

    # usage 是在哪份视图上量的（前缀指纹，Context.add 盖上）；前缀被改了 usage 就作废
    measured_on: str | None = None

    # 工具给的一句话摘要，结果被清理时留在占位里当线索
    summary: str = ""

    # Agent 自己补的消息（nudge、步数耗尽的兜底），不是真人说的。Claude Code 叫 isMeta
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """一条对话消息。不可变：上下文管理要「改」只能生成新对象，否则回滚会恢复出改过的内容。"""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None   # role="tool" 时，对应哪次调用
    # role="tool" 时这次调用失败了。上下文清理不清失败的结果
    is_error: bool = False
    # 附带的图片（user 消息、工具结果）。放在 content 后面发，各家怎么放由 provider 决定
    images: tuple[Image, ...] = ()

    # provider 的原生 content（thinking 块、reasoning_content 等），回传时优先用它。
    # 必须是纯 JSON 数据：会话日志要原样存盘、读回
    raw: Any = None

    meta: MessageMeta = field(default_factory=MessageMeta)

    def with_meta(self, **changes: Any) -> "Message":
        """返回一个只改了 meta 某几项的新消息。"""
        return replace(self, meta=replace(self.meta, **changes))

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", content=text)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(role="assistant", content=text)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, *, is_error: bool = False,
                    summary: str = "", images: tuple[Image, ...] = ()) -> "Message":
        return Message(role="tool", content=content, tool_call_id=tool_call_id,
                       is_error=is_error, images=images, meta=MessageMeta(summary=summary))


# stop_reason 的分类，各家叫法不同。没见过的值 Agent 会直接报错，不当成「完成」
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})
REFUSAL_STOP_REASONS = frozenset({"refusal", "content_filter"})
NORMAL_STOP_REASONS = frozenset({"end_turn", "stop", "tool_use", "tool_calls", "function_call", ""})


@dataclass(slots=True)
class LLMResponse:
    """一次模型调用的结果（已归一化）。"""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    @property
    def _reason(self) -> str:
        return (self.stop_reason or "").lower()

    @property
    def truncated(self) -> bool:
        """被截断了：没有工具调用也不代表说完了。Agent 主循环和写摘要都要查它。"""
        return self._reason in TRUNCATED_STOP_REASONS

    @property
    def refused(self) -> bool:
        return self._reason in REFUSAL_STOP_REASONS

    @property
    def finished_normally(self) -> bool:
        return self._reason in NORMAL_STOP_REASONS

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls,
            raw=self.raw_content,
            # 全 0 = 厂商没报用量，不能当锚点
            meta=MessageMeta(usage=self.usage if self.usage.context_tokens > 0 else None),
        )
