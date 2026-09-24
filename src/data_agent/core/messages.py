"""统一的消息 / 工具调用结构 —— 整个 Agent 的「通用语」。

为什么要单独一层？
    不同厂商的消息格式差别很大：Anthropic 用 content blocks（text / tool_use /
    tool_result / thinking），OpenAI 用 tool_calls + role="tool"，参数一个是 dict
    一个是 JSON 字符串。如果主循环直接操作厂商格式，换模型就得重写一遍。

    所以这里定义一套中立结构：core / tools 只认它，格式翻译全部关在 llm/ 里面。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型发出的一次工具调用请求。注意：它只是「请求」，执行的是我们自己的代码。"""

    id: str                      # 调用 ID，回传结果时必须原样带上
    name: str                    # 工具名
    arguments: dict[str, Any]    # 已解析成 dict 的参数


@dataclass(frozen=True, slots=True)
class MessageMeta:
    """挂在消息上、**只在本地用、不发给模型**的信息。

    和 Message 的其余字段分开放，是为了让边界一眼可见：
        Message 本身的字段   → provider 会翻译成请求体发出去
        meta 里的字段        → 只给上下文管理、记账、界面用，provider 碰都不碰

    以后要给消息加本地信息（来源、时间、检索片段 ID…），加在这里，
    不要再往 Message 上加字段 —— 那会让「什么会发给模型」越来越说不清。
    """

    # 只有模型真实返回的 assistant 消息才带。它是估算上下文大小的「锚点」：
    # 这条消息之前（含它自己）的 token 数是 API 报的精确值，之后的才需要估。
    # 挂在消息上而不是 Agent 上：一轮失败被回滚时，它跟着消息一起消失。
    usage: Usage | None = None

    # usage 是在哪份视图上量的：这条消息**之前**那段视图的指纹，由 Context.add 盖上。
    # 之后不管哪种编辑策略改了它前面的内容，指纹就对不上，这个 usage 自动作废。
    measured_on: str | None = None

    # 工具结果的一句话摘要（「42 行 × 4 列（region, gmv, …）」），由工具自己给。
    # 结果被清理时，它留在占位文字里当线索（可恢复的压缩）。
    summary: str = ""

    # Agent 自己补的消息：finish_turn 的 nudge、步数耗尽时的兜底回答。
    # 它们不是真人说的话，也不是模型真实的输出。发给模型时和普通消息没区别，
    # 只影响本地怎么认它 —— 比如 nudge 虽然是 role="user"，却不算新回合的开头。
    # （Claude Code 给这类消息标 isMeta，是同一回事。）
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """一条对话消息。

    **不可变**（frozen）。上下文管理要「改」一条消息时，只能生成新对象 ——
    原消息还在历史和快照里，原地修改会让回滚恢复出被改过的内容。
    以前这条规则只写在注释里，现在由语言保证：直接赋值会抛异常。
    需要改的时候用 dataclasses.replace() 或下面的 with_meta()。
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None   # role="tool" 时，对应哪次调用

    # provider 的原生 content（比如 Anthropic 的 content blocks 列表）。
    # 回传历史时优先用它 —— 自己拼 text 回去会丢掉 thinking 块等信息。
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
    def tool_result(tool_call_id: str, content: str, summary: str = "") -> "Message":
        return Message(role="tool", content=content, tool_call_id=tool_call_id,
                       meta=MessageMeta(summary=summary))


# 「话说到一半被 max_tokens 砍断」的 stop_reason。Anthropic 叫 max_tokens，OpenAI 系叫 length。
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})


@dataclass(slots=True)
class LLMResponse:
    """一次模型调用的结果（已归一化）。"""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """被截断了：没有工具调用也不代表说完了。Agent 主循环和写摘要都要查它。"""
        return (self.stop_reason or "").lower() in TRUNCATED_STOP_REASONS

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls,
            raw=self.raw_content,
            # 全 0 说明厂商没报用量（有些兼容接口会这样），不能当锚点用
            meta=MessageMeta(usage=self.usage if self.usage.context_tokens > 0 else None),
        )
