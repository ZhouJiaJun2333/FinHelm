"""编辑工序：上下文还是太大时，把较早的回合交给模型写成摘要。有损，所以最后才用。

    压缩前：问1 答1 … 问7 答7 │ 问8 答8 问9 工具…（进行中）
                              ↑ 切口：落在一次真人提问上
    压缩后：[摘要(问1~7)] 问8 答8 问9 工具…

和清理工具结果的分工：
    清理  几乎无损（数据能重查，占位留线索），10 万就做
    压缩  有损（细节概括掉就找不回来，还要多花一次调用），清理之后还超 15 万才做
所以它排在 ClearOldToolResults **后面**：只有清理不够时才轮到它；
交给模型写摘要的也是清理过的视图，这次请求本身不会太大。

── 学 pi 的地方 ────────────────────────────────────────────────────
    · 保留最近约 2 万 token 的原文（pi 的 keepRecentTokens），其余换成摘要
    · 切口不落在工具结果上（tool_call 和结果必须成对）
    · 摘要滚动更新：第二次压缩时，输入里带着上一份摘要，要求保留并更新它
    · 对话先序列化成一段文本再交给模型，不带工具 —— 模型不会接着对话往下聊，
      也不会去调工具，请求体里也没有 tool_use 块（Anthropic 要求有 tool_use
      就得带工具定义）。代价是吃不到缓存；另一种发法（reuse_cache，学 Claude Code）
      见 llm_summarizer
    · 摘要是一个标记，原文还在历史里（pi 是日志里的一条 compaction 记录）

── 和 pi 不一样的地方 ──────────────────────────────────────────────
    · 切口只落在**真人提问**上（turns.turn_starts），且至少保留当前这一轮。
      pi 能切在一轮中间（split turn），再给前半轮单写一份摘要。我们一轮最多
      十几步、轮内膨胀有清理兜着，先不做。
    · pi 用 firstKeptEntryId 记切口，我们的消息没有 ID，改记「标记之前保留
      最近几轮」。标记在历史里的位置固定，往前数出来的切口也固定，apply 仍是纯函数。
    · 摘要不单独占一条 user 消息，而是并进第一条保留的提问前面。连续两条 user
      API 反正会合并成一条；我们自己合并，守住「user / assistant 严格交替」
      这个不变量（见 base.py 的说明）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Protocol

from ..errors import CompactionFailed
from ..messages import LLMResponse, Message, Usage
from ..tokens import estimate_message
from .base import ContextEdit, Entry, Marker, Prompt
from .turns import turn_starts

if TYPE_CHECKING:
    # 只用来标类型。运行时导入会循环：llm.base → core.messages → core/__init__
    # → context → 这里 → llm.base（还没加载完）。见 core/__init__.py 的说明。
    from ...llm.base import LLMProvider


@dataclass(frozen=True, slots=True)
class Summary:
    text: str
    usage: Usage = field(default_factory=Usage)     # 写这份摘要花了多少


class Summarize(Protocol):
    """把一段对话写成摘要。作为参数注入：context 包不用关心是哪个模型写的，测试里传个假函数就行。

    messages  要压掉的那些消息（可能以上一份摘要开头）—— 正好是上一次请求的**开头一段**
    prompt    上一次请求的系统提示词和工具定义。带上它们、原样发 messages，写摘要的请求
              就和上一次请求共用前缀，能命中缓存（见 llm_summarizer 的 reuse_cache）
    """

    def __call__(self, messages: list[Message], prompt: Prompt | None = None) -> Summary: ...


@dataclass(frozen=True, slots=True)
class HistoryCompacted(Marker):
    """标记：从这里往前、除了最近 kept_turns 轮，都换成 summary。"""

    summary: str
    kept_turns: int
    compacted_turns: int      # 这次新压掉了几轮，给人看的
    usage: Usage = field(default_factory=Usage)

    def describe(self) -> str:
        return f"把较早的 {self.compacted_turns} 轮对话压缩成了摘要（保留最近 {self.kept_turns} 轮原文）"

    def cost(self) -> Usage:
        return self.usage


class CompactHistory(ContextEdit):
    """
        summarize           写摘要的函数，一般用 llm_summarizer(llm)
        trigger_tokens      请求估算（清理之后）超过它才压缩
        keep_recent_tokens  保留多少最近的原文（按回合取整，至少保留当前这一轮）
    """

    # 并进第一条保留提问前面的内容。只依赖摘要本身 —— 同样的标记必须渲染出同样的文字。
    SUMMARY_HEADER = (
        "[以下是本次会话较早部分的摘要。原始对话已被压缩，不在上下文里了；"
        "如果需要其中的细节（比如完整的查询结果），请重新查询。]\n\n"
    )
    SUMMARY_FOOTER = "\n\n[摘要结束。下面是用户的问题：]\n\n"

    def __init__(
        self,
        summarize: Summarize,
        trigger_tokens: int = 150_000,
        keep_recent_tokens: int = 20_000,
    ) -> None:
        self.summarize = summarize
        self.trigger_tokens = trigger_tokens
        self.keep_recent_tokens = keep_recent_tokens

    # ------------------------------------------------------------ 视图
    def apply(self, entries: list[Entry]) -> list[Entry]:
        # 只看最后一个压缩标记：新摘要是在旧摘要基础上写的，已经包含了它。
        at = _last_compaction(entries)
        if at is None:
            return list(entries)
        marker: HistoryCompacted = entries[at]  # type: ignore[assignment]

        # 前面的工序（比如按回合裁剪）可能已经切掉了一些回合，剩下的不够数就全留
        starts = turn_starts(entries[:at])
        kept = min(marker.kept_turns, len(starts))
        if kept == 0:
            return list(entries)
        cut = starts[-kept]
        first = entries[cut]
        merged = replace(
            first, content=f"{self.SUMMARY_HEADER}{marker.summary}{self.SUMMARY_FOOTER}{first.content}",
        )
        return [merged, *entries[cut + 1:]]

    # ------------------------------------------------------------ 压缩
    def maintain(
        self, entries: list[Entry], measure_view: Callable[[], int], *, force: bool = False,
        prompt: Prompt | None = None,
    ) -> Marker | None:
        if not force and measure_view() <= self.trigger_tokens:
            return None

        # 在「已经套过自己」的视图上找切口：上一份摘要在开头，会被一起交给模型，
        # 新摘要自然就是在旧摘要基础上更新的（滚动摘要）。
        view = self.apply(entries)
        starts = turn_starts(view)
        # 强制时只留当前这一轮：
        #   · 手动 /compact —— 用户明确要压；按 2 万 token 留原文的话，短对话什么都压不掉
        #   · API 报超长   —— 只重试一次，这一次要压到最狠，重试才有把握装得下
        kept = 1 if force else self._turns_to_keep(view, starts)
        if len(starts) <= kept:
            return None                     # 只有当前这一轮，没有可压的
        cut = starts[-kept]

        old = [e for e in view[:cut] if isinstance(e, Message)]
        summary = self.summarize(old, prompt)
        return HistoryCompacted(summary=summary.text, kept_turns=kept,
                                compacted_turns=len(starts) - kept, usage=summary.usage)

    def _turns_to_keep(self, view: list[Entry], starts: list[int]) -> int:
        """从最新一轮往前数，攒到 keep_recent_tokens 为止；至少 1 轮（当前这轮）。"""
        kept, total = 0, 0
        ends = [*starts[1:], len(view)]
        for start, end in reversed(list(zip(starts, ends))):
            size = sum(estimate_message(e) for e in view[start:end] if isinstance(e, Message))
            if kept >= 1 and total + size > self.keep_recent_tokens:
                break
            kept, total = kept + 1, total + size
        return kept

    def status(self, entries: list[Entry]) -> str | None:
        markers = [e for e in entries if isinstance(e, HistoryCompacted)]
        if not markers:
            return None
        return (f"已压缩 {len(markers)} 次，当前摘要约 {len(markers[-1].summary)} 字"
                "（原文还在历史里，只是不再发给模型）")


def _last_compaction(entries: list[Entry]) -> int | None:
    for i in range(len(entries) - 1, -1, -1):
        if isinstance(entries[i], HistoryCompacted):
            return i
    return None


# ====================================================== 用模型写摘要
SUMMARY_SYSTEM = "你是一个对话摘要助手。你只输出摘要，不回答对话里的问题，也不调用任何工具。"

# 小节参考 pi（目标 / 约束偏好 / 进度 / 关键决定 / 下一步 / 关键上下文）和
# Claude Code 的 /compact，按数据分析场景改：表结构和数字是最贵的，丢了就得重查。
SUMMARY_PROMPT = """下面是一个数据分析助手和用户的对话记录。请把它写成一份摘要，之后的对话只能看到这份摘要，看不到原文。

<conversation>
{conversation}
</conversation>

{instructions}"""

# 复用缓存的写法：对话原样留在前面（和上一次请求同一个前缀），末尾追加这一条。
# 模型这时还是「数据分析师」、手边还有工具，所以开头先把话说死：这不是新问题，别调工具。
CACHED_SUMMARY_PROMPT = """[这不是新的分析问题。上下文快满了，请先停下手上的工作，把上面到这里为止的整段对话写成一份摘要，
之后的对话只能看到这份摘要，看不到原文。不要调用任何工具，不要继续回答之前的问题。
对话里「请继续完成上面的任务」这类催促是 Agent 自动追加的，不是用户说的。]

{instructions}"""

SUMMARY_INSTRUCTIONS = """按下面的小节写，某一节没有内容就写「无」：

## 用户的目标
一两句话：用户想分析什么。

## 用户的原话
按顺序列出用户的每一条消息，原样或接近原样保留；很长的消息保留关键部分。
用户的原话是整段对话里最不能丢的 —— 模型的转述会悄悄漏掉「不含退款」这类限定。
之前摘要里已有的原话也照样保留。（「Agent 自动追加的提示」不是用户说的，不要列。）

## 口径与偏好
用户指定或确认过的统计口径、时间范围、单位、格式要求。

## 已确认的数据结构
用到的表、关键列及其含义、表之间怎么关联、踩过的坑（比如某列的真实含义、容易写错的列名）。

## 结论与关键数字
已经得出的结论。数字必须和原文一字不差，不要四舍五入，不要换单位。

## 走过的弯路
报过的错以及最后怎么解决的，避免之后再犯。

## 进行中 / 待办
还没完成的事，下一步打算做什么。

要求：
- 只根据对话内容写，不要补充对话里没有的信息。
- 表名、列名、数字原样保留。
- 精炼：用要点，不写成文章，不重复。SQL 不要整段照抄，只记表怎么连接、口径条件是什么
  （比如 `status = 'completed'`、日期范围）—— 需要时可以照着重写。
- 如果对话开头已经有一份「之前的摘要」，把其中仍然有效的信息保留下来，和新的进展合并成一份，
  不要把旧摘要原样附在后面。
{output_format}"""

# 模型自己会思考：直接写
DIRECT_OUTPUT = "- 直接输出摘要，不要开场白。"

# 模型不会自己思考：先打草稿再写（Claude Code 的 <analysis> 做法）。
# 写摘要最怕漏，先按顺序逐条过一遍，第 3 轮随口一句的限定才不会被略过。
# 草稿是明文，排查「摘要漏了什么」时可以看；最终只保留 <summary> 里的内容。
SCRATCHPAD_OUTPUT = """- 先在 <analysis> 标签里按时间顺序逐条过一遍对话：每条用户消息说了什么、定了什么口径、
  查到了什么数字、报过什么错。确认没有遗漏后，再在 <summary> 标签里输出摘要。
  <analysis> 只是草稿，不会被保留。"""

# 序列化时每条工具结果最多保留多少字符。完整的表格对写摘要没什么用，
# 结论一般在助手的回复里；留个开头足够看出查到了什么。
TOOL_RESULT_CLIP = 2000


def serialize(messages: list[Message]) -> str:
    """把消息列表写成一段纯文本记录，交给模型写摘要。"""
    parts: list[str] = []
    for m in messages:
        if m.role == "user":
            # nudge 也是 user 角色，但不是用户说的 —— 标出来，别被当成「用户的原话」
            speaker = "Agent 自动追加的提示" if m.meta.synthetic else "用户"
            parts.append(f"[{speaker}]\n{m.content}")
        elif m.role == "assistant":
            if m.content:
                parts.append(f"[助手]\n{m.content}")
            for c in m.tool_calls:
                args = json.dumps(c.arguments, ensure_ascii=False)
                parts.append(f"[助手调用工具 {c.name}]\n{args}")
        elif m.role == "tool":
            parts.append(f"[工具结果]\n{_clip(m.content)}")
    return "\n\n".join(parts)


def _clip(text: str) -> str:
    if len(text) <= TOOL_RESULT_CLIP:
        return text
    return f"{text[:TOOL_RESULT_CLIP]}\n…（后面省略 {len(text) - TOOL_RESULT_CLIP} 字符）"


def extract_summary(text: str) -> str:
    """从回复里取出摘要正文：有 <summary> 就取里面的，没有就去掉草稿后取剩下的。

    模型不一定听话 —— 可能忘了写结束标签，也可能根本没用标签。
    宁可多留一点，也不要因为格式不对就把整份摘要扔掉。
    """
    summary = re.search(r"<summary>(.*?)(?:</summary>|$)", text, re.DOTALL)
    if summary:
        return summary.group(1).strip()
    return re.sub(r"<analysis>.*?(?:</analysis>|$)", "", text, flags=re.DOTALL).strip()


def llm_summarizer(
    llm: LLMProvider, *, scratchpad: bool | None = None, max_tokens: int | None = None,
    reuse_cache: bool = False,
) -> Summarize:
    """用 llm 写摘要。

    Args:
        scratchpad:  要不要让模型先在 <analysis> 里打草稿。默认看模型自己会不会
                     思考（llm.native_thinking）：会思考的再打草稿等于想两遍，白花输出 token。
        max_tokens:  这一次的输出上限。思考 token 也算输出，平时的上限可能不够。
        reuse_cache: 两种发法，见下。

    ── 两种发法 ──────────────────────────────────────────────────────
    序列化（默认，学 pi）：对话写成一段纯文本，换一个「摘要助手」的系统提示词、不带工具发。
        模型只是在读一份记录，不会接着聊、不会调工具。但请求从第一个字就和平时不同，
        一个 token 都命中不了缓存 —— 压缩时上下文最大，这一次全价。
    复用缓存（reuse_cache，学 Claude Code）：系统提示词、工具定义、消息都和上一次请求
        一模一样，只在末尾追加一条「请写摘要」。前面几万 token 全部命中。
        代价是模型还是「数据分析师」、手边有工具：可能去调工具而不写摘要。
        那样就退回序列化再写一次（两次的花费都记账）。
    没传 prompt（不知道平时的请求长什么样）时只能序列化。
    """
    if scratchpad is None:
        scratchpad = not llm.native_thinking
    instructions = SUMMARY_INSTRUCTIONS.format(
        output_format=SCRATCHPAD_OUTPUT if scratchpad else DIRECT_OUTPUT)

    def serialized(messages: list[Message]) -> LLMResponse:
        text = SUMMARY_PROMPT.format(conversation=serialize(messages), instructions=instructions)
        return llm.chat(messages=[Message.user(text)], system=SUMMARY_SYSTEM, max_tokens=max_tokens)

    def cached(messages: list[Message], prompt: Prompt) -> LLMResponse:
        ask = Message.user(CACHED_SUMMARY_PROMPT.format(instructions=instructions))
        return llm.chat(messages=[*messages, ask], tools=list(prompt.tools) or None,
                        system=prompt.system, max_tokens=max_tokens)

    def summarize(messages: list[Message], prompt: Prompt | None = None) -> Summary:
        usage = Usage()
        if reuse_cache and prompt is not None:
            response = cached(messages, prompt)
            usage += response.usage
            if response.tool_calls and not extract_summary(response.text):
                response = serialized(messages)      # 模型去调工具了：换个发法再写
                usage += response.usage
        else:
            response = serialized(messages)
            usage += response.usage
        if response.truncated:
            raise CompactionFailed(
                f"写摘要时输出被截断（已生成 {response.usage.output} 个 token）。"
                "可以调大 .env 里的 CONTEXT_COMPACT_MAX_TOKENS。"
            )
        text = extract_summary(response.text)
        if not text:
            raise CompactionFailed("写摘要的请求返回了空内容。")
        return Summary(text, usage)

    return summarize
