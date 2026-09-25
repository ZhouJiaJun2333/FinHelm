"""工序：清理之后上下文还太大，把较早的回合交给模型写成摘要。有损，所以排在清理后面。

    压缩前：问1 答1 … 问7 答7 │ 问8 答8 问9 工具…（进行中）
                              ↑ 切口：落在一次真人提问上，至少保留当前这一轮
    压缩后：[摘要(问1~7)] 问8 答8 问9 工具…

学 pi：保留最近约 2 万 token 原文；滚动摘要（上一份摘要一起交给模型更新）；摘要是一个标记。
和 pi 不同：写摘要原样发对话、共用缓存（学 Claude Code）；不在一轮中间切；
摘要并进第一条保留的提问前面，守住 user / assistant 交替。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from ..errors import CompactionFailed, ContextOverflow
from ..messages import Message, Usage
from ..provider import LLMProvider
from ..tokens import estimate_message
from .base import ContextEdit, Entry, Marker, Prompt
from .turns import turn_starts


@dataclass(frozen=True, slots=True)
class Summary:
    text: str
    usage: Usage = field(default_factory=Usage)     # 写这份摘要花了多少


class Summarize(Protocol):
    """把一段对话写成摘要。注入进来：测试里传个假函数就行。

    messages 是要压掉的消息（可能以上一份摘要开头），正好是上一次请求的开头一段；
    带上 prompt 原样发，就和上一次请求共用缓存前缀。
    """

    def __call__(self, messages: list[Message], prompt: Prompt | None = None) -> Summary: ...


@dataclass(frozen=True, slots=True)
class HistoryCompacted(Marker):
    """标记：从这里往前、除了最近 kept_turns 轮，都换成 summary。"""

    summary: str
    kept_turns: int
    compacted_turns: int      # 这次新压掉了几轮，给人看的
    usage: Usage = field(default_factory=Usage)
    dropped_turns: int = 0    # 其中最早的几轮太长、写摘要时放不下，没写进摘要

    def describe(self) -> str:
        text = f"把较早的 {self.compacted_turns} 轮对话压缩成了摘要（保留最近 {self.kept_turns} 轮原文）"
        if self.dropped_turns:
            text += f"，其中最早的 {self.dropped_turns} 轮太长放不下，没写进摘要"
        return text

    def cost(self) -> Usage:
        return self.usage


class CompactHistory(ContextEdit):
    """trigger_tokens：清理之后还超过它才压缩；keep_recent_tokens：保留多少最近的原文（按回合取整）。"""

    # 只依赖摘要本身：同样的标记必须渲染出同样的文字
    SUMMARY_HEADER = (
        "[以下是本次会话较早部分的摘要。原始对话已被压缩，不在上下文里了；"
        "如果需要其中的细节（比如完整的查询结果），请重新查询。]\n\n"
    )
    SUMMARY_FOOTER = "\n\n[摘要结束。下面是用户的问题：]\n\n"
    OMITTED_NOTE = "[更早的对话太长，写摘要时放不下，已省略。]"
    # 写摘要的请求自己超长时，最多丢几次最老的回合
    OVERFLOW_RETRIES = 3

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
        # 只看最后一个压缩标记：新摘要已经包含了旧摘要
        at = _last_compaction(entries)
        if at is None:
            return list(entries)
        marker: HistoryCompacted = entries[at]  # type: ignore[assignment]

        # 前面的工序可能已经切掉了一些回合，不够数就全留
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

        # 在套过自己的视图上找切口：上一份摘要在开头，一起交给模型更新（滚动摘要）
        view = self.apply(entries)
        starts = turn_starts(view)
        # 强制时只留当前这一轮：/compact 是用户明确要压；API 报超长只重试一次，要压到最狠
        kept = 1 if force else self._turns_to_keep(view, starts)
        if len(starts) <= kept:
            return None                     # 只有当前这一轮，没有可压的
        cut = starts[-kept]

        old = [e for e in view[:cut] if isinstance(e, Message)]
        summary, dropped = self._summarize_fitting(old, prompt)
        return HistoryCompacted(summary=summary.text, kept_turns=kept, compacted_turns=len(starts) - kept,
                                usage=summary.usage, dropped_turns=dropped)

    def _summarize_fitting(self, old: list[Message], prompt: Prompt | None) -> tuple[Summary, int]:
        """写摘要的请求自己也超长时，丢掉最老的一半回合再试（学 Claude Code），上一份摘要留着。

        强制压缩只留当前一轮，当前一轮又很小时，写摘要的请求和刚报超长的那次几乎一样大。
        """
        dropped = 0
        for attempt in range(self.OVERFLOW_RETRIES + 1):
            try:
                return self.summarize(old, prompt), dropped
            except ContextOverflow as exc:
                starts = turn_starts(old)
                if attempt == self.OVERFLOW_RETRIES or len(starts) < 2:
                    raise CompactionFailed(
                        f"写摘要的请求本身超出了上下文窗口，丢掉最早的 {dropped} 轮之后还是放不下。"
                    ) from exc
                n = max(1, (len(starts) - 1) // 2)
                old = self._drop_oldest(old, starts[n])
                dropped += n
        raise AssertionError("unreachable")

    def _drop_oldest(self, old: list[Message], cut: int) -> list[Message]:
        """丢掉 old[:cut]。开头如果带着上一份摘要，把它并进新的开头。"""
        first, rest = old[cut], old[cut + 1:]
        head = old[0].content
        if head.startswith(self.SUMMARY_HEADER) and self.SUMMARY_FOOTER in head:
            previous = head[len(self.SUMMARY_HEADER):].split(self.SUMMARY_FOOTER, 1)[0]
            previous = previous.removesuffix(f"\n\n{self.OMITTED_NOTE}")      # 第二次丢时别叠两句
            prefix = f"{self.SUMMARY_HEADER}{previous}\n\n{self.OMITTED_NOTE}{self.SUMMARY_FOOTER}"
        else:
            prefix = f"{self.OMITTED_NOTE}\n\n"
        return [replace(first, content=prefix + first.content), *rest]

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
# 追加在原样的对话后面。模型手边还有工具，所以开头先说死：这不是新问题，别调工具。
# 小节参考 pi 和 Claude Code，按数据分析改：用户原话、口径、表结构、数字最不能丢。
SUMMARY_PROMPT = """[这不是新的分析问题。上下文快满了，请先停下手上的工作，把上面到这里为止的整段对话写成一份摘要，
之后的对话只能看到这份摘要，看不到原文。不要调用任何工具，不要继续回答之前的问题。]

按下面的小节写，某一节没有内容就写「无」：

## 用户的目标
一两句话：用户想分析什么。

## 用户的原话
按顺序列出用户的每一条消息，原样或接近原样保留；很长的消息保留关键部分。
用户的原话是整段对话里最不能丢的 —— 模型的转述会悄悄漏掉「不含退款」这类限定。
之前摘要里已有的原话也照样保留。「请继续完成上面的任务」这类催促是 Agent 自动追加的，
不是用户说的，不要列。

## 口径与偏好
用户指定或确认过的统计口径、时间范围、单位、格式要求。

## 已确认的数据结构
用到的表、关键列及其含义、表之间怎么关联、踩过的坑（比如某列的真实含义、容易写错的列名）。

## 结论与关键数字
已经得出的结论。数字必须和原文一字不差，不要四舍五入，不要换单位。
给用户展示过的结果编号（r3 这种）连同它是什么一起记下 —— 之后回答里还能用 {{{{r3}}}} 引用。

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

DIRECT_OUTPUT = "- 直接输出摘要，不要开场白。"

# 模型不会自己思考时先打草稿（Claude Code 的 <analysis>），逐条过一遍才不会漏掉随口一句的限定
SCRATCHPAD_OUTPUT = """- 先在 <analysis> 标签里按时间顺序逐条过一遍对话：每条用户消息说了什么、定了什么口径、
  查到了什么数字、报过什么错。确认没有遗漏后，再在 <summary> 标签里输出摘要。
  <analysis> 只是草稿，不会被保留。"""


def extract_summary(text: str) -> str:
    """有 <summary> 取里面的，没有就去掉草稿取剩下的。标签没写全也不扔，宁可多留。"""
    summary = re.search(r"<summary>(.*?)(?:</summary>|$)", text, re.DOTALL)
    if summary:
        return summary.group(1).strip()
    return re.sub(r"<analysis>.*?(?:</analysis>|$)", "", text, flags=re.DOTALL).strip()


def llm_summarizer(
    llm: LLMProvider, *, scratchpad: bool | None = None, max_tokens: int | None = None,
) -> Summarize:
    """用 llm 写摘要：和上一次请求一模一样的系统提示词、工具、消息，末尾追加一条要求（学 Claude Code）。

    前缀全部命中缓存。以前学 pi 序列化成文本发，一个 token 都命中不了；换过来后多轮评测里
    写摘要命中 27% → 99%，摘要输出 5.8k → 1.6k token，准确率不变（2026-09-25）。
    代价是模型可能去调工具：原样再发一次，还不写就算失败。

    scratchpad：先在 <analysis> 里打草稿。默认只给不会自己思考的模型（会思考的再打草稿是想两遍）。
    max_tokens：写摘要这一次的输出上限，思考 token 也算在内。
    """
    if scratchpad is None:
        scratchpad = not llm.native_thinking
    ask = Message.user(SUMMARY_PROMPT.format(
        output_format=SCRATCHPAD_OUTPUT if scratchpad else DIRECT_OUTPUT))

    def summarize(messages: list[Message], prompt: Prompt | None = None) -> Summary:
        prompt = prompt or Prompt()
        usage = Usage()
        for _ in range(2):
            response = llm.chat(messages=[*messages, ask], tools=list(prompt.tools) or None,
                                system=prompt.system, max_tokens=max_tokens)
            usage += response.usage
            if not (response.tool_calls and not extract_summary(response.text)):
                break
        else:
            raise CompactionFailed("写摘要时模型两次都去调工具了，没写摘要。")
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
