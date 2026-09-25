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
    · 摘要是一个标记，原文还在历史里（pi 是日志里的一条 compaction 记录）

── 和 pi 不一样的地方 ──────────────────────────────────────────────
    · 写摘要不把对话序列化成文本，而是原样发、末尾追加要求（学 Claude Code），
      和平时的请求共用前缀、命中缓存。评测数据见 llm_summarizer
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

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from ..errors import CompactionFailed
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
    """把一段对话写成摘要。作为参数注入：context 包不用关心是哪个模型写的，测试里传个假函数就行。

    messages  要压掉的那些消息（可能以上一份摘要开头）—— 正好是上一次请求的**开头一段**
    prompt    上一次请求的系统提示词和工具定义。带上它们、原样发 messages，写摘要的请求
              就和上一次请求共用前缀，能命中缓存（见 llm_summarizer）
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
# 对话原样留在前面（和上一次请求同一个前缀），末尾追加这一条。模型这时还是「数据分析师」、
# 手边还有工具，所以开头先把话说死：这不是新问题，别调工具。
# 小节参考 pi（目标 / 约束偏好 / 进度 / 关键决定 / 下一步 / 关键上下文）和
# Claude Code 的 /compact，按数据分析场景改：表结构和数字是最贵的，丢了就得重查。
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

# 模型自己会思考：直接写
DIRECT_OUTPUT = "- 直接输出摘要，不要开场白。"

# 模型不会自己思考：先打草稿再写（Claude Code 的 <analysis> 做法）。
# 写摘要最怕漏，先按顺序逐条过一遍，第 3 轮随口一句的限定才不会被略过。
# 草稿是明文，排查「摘要漏了什么」时可以看；最终只保留 <summary> 里的内容。
SCRATCHPAD_OUTPUT = """- 先在 <analysis> 标签里按时间顺序逐条过一遍对话：每条用户消息说了什么、定了什么口径、
  查到了什么数字、报过什么错。确认没有遗漏后，再在 <summary> 标签里输出摘要。
  <analysis> 只是草稿，不会被保留。"""


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
) -> Summarize:
    """用 llm 写摘要。

    Args:
        scratchpad: 要不要让模型先在 <analysis> 里打草稿。默认看模型自己会不会
                    思考（llm.native_thinking）：会思考的再打草稿等于想两遍，白花输出 token。
        max_tokens: 这一次的输出上限。思考 token 也算输出，平时的上限可能不够。

    ── 为什么原样发（学 Claude Code） ────────────────────────────────
    系统提示词、工具定义、消息都和上一次请求一模一样，只在末尾追加一条「请写摘要」，
    前面几万 token 全部命中缓存。压缩发生在上下文最大的时候，这一次最贵。

    以前学 pi：对话序列化成一段纯文本，换一个「摘要助手」的系统提示词、不带工具发 ——
    模型只是在读一份记录，不会接着聊、不会调工具。但请求从第一个字就和平时不同，
    一个 token 都命中不了。多轮评测（2026-09-25，DeepSeek 官方）里换过来之后：写摘要命中
    27% → 99%，每段会话未命中少 35%，一份摘要的输出 5.8k → 1.6k token（原文就在上下文里，
    不用把几万字的记录从头理一遍），准确率、回忆、用户定的口径都没掉。

    代价是模型手边有工具，可能去调工具而不写摘要：那就原样再发一次（还是命中缓存，很便宜），
    还不写就算失败，这一轮按事务回滚。

    没传 prompt（调用方没说平时的请求长什么样）时不带系统提示词和工具发：照样能写，
    只是吃不到缓存。Agent 每次都会传。
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
