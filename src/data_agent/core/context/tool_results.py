"""编辑工序：超过阈值时，把较早的工具结果换成一句带线索的占位。最便宜的一种压缩。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..messages import Message
from ..tokens import estimate_message, estimate_text
from .base import ContextEdit, Entry, Marker


@dataclass(frozen=True, slots=True)
class ToolResultsCleared(Marker):
    """标记：这几条工具结果从此以后换成占位。"""

    tool_call_ids: frozenset[str]

    def describe(self) -> str:
        return f"清理了 {len(self.tool_call_ids)} 条较早的工具结果"


class ClearOldToolResults(ContextEdit):
    """参数照抄 Anthropic 服务端的同款功能（context editing 的 clear_tool_uses）：

        trigger_tokens  请求估算超过它才动手
        keep_recent     最近几条工具结果不动（模型多半正在用）
        clear_at_least  一次至少要省下这么多，否则不清 —— 见下面「缓存」
        exclude_tools   这些工具的结果永远不清

    为什么先清工具结果：它是上下文里最大的一块（SQL 结果表格），而且
    **能重新拿到** —— 调用参数（那条 SQL）还留在 assistant 消息里，占位里
    还留着线索。用户的原话、模型的结论清掉就找不回来了，那是摘要的事。

    ── 原件不删，只改视图 ────────────────────────────────────────────
    历史里的工具结果一个字不动。决定清理时往历史追加一个 ToolResultsCleared
    标记，apply() 看到标记才把对应结果换成占位。这个类自己不记任何东西。

    ── 缓存：为什么要攒一批才清 ──────────────────────────────────────
    prompt 缓存是前缀匹配。改了第 k 条消息，第 k 条之后的缓存全部失效，
    下一次请求要重新写缓存（Anthropic 写缓存比正常输入还贵 25%）。
    如果规则是「永远只留最近 3 条」，每来一条新结果，边界就往后挪一格，
    **每次请求都改历史** —— 缓存永远命中不了。
    所以：超过 trigger 才动手，一动手就把能清的全清掉，而且省得不够
    clear_at_least 就干脆不动。清完以后远低于阈值，之后的请求都是纯追加，
    缓存又能命中，直到下一次涨过阈值。
    """

    # 占位的开头。测试、日志靠它认出「这是被清理过的结果」。
    CLEARED_PREFIX = "[这条工具结果已被清理，以节省上下文。"

    def __init__(
        self,
        trigger_tokens: int = 100_000,
        keep_recent: int = 3,
        clear_at_least: int = 10_000,
        exclude_tools: Iterable[str] = (),
    ) -> None:
        self.trigger_tokens = trigger_tokens
        self.keep_recent = keep_recent
        self.clear_at_least = clear_at_least
        self.exclude_tools = frozenset(exclude_tools)

    @classmethod
    def placeholder(cls, m: Message) -> str:
        """被清理的结果换成什么。

        不只是说「清掉了」，还要留下**线索**：原来是几行几列、哪些列
        （工具自己给的 summary）。模型看到「1 行：total=4242」就不必重查；
        看到「42 行 × 4 列（region, gmv, …）」能判断跟当前问题有没有关系。
        工具没给摘要时，至少告诉它原来有多大。

        只依赖消息本身，不掺时间、计数之类会变的东西 —— 占位一变，缓存就废。
        """
        clue = m.meta.summary or f"约 {len(m.content)} 字符"
        return (
            f"{cls.CLEARED_PREFIX}原结果：{clue}。"
            "调用参数还在上面的工具调用里；如果还需要完整数据，重新调用一次即可。]"
        )

    # ------------------------------------------------------------ 视图
    def apply(self, entries: list[Entry]) -> list[Entry]:
        cleared = _cleared_ids(entries)
        return [
            Message.tool_result(e.tool_call_id, self.placeholder(e))
            if isinstance(e, Message) and e.role == "tool" and e.tool_call_id in cleared
            else e
            for e in entries
        ]

    # ------------------------------------------------------------ 清理
    def maintain(self, entries: list[Entry], measure_view: Callable[[], int]) -> Marker | None:
        if measure_view() <= self.trigger_tokens:
            return None

        targets = self._clearable(entries)
        freed = sum(estimate_message(m) - estimate_text(self.placeholder(m)) for m in targets)
        if freed < self.clear_at_least:
            # 省得太少，不值得为此让缓存失效一次
            return None
        return ToolResultsCleared(frozenset(m.tool_call_id for m in targets))

    def _clearable(self, entries: list[Entry]) -> list[Message]:
        """能清的工具结果：还没清过、不在排除名单里、不是最近 keep_recent 条。"""
        messages = [e for e in entries if isinstance(e, Message)]
        cleared = _cleared_ids(entries)
        tool_names = {c.id: c.name for m in messages for c in m.tool_calls}
        results = [m for m in messages if m.role == "tool"]
        older = results[: max(len(results) - self.keep_recent, 0)]
        return [
            m for m in older
            if m.tool_call_id not in cleared
            and tool_names.get(m.tool_call_id) not in self.exclude_tools
        ]

    def status(self, entries: list[Entry]) -> str | None:
        n = len(_cleared_ids(entries))
        return f"已清理的旧工具结果：{n} 条（原件还在，只是不再发给模型）" if n else None


def _cleared_ids(entries: list[Entry]) -> set[str]:
    """历史里所有清理标记记下的 tool_call_id。"""
    ids: set[str] = set()
    for e in entries:
        if isinstance(e, ToolResultsCleared):
            ids |= e.tool_call_ids
    return ids
