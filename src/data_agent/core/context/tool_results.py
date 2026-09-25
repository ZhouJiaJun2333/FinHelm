"""工序：超过阈值时，把较早的工具结果换成一句带线索的占位。最便宜的一种压缩。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..messages import Message
from ..tokens import estimate_message, estimate_text
from .base import ContextEdit, Entry, Marker, Prompt


@dataclass(frozen=True, slots=True)
class ToolResultsCleared(Marker):
    """这几条工具结果从此换成占位。"""

    tool_call_ids: frozenset[str]

    def describe(self) -> str:
        return f"清理了 {len(self.tool_call_ids)} 条较早的工具结果"


class ClearOldToolResults(ContextEdit):
    """参数照抄 Anthropic 服务端的 clear_tool_uses。

        trigger_tokens  请求估算超过它才动手
        keep_recent     最近几条不动（模型多半正在用）
        clear_at_least  一次省不到这么多就不清
        tools           只清这些工具的结果（app.py 传 rerunnable 的工具）；None = 不限

    只清能重拿、而且成功的结果：占位叫模型「重新调用一次」，只对只读工具成立；
    失败的结果记着「这条路走不通」，清掉等于鼓励它再撞一次墙。

    攒一批才清：缓存是前缀匹配，改一条消息后面全部失效。「永远只留最近 3 条」会让每次请求都改历史；
    超过阈值才动手、一次清完，之后又是纯追加。（评测试过「小结果不清」「降不到低水位就不清」，
    未命中打平；不清理总输入多 17%，更贵。2026-09-25）
    """

    # 测试、日志靠它认出被清理过的结果
    CLEARED_PREFIX = "[这条工具结果已被清理，以节省上下文。"

    def __init__(
        self,
        trigger_tokens: int = 100_000,
        keep_recent: int = 3,
        clear_at_least: int = 10_000,
        tools: Iterable[str] | None = None,
    ) -> None:
        self.trigger_tokens = trigger_tokens
        self.keep_recent = keep_recent
        self.clear_at_least = clear_at_least
        self.tools = None if tools is None else frozenset(tools)

    @classmethod
    def placeholder(cls, m: Message) -> str:
        """占位里留线索（工具给的 summary，比如「1 行：total=4242」），模型常常不用重查。

        只依赖消息本身，不掺时间、计数这类会变的东西：占位一变，缓存就废。
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
    def maintain(
        self, entries: list[Entry], measure_view: Callable[[], int], *, force: bool = False,
        prompt: Prompt | None = None,
    ) -> Marker | None:
        if not force and measure_view() <= self.trigger_tokens:
            return None

        targets = self._clearable(entries)
        if not targets:
            return None
        freed = sum(estimate_message(m) - estimate_text(self.placeholder(m)) for m in targets)
        if not force and freed < self.clear_at_least:
            # 省得太少，不值得断一次缓存。强制时（已经超长了）能省一点是一点
            return None
        return ToolResultsCleared(frozenset(m.tool_call_id for m in targets))

    def _clearable(self, entries: list[Entry]) -> list[Message]:
        """还没清过、成功的、白名单里的、不是最近 keep_recent 条、换成占位确实更短的。"""
        messages = [e for e in entries if isinstance(e, Message)]
        cleared = _cleared_ids(entries)
        tool_names = {c.id: c.name for m in messages for c in m.tool_calls}
        results = [m for m in messages if m.role == "tool"]
        older = results[: max(len(results) - self.keep_recent, 0)]
        return [
            m for m in older
            if m.tool_call_id not in cleared
            and not m.is_error
            and (self.tools is None or tool_names.get(m.tool_call_id) in self.tools)
            and estimate_message(m) > estimate_text(self.placeholder(m))
        ]

    def status(self, entries: list[Entry]) -> str | None:
        n = len(_cleared_ids(entries))
        return f"已清理的旧工具结果：{n} 条（原件还在，只是不再发给模型）" if n else None


def _cleared_ids(entries: list[Entry]) -> set[str]:
    ids: set[str] = set()
    for e in entries:
        if isinstance(e, ToolResultsCleared):
            ids |= e.tool_call_ids
    return ids
