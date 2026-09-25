"""回合的切分，以及按回合裁剪的工序。"""

from __future__ import annotations

from ..messages import Message
from .base import ContextEdit, Entry


def turn_starts(entries: list[Entry]) -> list[int]:
    """每个回合从哪一条开始（下标）。回合 = 一次真人提问到下一次之前。

    Agent 补的 nudge 也是 user 角色，但标了 synthetic，不算回合开头 —— 否则切口落在一轮中间，
    模型只看到「还缺占比，补上」却看不到原来的问题。
    """
    return [
        i for i, e in enumerate(entries)
        if isinstance(e, Message) and e.role == "user" and not e.meta.synthetic
    ]


class KeepRecentTurns(ContextEdit):
    """只保留最近 max_turns 个回合。按回合切，tool_call 和结果不会被拆散。"""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns

    def apply(self, entries: list[Entry]) -> list[Entry]:
        starts = turn_starts(entries)
        if len(starts) <= self.max_turns:
            return list(entries)
        return list(entries[starts[-self.max_turns]:])
