"""编辑工序：按「回合」裁剪，只保留最近几轮。"""

from __future__ import annotations

from ..messages import Message
from .base import ContextEdit, Entry


def turn_starts(entries: list[Entry]) -> list[int]:
    """每个回合从哪一条开始（下标）。

    回合 = 一次**真人提问**，加上它引出的所有东西：模型回复、工具调用和结果、
    Agent 补的 nudge，一直到下一次真人提问之前。

    不能只看 role="user"：finish_turn 让模型继续时补的 nudge 也是 user 角色。
    把它当成回合起点，切口就可能落在一轮中间 —— 用户原来的问题被切掉，
    模型只看到一句「还缺占比，补上」，不知道要补什么。
    所以 Agent 补的消息都标了 meta.synthetic，这里跳过它们。

    以后摘要压缩找「从哪里开始摘要」也用这个切口。
    """
    return [
        i for i, e in enumerate(entries)
        if isinstance(e, Message) and e.role == "user" and not e.meta.synthetic
    ]


class KeepRecentTurns(ContextEdit):
    """只保留最近 max_turns 个回合。

    不能简单地「只保留最后 N 条消息」：assistant 的 tool_calls 和后面的
    tool 结果**必须成对出现**，从中间一刀切下去，API 直接 400。
    以回合为单位切，就不会把它们拆散。
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns

    def apply(self, entries: list[Entry]) -> list[Entry]:
        # 切口一定落在一条真人提问上。标记总是追加在它影响的消息之后，
        # 所以被切掉的标记，它影响的消息也一定在切口之前，不会错位。
        starts = turn_starts(entries)
        if len(starts) <= self.max_turns:
            return list(entries)
        return list(entries[starts[-self.max_turns]:])
