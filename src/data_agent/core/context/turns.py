"""编辑工序：按「回合」裁剪，只保留最近几轮。"""

from __future__ import annotations

from ..messages import Message
from .base import ContextEdit, Entry


class KeepRecentTurns(ContextEdit):
    """只保留最近 max_turns 轮用户提问及其后续。

    不能简单地「只保留最后 N 条消息」：assistant 的 tool_calls 和后面的
    tool 结果**必须成对出现**，从中间一刀切下去，API 直接 400。
    以回合为单位切，就不会把它们拆散。

    回合的定义：从一条 role="user" 的消息开始，到下一条 role="user" 之前为止。

    ⚠️ 已知问题（下一步修）：finish_turn 补的 nudge 也是 role="user"，
       会被当成新回合的开头 —— 窗口可能切在一轮中间，把用户原来的问题切掉。
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns

    def apply(self, entries: list[Entry]) -> list[Entry]:
        # 切口一定落在一条 user 消息上。标记总是追加在它影响的消息之后，
        # 所以被切掉的标记，它影响的消息也一定在切口之前，不会错位。
        turn_starts = [
            i for i, e in enumerate(entries) if isinstance(e, Message) and e.role == "user"
        ]
        if len(turn_starts) <= self.max_turns:
            return list(entries)
        return list(entries[turn_starts[-self.max_turns]:])
