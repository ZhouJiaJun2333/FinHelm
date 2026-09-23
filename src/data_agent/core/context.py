"""上下文管理 —— 「对话变长之后怎么办」的扩展点。

现在只有最朴素的实现。先把接口定下来，以后加摘要压缩 / 向量检索 / 记忆注入
只要写一个新的子类，agent.py 一行不用改。

⚠️ 新手最容易踩的坑：
    不能简单地「只保留最后 N 条消息」。assistant 的 tool_calls 和后面 role="tool"
    的结果**必须成对出现**，从中间一刀切下去，API 直接返回 400。
    真要裁剪，得以「一个完整回合」为单位 —— 见 TurnWindowContext。

同一个坑还有另外两个入口，都是「半截状态毒化历史」：
    · assistant 有 tool_calls 但没有对应的 tool 结果  → 每轮都 400
    · 连着两条 user 消息（提问后这轮失败了，用户又问一次）→ Anthropic 角色
      不交替，同样每轮都 400（OpenAI 兼容接口宽容，所以只在换厂商时才炸）
两个都靠 snapshot/restore 兜住 —— 见 Agent.run() 的事务语义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .messages import Message


class BaseContext(ABC):
    """上下文管理器接口。"""

    @abstractmethod
    def add(self, message: Message) -> None:
        """追加一条消息。"""

    @abstractmethod
    def render(self) -> list[Message]:
        """返回这一轮真正要发给模型的消息列表。

        注意方法名是 render 不是 messages —— 它强调这是「算出来的视图」，
        可以和内部存的历史不一样（压缩、裁剪、插入检索结果都发生在这里）。
        """

    @abstractmethod
    def clear(self) -> None:
        """清空。"""

    @abstractmethod
    def snapshot(self) -> Any:
        """拍一张当前状态的快照，交给 restore() 用。

        用来给 Agent.run() 提供**事务语义**：一轮对话要么完整完成，
        要么历史回到进来之前的样子，不留任何残骸。

        为什么需要：一轮失败（被截断、网络错、用户 Ctrl-C）时，历史里可能
        留下「有提问没回答」或「有工具调用没结果」的半截状态。这种状态会让
        **之后每一轮**请求都 400，一次失败升级成整个会话报废。
        """

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """回滚到 snapshot() 拍下的状态。"""


class FullContext(BaseContext):
    """全量保留。够用到你开始撞上下文上限为止。"""

    def __init__(self) -> None:
        self._history: list[Message] = []

    def add(self, message: Message) -> None:
        self._history.append(message)

    def render(self) -> list[Message]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    # 快照存整个列表的浅拷贝，而不是只记长度。
    # 只记长度对「只追加」的实现够用，但压缩类上下文会**改写**已有消息，
    # 那时候长度回滚不了内容。拷贝一份最省心，历史规模也就几百条。
    def snapshot(self) -> list[Message]:
        return list(self._history)

    def restore(self, snapshot: list[Message]) -> None:
        self._history[:] = snapshot

    def __len__(self) -> int:
        return len(self._history)


class TurnWindowContext(FullContext):
    """按「回合」裁剪：只保留最近 max_turns 轮用户提问及其后续。

    回合的定义：从一条 role="user" 的消息开始，到下一条 role="user" 之前为止。
    以回合为单位切，就不会把 tool_calls / tool_result 拆散。
    """

    def __init__(self, max_turns: int = 6) -> None:
        super().__init__()
        self.max_turns = max_turns

    def render(self) -> list[Message]:
        history = self._history
        turn_starts = [i for i, m in enumerate(history) if m.role == "user"]
        if len(turn_starts) <= self.max_turns:
            return list(history)
        cut = turn_starts[-self.max_turns]
        return list(history[cut:])
