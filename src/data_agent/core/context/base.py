"""Context：存历史，按顺序套用一组编辑工序，得到真正发给模型的视图。

    内部历史（只追加，原件不动）
        │
        ├─ edit 1  例如：只保留最近 N 轮
        ├─ edit 2  例如：把较早的工具结果换成占位
        ├─ ...     以后：摘要压缩、去重、图片降级……
        ▼
    发给模型的视图 = render()

为什么是「一组工序」而不是「一个子类一种策略」：
    继承只能选一种。想要「先清工具结果，不够再做摘要」，要么多重继承，
    要么复制代码。Anthropic 自己的接口也是这么设计的 ——
    context_management.edits 是一个列表，按顺序生效。
    加一种新策略 = 写一个新的 ContextEdit，Context 和 Agent 一行不用改。

── 半截状态毒化历史 ──────────────────────────────────────────────────
一共有三个入口，后果轻重不一样：

    1. assistant 有 tool_calls 但没有对应的 tool 结果
       → 两家都 400，而且是永久的：之后每一轮都报同样的错，会话报废。
       所以任何编辑工序都**不能删工具结果消息**，只能改它的内容。
    2. 连着两条 user 消息（提问后这轮失败了，用户又问一次）
       → 不报错，连续同角色的消息会被合并成一条。坏在语义：失败那轮的问题
         和新问题被悄悄拼在一起。不报错反而更难发现。
    3. 历史以 tool 结果或 nudge 的 user 消息结尾，没有收尾的 assistant
       → 下一轮追加 user 后同样被合并；而且模型不知道上一轮卡住了。

前两个走异常路径，靠 Agent.run() 的事务语义（snapshot/restore）兜住；
第三个走正常返回路径（步数耗尽），由 run() 自己补一条收尾的 assistant。

反过来，「以 assistant 结尾」发请求会报错：Anthropic 把末尾的 assistant 当成
prefill，Opus 4.6 之后的模型不支持，直接 400。所以 finish_turn 继续时要补 nudge。

tests/ 里的哨兵按「严格交替 + tool_use 配对」来验。交替不是 API 硬约束，
是我们自己要的不变量：它保证每个问题有且只有一个回答。
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

from ..events import Event
from ..messages import Message

# 给一份消息列表估 token 数。由 Agent 提供 —— 只有它知道系统提示词和工具定义。
Measure = Callable[[list[Message]], int]


# =================================================================== 接口
class BaseContext(ABC):
    """Agent 眼里的上下文。Agent 只调用这几个方法。

    标准实现是下面的 Context。单独留一个接口，是给「历史存在别处」的实现
    留位置（比如存数据库、存文件的持久化会话）。
    """

    @abstractmethod
    def add(self, message: Message) -> None:
        """追加一条消息。"""

    @abstractmethod
    def render(self) -> list[Message]:
        """返回这一次真正要发给模型的消息列表。

        叫 render 不叫 messages：它是「算出来的视图」，可以和内部存的历史
        不一样（裁剪、清理、摘要都发生在这里）。
        """

    def maintain(self, measure: Measure) -> list[Event]:
        """每次请求模型**之前**调用，给上下文一个整理自己的机会。

        返回这次做了什么（事件），交给界面展示。默认什么都不做。

        为什么是「请求前」而不是「追加消息时」：要不要整理取决于**整个请求**
        有多大（含系统提示词、工具定义），只有马上要发的时候才算得准。
        """
        return []

    def status(self) -> list[str]:
        """几行人话，描述上下文现在的状态（清理了几条、裁掉了几轮…）。给界面用。"""
        return []

    @abstractmethod
    def clear(self) -> None:
        """清空。"""

    @abstractmethod
    def snapshot(self) -> Any:
        """拍一张当前状态的快照，交给 restore() 用。

        给 Agent.run() 提供**事务语义**：一轮对话要么完整完成，要么上下文
        回到进来之前的样子。快照必须包含**全部**状态 —— 历史，以及各个编辑
        工序自己记的东西（清理了哪些）。漏一样，回滚就不干净。
        """

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """回滚到 snapshot() 拍下的状态。"""


class ContextEdit(ABC):
    """一道加工工序：把上一道工序给的消息列表，加工成下一道要用的。

    写一种新策略，只需要实现 apply()。有状态的（比如记着清理了哪些）再实现
    maintain / snapshot / restore / reset；想在界面上露脸就实现 status。

    写 apply() 的三条规矩：
      · 纯函数：给定工序自己的状态，同样的输入必须给出**一模一样**的输出。
        输出一变，发出去的前缀就变，prompt 缓存就废了。
      · 不能拆散 tool_call 和 tool_result：删消息要按完整回合删，
        工具结果只能改内容、不能删消息。
      · 不改原消息：消息是不可变的，要改就生成新对象。
    """

    @abstractmethod
    def apply(self, messages: list[Message]) -> list[Message]:
        """加工视图。"""

    def maintain(self, messages: list[Message], measure_view: Callable[[], int]) -> Event | None:
        """请求前的整理机会：决定要不要改变自己的状态（比如再清理一批）。

        Args:
            messages:     这道工序的输入（前面各道工序加工过的）
            measure_view: 估算**最终**视图（所有工序都套用之后）有多少 token。
                          状态改了之后再调一次，就能知道省了多少。
        """
        return None

    def snapshot(self) -> Any:
        return None

    def restore(self, snapshot: Any) -> None:
        pass

    def reset(self) -> None:
        """对话清空时调用。"""

    def status(self) -> str | None:
        return None


# ================================================================ 标准实现
class Context(BaseContext):
    """存历史 + 一组按顺序套用的编辑工序。不传工序就是全量保留。"""

    def __init__(self, edits: Iterable[ContextEdit] = ()) -> None:
        self._history: list[Message] = []
        self.edits: list[ContextEdit] = list(edits)

    # ------------------------------------------------------------ 历史
    def add(self, message: Message) -> None:
        if message.meta.usage is not None:
            # 这条回复的 usage 量的是「刚刚发出去的那份视图」，也就是现在的 render()。
            # 盖个指纹，以后视图被哪道工序改了，render() 能认出它已经不准。
            message = message.with_meta(measured_on=_fingerprint(self.render()))
        self._history.append(message)

    def render(self) -> list[Message]:
        return _drop_stale_anchors(self._apply(len(self.edits)))

    def _apply(self, upto: int) -> list[Message]:
        """把历史依次过前 upto 道工序。"""
        view = list(self._history)
        for edit in self.edits[:upto]:
            view = edit.apply(view)
        return view

    # ------------------------------------------------------------ 整理
    def maintain(self, measure: Measure) -> list[Event]:
        events: list[Event] = []
        for i, edit in enumerate(self.edits):
            event = edit.maintain(self._apply(i), lambda: measure(self.render()))
            if event is not None:
                events.append(event)
        return events

    def status(self) -> list[str]:
        return [s for s in (e.status() for e in self.edits) if s]

    # ------------------------------------------------------- 状态 / 事务
    def clear(self) -> None:
        self._history.clear()
        for edit in self.edits:
            edit.reset()

    # 历史存整个列表的浅拷贝（消息不可变，浅拷贝就够）+ 每道工序自己的状态。
    def snapshot(self) -> tuple[list[Message], list[Any]]:
        return list(self._history), [e.snapshot() for e in self.edits]

    def restore(self, snapshot: tuple[list[Message], list[Any]]) -> None:
        history, edit_states = snapshot
        self._history[:] = history
        for edit, state in zip(self.edits, edit_states):
            edit.restore(state)


# ========================================================= 锚点是否还有效
# usage 锚点的前提是「量它的时候，它前面的内容和现在一样」。
# 以前这个判断写在清理策略里，只认得清理这一种改法，每加一种策略都得重写一遍。
# 现在统一在这里做：每个锚点带着「量它时前面那段视图」的指纹，render() 时
# 边走边算当前前缀的指纹，对不上就把 usage 去掉。任何工序改了前面的内容
# （清理、裁剪、摘要……）都会被自动认出来，工序自己不用操心锚点。
def _drop_stale_anchors(view: list[Message]) -> list[Message]:
    running = hashlib.sha1()
    out: list[Message] = []
    for m in view:
        if m.meta.usage is not None and m.meta.measured_on != running.hexdigest():
            m = m.with_meta(usage=None)          # 生成新对象，原消息不动
        running.update(_wire_bytes(m))
        out.append(m)
    return out


def _fingerprint(view: list[Message]) -> str:
    running = hashlib.sha1()
    for m in view:
        running.update(_wire_bytes(m))
    return running.hexdigest()


def _wire_bytes(m: Message) -> bytes:
    """一条消息里**会发给模型**的部分。meta 不算 —— 它不影响请求内容。"""
    return json.dumps(
        [m.role, m.content, m.tool_call_id,
         [[c.id, c.name, c.arguments] for c in m.tool_calls],
         repr(m.raw)],
        ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8") + b"\x00"
