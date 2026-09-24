"""Context：历史是一条只追加的日志，发给模型的视图从日志推算出来。

    历史（只追加）：消息 + 标记
        user、assistant、tool、tool、[标记：清理了 c1,c2]、user、…
        │
        ├─ 工序 1  例如：只保留最近 N 轮
        ├─ 工序 2  例如：看到「清理」标记，就把对应的工具结果换成占位
        ├─ ...     以后：看到「摘要」标记，就把它之前的回合换成摘要……
        ▼
    去掉标记（标记不发给模型）→ 发给模型的视图 = render()

── 状态放在哪：日志里，不放在工序对象里 ─────────────────────────────
工序做出的决定（清理了哪些、摘要写了什么）作为**标记**追加进历史；
工序本身不持有任何状态，只是一个「看着日志算视图」的投影。这是 pi 的做法
（它的会话是 JSONL 日志，压缩结果是一条记录，上下文由 buildSessionContext
从日志推算）。好处：
    · 快照 = 复制一份历史。以前每道工序都要自己实现 snapshot / restore /
      reset，漏一个回滚就不干净；现在没有「别处的状态」可漏。
    · 以后存盘、恢复会话、做分支，存的就是这条日志，天然完整。
    · 摘要压缩产生的摘要文本本来就得存在某处 —— 就是一条标记。

标记总是追加在它影响的消息**之后**（先有工具结果，才有清理它的决定），
所以按回合裁剪切掉一段时，标记和它影响的消息会一起被切掉，不会错位。

── 为什么是「一组工序」而不是「一个子类一种策略」 ────────────────────
继承只能选一种。想要「先清工具结果，不够再做摘要」，要么多重继承，要么
复制代码。Anthropic 的 context_management.edits、pi 的 transformContext
钩子，都是「视图在发送前经过一串加工」这个思路。

── 半截状态毒化历史 ──────────────────────────────────────────────────
一共有三个入口，后果轻重不一样：

    1. assistant 有 tool_calls 但没有对应的 tool 结果
       → 两家都 400，而且是永久的：之后每一轮都报同样的错，会话报废。
       所以任何工序都**不能删工具结果消息**，只能改它的内容。
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
from dataclasses import dataclass
from typing import Callable, Iterable

from ..events import ContextEdited, Event
from ..messages import Message, Usage

# 给一份消息列表估 token 数。由 Agent 提供 —— 只有它知道系统提示词和工具定义。
Measure = Callable[[list[Message]], int]


@dataclass(frozen=True, slots=True)
class Marker:
    """历史里的一条标记：记录某道工序做过的一个决定。不是对话消息，不发给模型。

    每种工序定义自己的标记子类（比如「清理了哪些工具结果」），
    只能追加，不能修改 —— 和消息一样是不可变的。
    """

    def describe(self) -> str:
        """人话描述这个决定，给界面和事件用。"""
        return type(self).__name__

    def cost(self) -> Usage:
        """做这个决定调用模型花了多少（写摘要要花钱，清理不用）。"""
        return Usage()


# 历史里的一条：要么是对话消息，要么是标记
Entry = Message | Marker


# =================================================================== 接口
class BaseContext(ABC):
    """Agent 眼里的上下文。Agent 只调用这几个方法。

    标准实现是下面的 Context。单独留一个接口，是给「历史存在别处」的实现
    留位置（比如存数据库、存文件的持久化会话）。
    """

    @abstractmethod
    def add(self, entry: Entry) -> None:
        """往历史末尾追加一条消息或标记。"""

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
        """几行人话，描述上下文现在的状态（清理了几条…）。给界面用。"""
        return []

    @abstractmethod
    def clear(self) -> None:
        """清空。"""

    @abstractmethod
    def snapshot(self) -> object:
        """拍一张当前状态的快照，交给 restore() 用。

        给 Agent.run() 提供**事务语义**：一轮对话要么完整完成，要么上下文
        回到进来之前的样子。
        """

    @abstractmethod
    def restore(self, snapshot: object) -> None:
        """回滚到 snapshot() 拍下的状态。"""


class ContextEdit(ABC):
    """一道加工工序：一个「看着历史算视图」的投影，自己**不持有状态**。

    写一种新策略：
      · apply()     必须实现：加工视图
      · maintain()  需要「做决定」的才实现：返回一个标记，Context 把它追加进历史
      · status()    想在界面上露脸就实现：从历史里的标记数出来

    写 apply() 的三条规矩：
      · 纯函数：同样的输入必须给出**一模一样**的输出（状态全在输入的标记里）。
        输出一变，发出去的前缀就变，prompt 缓存就废了。
      · 不能拆散 tool_call 和 tool_result：删消息要按完整回合删，
        工具结果只能改内容、不能删消息。
      · 不改原消息：消息是不可变的，要改就生成新对象。
    """

    @abstractmethod
    def apply(self, entries: list[Entry]) -> list[Entry]:
        """加工视图。输入输出都含标记 —— 标记由 Context 在最后统一去掉。"""

    def maintain(self, entries: list[Entry], measure_view: Callable[[], int]) -> Marker | None:
        """请求前的整理机会：要做决定就返回一个标记，不做就返回 None。

        Args:
            entries:      这道工序的输入（前面各道工序加工过的）
            measure_view: 估算当前**最终**视图（所有工序都套用之后）有多少 token
        """
        return None

    def status(self, entries: list[Entry]) -> str | None:
        return None


# ================================================================ 标准实现
class Context(BaseContext):
    """一条只追加的历史 + 一组按顺序套用的编辑工序。不传工序就是全量保留。"""

    def __init__(self, edits: Iterable[ContextEdit] = ()) -> None:
        self._history: list[Entry] = []
        self.edits: list[ContextEdit] = list(edits)

    # ------------------------------------------------------------ 历史
    def add(self, entry: Entry) -> None:
        if isinstance(entry, Message) and entry.meta.usage is not None:
            # 这条回复的 usage 量的是「刚刚发出去的那份视图」，也就是现在的 render()。
            # 盖个指纹，以后视图被哪道工序改了，render() 能认出它已经不准。
            entry = entry.with_meta(measured_on=_fingerprint(self.render()))
        self._history.append(entry)

    @property
    def history(self) -> list[Entry]:
        """原始历史（含标记）的副本。调试、测试、以后存盘用。"""
        return list(self._history)

    def render(self) -> list[Message]:
        view = [e for e in self._apply(len(self.edits)) if isinstance(e, Message)]
        return _drop_stale_anchors(view)

    def _apply(self, upto: int) -> list[Entry]:
        """把历史依次过前 upto 道工序。"""
        view = list(self._history)
        for edit in self.edits[:upto]:
            view = edit.apply(view)
        return view

    # ------------------------------------------------------------ 整理
    def maintain(self, measure: Measure) -> list[Event]:
        def measure_view() -> int:
            return measure(self.render())

        events: list[Event] = []
        for i, edit in enumerate(self.edits):
            before = measure_view()
            marker = edit.maintain(self._apply(i), measure_view)
            if marker is not None:
                self.add(marker)
                events.append(ContextEdited(marker.describe(), before, measure_view(),
                                            usage=marker.cost()))
        return events

    def status(self) -> list[str]:
        return [s for s in (e.status(self._history) for e in self.edits) if s]

    # ------------------------------------------------------- 状态 / 事务
    # 所有状态都在历史里，快照就是复制一份历史（条目都不可变，浅拷贝就够）。
    def clear(self) -> None:
        self._history.clear()

    def snapshot(self) -> list[Entry]:
        return list(self._history)

    def restore(self, snapshot: list[Entry]) -> None:
        self._history[:] = snapshot


# ========================================================= 锚点是否还有效
# usage 锚点的前提是「量它的时候，它前面的内容和现在一样」。
# 统一在这里判断：每个锚点带着「量它时前面那段视图」的指纹，render() 时
# 边走边算当前前缀的指纹，对不上就把 usage 去掉。任何工序改了前面的内容
# （清理、裁剪、摘要……）都会被自动认出来，工序自己不用操心锚点。
#
# pi 没有这一层（它只看最后一条 assistant 的 usage，下一次响应自然会纠正）。
# 我们多做这一步，是为了让「任何工序都不用管锚点」这件事成立：如果改成
# 「看到标记就作废之前的锚点」，那每道改视图的工序都必须记得留标记，
# 而按规则滑动的裁剪根本不产生标记 —— 等于给写工序的人埋了一条隐形规矩。
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
