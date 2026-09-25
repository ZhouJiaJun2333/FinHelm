"""Context：历史是一条只追加的日志（消息 + 标记），发给模型的视图由一组工序从日志算出来。

    历史  user、assistant、tool、[标记：清理了 c1,c2]、user、…
      │   工序 1、工序 2 …（清理工具结果、摘要压缩）依次加工
      ▼
    视图（去掉标记）= render()

工序做出的决定作为标记追加进历史，工序本身不持有状态（学 pi：会话是日志，压缩是其中一条记录）。
所以快照 = 复制历史，存盘 = 写日志。标记总追加在它影响的消息之后，按回合裁剪时会一起被切掉。

历史必须保持的形状（tests 里的哨兵按这个验）：
    · assistant 的 tool_calls 必须有对应的 tool 结果，否则之后每次请求都 400
      —— 工序不能删工具结果消息，只能改内容
    · user / assistant 严格交替：连续两条 user 不报错，但会被悄悄合并成一条
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterable

from ..errors import CompactionFailed
from ..events import ContextEdited, ContextEditFailed, Event
from ..messages import Message, Usage

# 给一份消息列表估 token 数。由 Agent 提供 —— 只有它知道系统提示词和工具定义。
Measure = Callable[[list[Message]], int]


@dataclass(frozen=True, slots=True)
class Prompt:
    """一次请求里消息以外的部分。写摘要时原样带上，才能和平时的请求共用缓存前缀。"""

    system: str | None = None
    tools: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class Marker:
    """某道工序做过的一个决定。不可变，只追加，不发给模型。每种工序定义自己的子类。"""

    def describe(self) -> str:
        """给界面看的一句人话。"""
        return type(self).__name__

    def cost(self) -> Usage:
        """做这个决定调用模型花了多少（写摘要要花钱，清理不用）。"""
        return Usage()


# 历史里的一条：要么是对话消息，要么是标记
Entry = Message | Marker


# =================================================================== 接口
class BaseContext(ABC):
    """Agent 眼里的上下文。标准实现是下面的 Context。"""

    @abstractmethod
    def add(self, entry: Entry) -> None:
        """往历史末尾追加一条消息或标记。"""

    @abstractmethod
    def render(self) -> list[Message]:
        """这一次真正要发给模型的消息（算出来的视图，不是历史本身）。"""

    def maintain(self, measure: Measure, *, force: bool = False, prompt: Prompt | None = None,
                 model_calls: bool = True) -> list[Event]:
        """每次请求模型之前调用：超阈值就整理。返回做了什么（事件）。

        force：不看阈值，能整理的都整理（API 报超长、用户 /compact）。调模型的工序失败时抛
               CompactionFailed；不强制时只记一个 ContextEditFailed 事件，接着跑。
        prompt：这次请求的系统提示词和工具定义，写摘要时用来复用缓存。
        model_calls：False 时跳过调模型的工序（Agent 的熔断用）。
        """
        return []

    def status(self) -> list[str]:
        """几行人话描述现在的状态（清理了几条…），给界面用。"""
        return []

    @abstractmethod
    def clear(self) -> None:
        """清空。"""

    @abstractmethod
    def snapshot(self) -> object:
        """给 Agent.run() 的事务用：失败时 restore 回来。"""

    @abstractmethod
    def restore(self, snapshot: object) -> None:
        """回滚到 snapshot() 拍下的状态。"""


class ContextEdit(ABC):
    """一道工序：看着历史算视图，自己不持有状态。

    apply()    必须实现。纯函数（输出一变缓存就废）；不拆散 tool_call 和结果；不改原消息
    maintain() 要做决定的才实现：返回一个标记，Context 追加进历史
    status()   想在界面上露脸就实现
    """

    # 要调模型（写摘要）的工序。它们会失败、会花钱，Agent 的熔断只停它们
    calls_model: ClassVar[bool] = False

    @abstractmethod
    def apply(self, entries: list[Entry]) -> list[Entry]:
        """加工视图。输入输出都含标记 —— 标记由 Context 在最后统一去掉。"""

    def maintain(
        self, entries: list[Entry], measure_view: Callable[[], int], *, force: bool = False,
        prompt: Prompt | None = None,
    ) -> Marker | None:
        """entries 是前面各道工序加工过的输入；measure_view 估的是所有工序套用之后的最终视图。"""
        return None

    def status(self, entries: list[Entry]) -> str | None:
        return None


# ================================================================ 标准实现
class Context(BaseContext):
    """一条只追加的历史 + 一组按顺序套用的工序。不传工序就是全量保留。"""

    def __init__(self, edits: Iterable[ContextEdit] = ()) -> None:
        self._history: list[Entry] = []
        self.edits: list[ContextEdit] = list(edits)

    # ------------------------------------------------------------ 历史
    def add(self, entry: Entry) -> None:
        if isinstance(entry, Message) and entry.meta.usage is not None:
            # usage 量的是刚发出去的那份视图（现在的 render()）。盖个指纹，前缀被改了就能认出来
            entry = entry.with_meta(measured_on=_fingerprint(self.render()))
        self._history.append(entry)

    @property
    def history(self) -> list[Entry]:
        """原始历史（含标记）的副本。"""
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
    def maintain(self, measure: Measure, *, force: bool = False, prompt: Prompt | None = None,
                 model_calls: bool = True) -> list[Event]:
        def measure_view() -> int:
            return measure(self.render())

        events: list[Event] = []
        for i, edit in enumerate(self.edits):
            if edit.calls_model and not model_calls:
                continue
            before = measure_view()
            try:
                marker = edit.maintain(self._apply(i), measure_view, force=force, prompt=prompt)
            except CompactionFailed as exc:
                if force:
                    raise
                events.append(ContextEditFailed(str(exc), usage=exc.usage, kind=type(edit).__name__))
                continue
            if marker is not None:
                self.add(marker)
                events.append(ContextEdited(marker.describe(), before, measure_view(),
                                            usage=marker.cost(), kind=type(marker).__name__,
                                            used_model=edit.calls_model))
        return events

    def status(self) -> list[str]:
        return [s for s in (e.status(self._history) for e in self.edits) if s]

    # ------------------------------------------------------- 状态 / 事务
    # 所有状态都在历史里，条目都不可变，浅拷贝就够
    def clear(self) -> None:
        self._history.clear()

    def snapshot(self) -> list[Entry]:
        return list(self._history)

    def restore(self, snapshot: list[Entry]) -> None:
        self._history[:] = snapshot


# ========================================================= 锚点是否还有效
# usage 锚点的前提是「它前面的内容和量它的时候一样」。render() 边走边算前缀指纹，
# 对不上就去掉 usage —— 任何工序改了前缀都会被认出来，工序自己不用管锚点。（pi 没有这层）
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
    """一条消息里会发给模型的部分（不含 meta）。图片只取摘要；没图时和加图片之前的指纹一样。"""
    fields = [m.role, m.content, m.tool_call_id, m.is_error,
              [[c.id, c.name, c.arguments] for c in m.tool_calls],
              repr(m.raw)]
    if m.images:
        fields.append([hashlib.sha1(i.data.encode()).hexdigest() for i in m.images])
    return json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8") + b"\x00"
