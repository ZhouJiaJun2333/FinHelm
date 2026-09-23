"""Agent 运行过程中抛出的事件。

为什么要专门定义事件，而不是在主循环里直接 print？
    因为「怎么展示」和「怎么运行」是两件事。主循环只管抛事件，
    CLI 可以打到终端，Web 可以推到前端，测试可以收集起来做断言，
    以后接 trace / 日志 / 监控也是订阅同一批事件。

这一层让 core/agent.py 保持纯粹：它不知道终端的存在。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(slots=True)
class LLMResponded:
    """模型回了一轮。tool_calls 为空说明它认为任务做完了。"""

    step: int
    text: str
    tool_calls: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ToolStarted:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolFinished:
    name: str
    ok: bool
    content: str
    elapsed_ms: int


@dataclass(slots=True)
class ToolDenied:
    """审批钩子拒绝了这次调用。"""

    name: str
    reason: str


@dataclass(slots=True)
class TurnContinued:
    """finish_turn_hook 判定「还没完，接着干」，并补了一条推动消息。"""

    step: int
    nudge: str


@dataclass(slots=True)
class StepLimitReached:
    max_steps: int


Event = (
    LLMResponded | ToolStarted | ToolFinished | ToolDenied
    | TurnContinued | StepLimitReached
)


class EventSink(Protocol):
    """事件消费者。任何 `def (event) -> None` 的可调用对象都能当 sink。"""

    def __call__(self, event: Event) -> None: ...


def noop_sink(event: Event) -> None:
    """默认什么都不做。"""


def collect_sink(bucket: list[Event]) -> Callable[[Event], None]:
    """把事件收进列表，写测试时很好用。"""

    def sink(event: Event) -> None:
        bucket.append(event)

    return sink
