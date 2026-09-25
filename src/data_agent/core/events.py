"""Agent 运行中抛出的事件。主循环只管抛，CLI、测试、评测、以后的 Web 各自订阅。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .messages import Usage


@dataclass(slots=True)
class LLMResponded:
    """模型回了一轮。tool_calls 为空说明它认为任务做完了。"""

    step: int
    text: str
    tool_calls: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # 带上窗口大小，界面才能算百分比
    context_window: int | None = None


@dataclass(slots=True)
class ToolStarted:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolFinished:
    name: str
    content: str             # 模型看到的那份
    is_error: bool
    elapsed_ms: int
    # ToolOutput.details，只给界面
    details: Any = None


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


@dataclass(slots=True)
class ContextEdited:
    """某道上下文工序做了一次决定（清理、压缩…）。所有工序共用这一个事件。token 数是估算值。"""

    description: str         # 人话，比如「清理了 5 条较早的工具结果」
    tokens_before: int
    tokens_after: int
    # 这次整理调用模型花的 token（写摘要），Agent 记进 session_usage
    usage: Usage = field(default_factory=Usage)
    # 标记的类名（ToolResultsCleared / HistoryCompacted…），评测靠它分开统计
    kind: str = ""


@dataclass(slots=True)
class ContextOverflowed:
    """API 说超出了上下文窗口，接下来强制整理一次再重试。"""

    step: int


Event = (
    LLMResponded | ToolStarted | ToolFinished | ToolDenied
    | TurnContinued | StepLimitReached | ContextEdited | ContextOverflowed
)


class EventSink(Protocol):
    """任何 def (event) -> None 的可调用对象。"""

    def __call__(self, event: Event) -> None: ...


def noop_sink(event: Event) -> None:
    pass



def collect_sink(bucket: list[Event]) -> Callable[[Event], None]:
    """把事件收进列表（测试、评测用）。"""

    def sink(event: Event) -> None:
        bucket.append(event)

    return sink
