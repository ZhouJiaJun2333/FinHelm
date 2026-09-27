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
class ToolCallRepeated:
    """同一轮里同一个工具、同样的参数又调了一次（count 是第几次），工具结果后面附了提醒。"""

    name: str
    count: int


@dataclass(slots=True)
class StepLimitReached:
    max_steps: int
    wrapped_up: bool = False     # 收尾成功：模型根据已有结果给出了回答（不是兜底那句话）
    failure: str = ""            # 收尾没成的原因（又去调工具、被截断、API 报错…），用的是兜底那句话


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
    used_model: bool = False     # 做决定的工序调了模型（写摘要）


@dataclass(slots=True)
class ContextEditFailed:
    """自动整理时，调模型的工序（写摘要）没做成。这一步不中断，带着没压的上下文接着跑。"""

    reason: str
    usage: Usage = field(default_factory=Usage)     # 失败之前已经花掉的
    kind: str = ""                                  # 工序的类名


@dataclass(slots=True)
class AutoCompactionPaused:
    """自动压缩连续失败太多次，本会话不再自动尝试。/compact 手动压成功后恢复。"""

    failures: int


@dataclass(slots=True)
class ContextOverflowed:
    """API 说超出了上下文窗口，接下来强制整理一次再重试。"""

    step: int


Event = (
    LLMResponded | ToolStarted | ToolFinished | ToolDenied | ToolCallRepeated | TurnContinued | StepLimitReached
    | ContextEdited | ContextEditFailed | AutoCompactionPaused | ContextOverflowed
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
