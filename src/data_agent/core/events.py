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

from .messages import Usage


@dataclass(slots=True)
class LLMResponded:
    """模型回了一轮。tool_calls 为空说明它认为任务做完了。"""

    step: int
    text: str
    tool_calls: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # 带上窗口大小，界面才能算百分比 —— 事件自带完整信息，消费者不用回头问 Agent
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
    # 工具给界面的数据（ToolOutput.details），模型看不到。界面想展示完整结果就用它
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
    """某道上下文编辑工序在请求前做了一次决定（清理了一批工具结果、做了一次摘要…）。

    所有工序共用这一个事件，界面不用为每种新工序加一个分支。token 数都是估算值。
    """

    description: str         # 人话，比如「清理了 5 条较早的工具结果」
    tokens_before: int
    tokens_after: int
    # 这次整理本身调用模型花的 token（写摘要）。清理之类不调模型的是全 0。
    # Agent 会把它记进 session_usage —— 钱花了就得记账。
    usage: Usage = field(default_factory=Usage)
    # 做决定的是哪种标记（ToolResultsCleared / HistoryCompacted …）。界面不用管，
    # 评测靠它分开统计清理和压缩 —— 别去解析 description 那句人话
    kind: str = ""


@dataclass(slots=True)
class ContextOverflowed:
    """API 说请求超出了上下文窗口。接下来会强制整理一次再重试（之后跟着 ContextEdited）。"""

    step: int


Event = (
    LLMResponded | ToolStarted | ToolFinished | ToolDenied
    | TurnContinued | StepLimitReached | ContextEdited | ContextOverflowed
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
