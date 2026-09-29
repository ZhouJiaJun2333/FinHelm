"""运行状态：从事件折叠出来（学 pi 的 AgentState）。界面要画的东西都从这里读，不去翻 Agent 的内部字段。

    state = reduce(state, event)

reduce 是纯函数（不调模型、不看时钟）：同一串事件永远折出同一个状态，所以前端拿到事件流能自己折出一份一样的。
Agent 每个事件先 reduce 再交给订阅者，订阅者收到事件时 agent.state 已经是新的。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .events import (
    ContextEdited,
    ContextEditFailed,
    ConversationReset,
    Event,
    LLMResponded,
    StepStarted,
    TextDelta,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    TurnResumed,
    TurnStarted,
    UserAsked,
)
from .messages import Usage

# idle：空闲（上一轮答完了或断了，断了看 error）；thinking：在等模型（含整理上下文、流式输出）；
# tools：在跑工具；asking：停下来等用户回答
Status = Literal["idle", "thinking", "tools", "asking"]
ToolStatus = Literal["running", "done", "error", "denied", "asking"]


@dataclass(frozen=True, slots=True)
class ToolRun:
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: ToolStatus = "running"
    elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class AgentState:
    status: Status = "idle"
    question: str = ""                     # 这一轮的提问
    step: int = 0
    streaming_text: str = ""               # 这一步正在输出的回答，LLMResponded 后清空（pi 的 streamingMessage）
    streaming_thinking: str = ""
    tools: tuple[ToolRun, ...] = ()        # 这一轮的工具调用，按顺序
    asking: UserAsked | None = None        # 在等回答的问题
    answer: str = ""                       # 这一轮的回答（结束后才有）
    error: str = ""                        # 这一轮没跑完的原因（pi 的 errorMessage）
    turn_usage: Usage = field(default_factory=Usage)   # 这一轮花的 token，含写摘要
    context_tokens: int = 0                # 最近一次模型回复之后，对话占多少上下文
    context_window: int | None = None

    @property
    def running(self) -> bool:
        return self.status in ("thinking", "tools")

    @property
    def pending_tools(self) -> tuple[ToolRun, ...]:
        """正在跑的工具（pi 的 pendingToolCalls）。"""
        return tuple(t for t in self.tools if t.status == "running")


def reduce(state: AgentState, event: Event) -> AgentState:
    match event:
        case TurnStarted(question=question):
            # 新的一轮从零开始，只留上下文用量（它属于对话，不属于这一轮）
            return AgentState(status="thinking", question=question,
                              context_tokens=state.context_tokens, context_window=state.context_window)

        case TurnResumed():
            return replace(state, status="thinking", asking=None, answer="", error="")

        case StepStarted(step=step):
            return replace(state, status="thinking", step=step, streaming_text="", streaming_thinking="")

        case TextDelta(text=text, thinking=True):
            return replace(state, streaming_thinking=state.streaming_thinking + text)

        case TextDelta(text=text):
            return replace(state, streaming_text=state.streaming_text + text)

        case LLMResponded(step=step, usage=usage, context_window=window):
            return replace(state, step=step, streaming_text="", streaming_thinking="",
                           turn_usage=state.turn_usage + usage,
                           context_tokens=usage.context_tokens or state.context_tokens,
                           context_window=window)

        case ToolStarted(name=name, arguments=arguments, call_id=call_id):
            return replace(state, status="tools", tools=(*state.tools, ToolRun(call_id, name, arguments)))

        case ToolFinished(name=name, is_error=is_error, elapsed_ms=ms, call_id=call_id):
            return replace(state, tools=_settle(state.tools, call_id, name,
                                                "error" if is_error else "done", ms))

        case ToolDenied(name=name, call_id=call_id):
            return replace(state, tools=_settle(state.tools, call_id, name, "denied"))

        case UserAsked(name=name, call_id=call_id):
            return replace(state, asking=event, tools=_settle(state.tools, call_id, name, "asking"))

        case ContextEdited(usage=usage) | ContextEditFailed(usage=usage):
            return replace(state, turn_usage=state.turn_usage + usage)

        case TurnEnded(answer=answer, interrupted=why, awaiting_user=waiting):
            # 断在工具执行中间的，模型那边补的是「中断，结果未知」（is_error），这里也记成出错
            tools = tuple(replace(t, status="error") if t.status == "running" else t for t in state.tools)
            return replace(state, status="asking" if waiting else "idle", tools=tools,
                           streaming_text="", streaming_thinking="", answer=answer,
                           error="" if waiting else why, asking=state.asking if waiting else None)

        case ConversationReset():
            return AgentState(context_window=state.context_window)

    return state            # 其余事件（提醒、步数上限…）只给界面看，不改状态


def fold(events: list[Event], state: AgentState | None = None) -> AgentState:
    state = state or AgentState()
    for event in events:
        state = reduce(state, event)
    return state


def _settle(tools: tuple[ToolRun, ...], call_id: str, name: str, status: ToolStatus,
            elapsed_ms: int = 0) -> tuple[ToolRun, ...]:
    """把那次调用改成 status。找不到（进程重启后接着跑，前面的调用不在状态里）就补一条。"""
    for i in range(len(tools) - 1, -1, -1):
        t = tools[i]
        if t.call_id == call_id and t.name == name and t.status in ("running", "asking"):
            return (*tools[:i], replace(t, status=status, elapsed_ms=elapsed_ms), *tools[i + 1:])
    return (*tools, ToolRun(call_id, name, {}, status, elapsed_ms))
