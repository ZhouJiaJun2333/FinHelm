"""【设计草图，没有接进主程序】事件 + reducer 版的 Agent 状态机。

    python docs/state_sketch.py

这是把 pi 的 runtime/reducer.ts 思路搬到我们代码里的样子。先看类型，
判断值不值得上，再决定要不要真改 core/agent.py。

核心思想只有一句：
    **状态不是散在对象字段里的变量，而是把事件流折叠出来的结果。**

        state = reduce(reduce(reduce(初始状态, e1), e2), e3)

一旦做到这一点，白送四样东西：
    持久化   存事件流就行，不用序列化对象图
    恢复     重放事件就回到断点 —— 这是"可中断审批"的地基
    UI 同步  前端订阅事件流，自己 fold 出一份一模一样的状态
    可测     构造一串事件，断言最终状态，不用跑模型

对比现在的 core/agent.py：
    状态散在 Agent 实例和局部变量里（step 是 for 的循环变量、
    历史在 context 里、当前处理哪个 call 是个临时变量）。
    函数一返回就全没了，所以没法中断和恢复。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from typing import Any, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==========================================================================
# 一、状态：一个可序列化的快照
# ==========================================================================
ToolStatus = Literal["queued", "running", "succeeded", "failed", "denied"]

RunStatus = Literal[
    "thinking",          # 等模型回复
    "executing",         # 在跑工具
    "suspended",         # ★ 挂起，等人审批 —— 现在的实现根本表达不了这个状态
    "compacting",        # 在压缩上下文
    "done",
    "faulted",
]


@dataclass(frozen=True, slots=True)
class ToolCallState:
    """一次工具调用的完整生命周期。"""

    id: str
    name: str
    arguments: dict[str, Any]
    status: ToolStatus = "queued"
    output: str | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True, slots=True)
class RunState:
    """一次 run() 的全部状态。**这个对象存进数据库，就能换个进程接着跑。**"""

    run_id: str
    status: RunStatus = "thinking"
    step: int = 0

    # 对话历史。真实实现里是 list[Message]，草图里用 str 省事
    transcript: tuple[str, ...] = ()

    # 当前这一轮待处理/已处理的工具调用
    calls: tuple[ToolCallState, ...] = ()

    # 挂起时记录在等哪一个调用的审批
    awaiting_call_id: str | None = None

    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    result: str | None = None
    fault: str | None = None


# ==========================================================================
# 二、事件：把「发生了什么」建模成数据
# ==========================================================================
# 注意每个事件都是纯数据、可 JSON 序列化。这是能落盘和重放的前提。

@dataclass(frozen=True, slots=True)
class UserAsked:
    text: str


@dataclass(frozen=True, slots=True)
class AssistantResponded:
    step: int
    text: str
    tool_calls: tuple[ToolCallState, ...]
    usage: dict[str, int]
    stop_reason: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call_id: str
    ok: bool
    output: str
    elapsed_ms: int


# ---- 下面三个是现在的实现完全没有的能力 ----
@dataclass(frozen=True, slots=True)
class RunSuspended:
    """挂起等审批。状态存库，HTTP 连接可以断开，用户慢慢点。"""

    call_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class RunResumed:
    call_id: str
    approved: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class CompactionFinished:
    """上下文压缩完成 —— 在 pi 里压缩是和 run 平级的一等操作，不是工具函数。"""

    replaced_count: int
    summary: str


@dataclass(frozen=True, slots=True)
class RunFinished:
    result: str


@dataclass(frozen=True, slots=True)
class RunFaulted:
    error: str


Event = (
    UserAsked | AssistantResponded | ToolStarted | ToolFinished
    | RunSuspended | RunResumed | CompactionFinished | RunFinished | RunFaulted
)


# ==========================================================================
# 三、reducer：纯函数，(状态, 事件) -> 新状态
# ==========================================================================
# 唯一的纪律：这里面不准有副作用。不调模型、不查库、不打日志、不看时钟。
# 破坏了这条，重放就不再可靠，上面说的四样好处全部失效。

def reduce(state: RunState, event: Event) -> RunState:
    match event:
        case UserAsked(text=text):
            return replace(
                state,
                transcript=state.transcript + (f"user: {text}",),
                status="thinking",
            )

        case AssistantResponded(step=step, text=text, tool_calls=calls,
                                usage=usage, stop_reason=stop):
            merged = {k: state.usage.get(k, 0) + v for k, v in usage.items()}
            return replace(
                state,
                step=step,
                transcript=state.transcript + (f"assistant: {text}",),
                calls=calls,
                usage={**state.usage, **merged},
                stop_reason=stop,
                status="executing" if calls else "done",
            )

        case ToolStarted(call_id=cid):
            return replace(state, calls=_patch(state.calls, cid, status="running"))

        case ToolFinished(call_id=cid, ok=ok, output=out, elapsed_ms=ms):
            return replace(
                state,
                calls=_patch(state.calls, cid,
                             status="succeeded" if ok else "failed",
                             output=out, elapsed_ms=ms),
                transcript=state.transcript + (f"tool[{cid}]: {out[:40]}…",),
            )

        case RunSuspended(call_id=cid, reason=reason):
            return replace(state, status="suspended", awaiting_call_id=cid,
                           fault=None, result=reason)

        case RunResumed(call_id=cid, approved=approved, note=note):
            if approved:
                return replace(state, status="executing",
                               awaiting_call_id=None, result=None)
            return replace(
                state,
                status="executing",
                awaiting_call_id=None,
                result=None,
                calls=_patch(state.calls, cid, status="denied",
                             output=f"用户拒绝：{note}"),
            )

        case CompactionFinished(replaced_count=n, summary=summary):
            # 把前 n 条历史换成一条摘要
            return replace(
                state,
                transcript=(f"[摘要 · 压缩了 {n} 条] {summary}",) + state.transcript[n:],
                status="thinking",
            )

        case RunFinished(result=result):
            return replace(state, status="done", result=result)

        case RunFaulted(error=error):
            return replace(state, status="faulted", fault=error)

    raise TypeError(f"未处理的事件类型：{type(event).__name__}")


def _patch(calls: tuple[ToolCallState, ...], call_id: str, **changes) -> tuple[ToolCallState, ...]:
    return tuple(replace(c, **changes) if c.id == call_id else c for c in calls)


def fold(events: list[Event], run_id: str = "run-1") -> RunState:
    """把整条事件流折叠成状态。**恢复现场就是调它。**"""
    state = RunState(run_id=run_id)
    for event in events:
        state = reduce(state, event)
    return state


# ==========================================================================
# 四、主循环会变成什么样
# ==========================================================================
# 现在：
#     def run(self, user_input: str) -> str:      # 一路跑到底，中间没法停
#         ...
#
# 改造后：
#     def step(state: RunState, deps) -> tuple[RunState, list[Event]]:
#         """推进一步，返回新状态和这一步产生的事件。纯粹靠入参，不依赖实例字段。"""
#
#     # 调用方（CLI / HTTP / 任务队列）自己决定怎么驱动：
#     while state.status not in ("done", "faulted", "suspended"):
#         state, events = step(state, deps)
#         store.append(events)          # 落盘
#         sink(events)                  # 推前端
#
#     # 挂起了就把 state 存库，HTTP 返回，等用户点确认再：
#     state = reduce(state, RunResumed(call_id=..., approved=True))
#     # 然后继续 while —— 换个进程、换台机器都行
#
# 关键差别：状态从「Agent 实例的私有字段」变成「一个能存库的值」。
# 函数返回后状态不消失，这就是可中断的全部秘密。


# ==========================================================================
# 演示
# ==========================================================================
def _demo() -> None:
    events: list[Event] = [
        UserAsked("2025年哪个大区销售额最高"),
        AssistantResponded(
            step=1, text="先看看有哪些表",
            tool_calls=(ToolCallState("c1", "list_tables", {}),),
            usage={"input_tokens": 1200, "output_tokens": 30},
            stop_reason="tool_use",
        ),
        ToolStarted("c1"),
        ToolFinished("c1", ok=True, output="共 4 张表：customers, orders, ...", elapsed_ms=12),
        AssistantResponded(
            step=2, text="我来查一下",
            tool_calls=(ToolCallState("c2", "run_sql", {"sql": "SELECT ..."}),),
            usage={"input_tokens": 1800, "output_tokens": 90},
            stop_reason="tool_use",
        ),
        # ★ 这里挂起等审批 —— 现在的实现根本走不到这个状态
        RunSuspended("c2", reason="run_sql 需要人工确认"),
        RunResumed("c2", approved=True),
        ToolStarted("c2"),
        ToolFinished("c2", ok=True, output="| region | gmv |\n| 华东 | 810万 |", elapsed_ms=13),
        AssistantResponded(
            step=3, text="华东最高，810 万。", tool_calls=(),
            usage={"input_tokens": 2400, "output_tokens": 150},
            stop_reason="end_turn",
        ),
        RunFinished("华东最高，810 万。"),
    ]

    print("=" * 74)
    print("逐个事件 fold，看状态怎么变：")
    print("=" * 74)
    state = RunState(run_id="run-1")
    for e in events:
        state = reduce(state, e)
        extra = f"  awaiting={state.awaiting_call_id}" if state.awaiting_call_id else ""
        print(f"{type(e).__name__:<22} → status={state.status:<10} "
              f"step={state.step} calls={len(state.calls)}{extra}")

    print()
    print("=" * 74)
    print("最终状态")
    print("=" * 74)
    print(f"status   : {state.status}")
    print(f"result   : {state.result}")
    print(f"usage    : {state.usage}")
    print(f"transcript ({len(state.transcript)} 条):")
    for line in state.transcript:
        print(f"    {line}")

    print()
    print("=" * 74)
    print("关键性质验证")
    print("=" * 74)
    # 1. 纯函数 → 重放必然得到同样结果（这是崩溃恢复能成立的原因）
    assert fold(events) == state
    print("✅ 重放同一串事件 → 得到完全相同的状态（可恢复）")

    # 2. 从中间任意一点恢复
    mid = fold(events[:6])          # 恰好停在 RunSuspended
    print(f"✅ 从第 6 个事件恢复 → status={mid.status}, "
          f"在等 {mid.awaiting_call_id} 的审批")
    resumed = reduce(mid, RunResumed("c2", approved=False, note="这条 SQL 太贵了"))
    denied = [c for c in resumed.calls if c.status == "denied"]
    print(f"✅ 换个决定（拒绝）→ calls[0].status={denied[0].status}, "
          f"output={denied[0].output!r}")

    # 3. 不跑模型、不连数据库就能测完整流程
    print("✅ 以上全程没调模型、没连数据库 —— 状态机可以单独测")


if __name__ == "__main__":
    _demo()
