"""Agent 主循环：请求模型 → 有工具调用就执行、把结果放回历史 → 直到模型不再调工具。

只依赖 core 里的三个抽象：LLMProvider、ToolRegistry、BaseContext。
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

from .context import BaseContext, Context, Entry, Measure, Prompt
from .errors import (
    CompactionFailed,
    ContextOverflow,
    ModelRefused,
    OutputTruncated,
    UnexpectedStopReason,
)
from .events import (
    AutoCompactionPaused,
    ContextEdited,
    ContextEditFailed,
    ContextOverflowed,
    Event,
    LLMResponded,
    StepLimitReached,
    ToolCallRepeated,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
    TurnResumed,
    noop_sink,
)
from .messages import LLMResponse, Message, ToolCall, Usage
from .provider import LLMProvider
from .tokens import ContextEstimate, estimate_context, estimate_overhead
from .tools import ToolRegistry

# 执行工具前的审批钩子：返回 (是否放行, 拒绝理由)
ApprovalHook = Callable[[ToolCall], "tuple[bool, str]"]

# 自动压缩连续失败几次就熔断（Claude Code 也是 3）
MAX_COMPACTION_FAILURES = 3

# 同一轮里同一个工具、同样的参数调到第几次，就在结果后面附一句提醒（软干预，不拦）。
# 第 2 次常常是正当的：旧结果被清理后，占位就是叫它重调
REPEAT_WARN_AT = 3
REPEAT_NOTE = ("\n\n[提醒：这一轮里你已经用完全相同的参数调用 {name} {n} 次了。结果没变的话再调也一样，"
               "换个思路，或者根据已经得到的结果直接回答。]")

# 步数用完时追加的提示（{n} = 步数）。默认保守：确定的照实说，没做完的说清楚做到哪一步，不拿没核实的数当结论。
# 想让模型尽量给出答案（比如评测），组装时换一段（prompts.WRAP_UP_BEST_GUESS）
WRAP_UP = ("[步数用完了（{n} 步），不能再调用工具。请根据上面已经得到的结果直接回答："
           "已经确定的结论照实给出；没做完的部分说明做到了哪一步、还差什么，不要把没核实过的数字当成结论。"
           "用户原来对回答格式的要求照样遵守。]")

# 工具执行到一半断了：不重做（副作用可能已经发生，比如沙箱里的变量改了一半），如实告诉模型（学 Claude Code / pi）
INTERRUPTED_RESULT = "[中断：这次调用没有执行完，结果未知（可能已经执行了一部分）。需要的话先确认状态再继续。]"
CONTINUE_NUDGE = "请从中断的地方继续完成上面的任务。"


@dataclass(frozen=True, slots=True)
class InterruptedTurn:
    """一轮没跑完时留下的进度（检查点）。Agent.resume() 把它接回历史，从下一步接着跑。

    正式历史照样回滚、只放完整的回合；进度另外放在这里。entries 的形状一定合法：
    没执行完的工具调用补了「结果未知」，接回去不会出现没配对的 tool_call。
    """

    entries: tuple[Entry, ...]     # 这一轮已有的条目，第一条是提问
    steps: int                     # 已经完成了几步（模型回复了几次）
    reason: str = ""               # 为什么断的；每一步存的检查点是空的

    @property
    def question(self) -> str:
        first = self.entries[0]
        return first.content if isinstance(first, Message) else ""


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """一步结束时交给 finish_turn_hook 判断的材料。"""

    step: int
    response: LLMResponse
    # 模型**请求**了工具（被审批拒掉的也算），不代表工具跑成功了
    requested_tools: bool


@dataclass(frozen=True, slots=True)
class TurnDecision:
    action: Literal["end", "continue"]
    nudge: str = ""          # continue 且这一步没调工具时，用它推动模型继续

    @staticmethod
    def end() -> "TurnDecision":
        return TurnDecision("end")

    @staticmethod
    def keep_going(nudge: str = "") -> "TurnDecision":
        return TurnDecision("continue", nudge)


# 决定「这一步之后收工还是继续」。不装时：调了工具就继续，没调就结束（pi 的 finishTurn）
FinishTurnHook = Callable[[TurnOutcome], TurnDecision]

# 每走完一步、以及一轮断掉时，交出当前进度（存盘用：进程被杀了也能接着跑）
CheckpointHook = Callable[[InterruptedTurn], None]


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str,
        context: BaseContext | None = None,
        max_steps: int = 12,
        approval_hook: ApprovalHook | None = None,
        finish_turn_hook: FinishTurnHook | None = None,
        on_event: Callable[[Event], None] = noop_sink,
        session_context: Callable[[], str] | None = None,
        wrap_up_prompt: str = WRAP_UP,
        checkpoint_hook: CheckpointHook | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.context = context if context is not None else Context()
        self.max_steps = max_steps
        self.wrap_up_prompt = wrap_up_prompt
        self.approval_hook = approval_hook
        self.finish_turn_hook = finish_turn_hook
        self.checkpoint_hook = checkpoint_hook
        self.on_event = on_event
        # 会话开始时拼到系统提示词末尾的内容（库概览，以后的记忆索引）。只算一次
        self.session_context = session_context
        self._system: str | None = None
        # 本次会话花了多少 token。钱花了就是花了：失败的一轮回滚历史，但不回滚这里
        self.session_usage = Usage()
        # 自动压缩连续失败了几次（熔断用）。和 session_usage 一样不跟着回滚
        self._compaction_failures = 0
        # 这一轮里每种调用（工具名 + 参数）调了几次，发现原地打转用
        self._calls_this_turn: Counter[str] = Counter()
        # 上一轮没跑完留下的进度，resume() 接着跑；问新问题、/reset 就作废
        self.interrupted: InterruptedTurn | None = None
        self._turn_snapshot: object = None
        self._steps_done = 0

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """跑一轮对话，返回最终回答。上一轮没跑完的进度作废。"""
        self.interrupted = None
        return self._transaction(lambda: self._run_turn(user_input))

    def resume(self, message: str = "") -> str:
        """接着跑上一轮没跑完的（self.interrupted），从下一步开始，步数用剩下的。

        message：用户接着说的话（「别按月拆了，直接算全年」），接在进度后面；不给就直接接着跑。
        """
        turn = self.interrupted
        if turn is None:
            raise ValueError("没有没跑完的回合可以接着跑")
        self.interrupted = None
        return self._transaction(lambda: self._resume_turn(turn, message))

    def _transaction(self, body: Callable[[], str]) -> str:
        """事务：要么完整完成，要么历史回到进来之前。半截的一轮会毒化历史——
        tool_call 没有结果，之后每次请求都 400；只有提问没有回答，下一次提问会和它粘在一起。
        回滚之前把进度存进 self.interrupted（检查点），resume() 能接着跑。
        捕获 BaseException，连 Ctrl-C 一起兜住：Ctrl-C 就是「暂停」。
        """
        self._turn_snapshot = self.context.snapshot()
        self._steps_done = 0
        try:
            return body()
        except BaseException as exc:
            progress = self._progress(type(exc).__name__ + (f": {exc}" if str(exc) else ""))
            self.interrupted = progress if progress.entries else None
            if self.interrupted is not None:
                self._save(self.interrupted)
            self.context.restore(self._turn_snapshot)
            raise

    def _progress(self, reason: str = "") -> InterruptedTurn:
        """这一轮到现在的进度，形状修成合法的：可以原样接回历史。"""
        entries = self.context.since(self._turn_snapshot)
        last = next((e for e in reversed(entries) if isinstance(e, Message) and e.role == "assistant"), None)
        if last is not None and last.tool_calls:
            answered = {e.tool_call_id for e in entries if isinstance(e, Message) and e.role == "tool"}
            entries += [Message.tool_result(c.id, INTERRUPTED_RESULT, is_error=True)
                        for c in last.tool_calls if c.id not in answered]
        # 断在收尾那次请求上：收尾提示不留，接着跑时会重新收尾、重新加
        wrap_up = self.wrap_up_prompt.format(n=self.max_steps)
        if entries and isinstance(entries[-1], Message) and entries[-1].meta.synthetic \
                and entries[-1].content == wrap_up:
            entries.pop()
        return InterruptedTurn(tuple(entries), self._steps_done, reason)

    def _checkpoint(self) -> None:
        if self.checkpoint_hook is not None:
            self._save(self._progress())

    def _save(self, turn: InterruptedTurn) -> None:
        # 存盘是锦上添花：写失败不能把跑了几十步的一轮搞没，也不能盖掉原来的异常
        if self.checkpoint_hook is not None:
            with contextlib.suppress(Exception):
                self.checkpoint_hook(turn)

    # ------------------------------------------------------------------
    def _run_turn(self, user_input: str) -> str:
        self.context.add(Message.user(user_input))
        self._calls_this_turn.clear()
        self._checkpoint()
        return self._loop(1)

    def _resume_turn(self, turn: InterruptedTurn, message: str) -> str:
        for entry in turn.entries:
            self.context.add(entry)
        self._steps_done = turn.steps
        self._calls_this_turn = Counter(_call_key(c) for e in turn.entries if isinstance(e, Message)
                                        for c in e.tool_calls)
        last = turn.entries[-1]
        if not message and turn.steps < self.max_steps and isinstance(last, Message) and last.role == "assistant":
            message = CONTINUE_NUDGE          # 历史不能以 assistant 结尾再请求
        if message:
            # 不是新问题，是这一轮里的补充：标 synthetic，不算回合开头
            self.context.add(Message.user(message).with_meta(synthetic=True))
        self.on_event(TurnResumed(turn.steps, message))
        return self._loop(turn.steps + 1)

    def _loop(self, start: int) -> str:
        for step in range(start, self.max_steps + 1):
            response = self._request(step)

            # 先查 stop_reason 再进历史：截断、被拒的回复不可信，不能让它进去
            self._check_stop_reason(response)

            self.context.add(response.to_message())
            self._steps_done = step

            requested_tools = bool(response.tool_calls)
            for call in response.tool_calls:
                self._execute(call)

            decision = self._decide_next(TurnOutcome(step, response, requested_tools))
            if decision.action == "end":
                return response.text

            if not requested_tools and step < self.max_steps:
                # 历史以 assistant 结尾不能直接再请求（Anthropic 会当成 prefill，新模型直接 400），
                # 补一条 nudge。标成 synthetic：它不是真人的新问题，不算新回合的开头。
                # 最后一步不补：接下来的收尾提示就是这一步的「继续」，两条说的是一回事
                nudge = decision.nudge or "请继续完成上面的任务。"
                self.context.add(Message.user(nudge).with_meta(synthetic=True))
                self.on_event(TurnContinued(step=step, nudge=nudge))

            self._checkpoint()

        return self._wrap_up()

    def _request(self, step: int) -> LLMResponse:
        """请求一次模型：先整理上下文，再请求、记账、发事件。stop_reason 由调用方判断。"""
        system = self._render_system_prompt()
        tools = self.tools.schemas()

        # 每一步都整理：一轮里连调十几次工具，上下文在一轮之内就可能涨过阈值
        self._maintain(system, tools)

        response = self._chat(step, system, tools)
        self.session_usage += response.usage      # 被截断的回复同样收费，先记账
        self.on_event(LLMResponded(
            step=step,
            text=response.text,
            tool_calls=[c.name for c in response.tool_calls],
            usage=response.usage,
            context_window=self.llm.context_window,
        ))
        return response

    def _wrap_up(self) -> str:
        """步数用完：不再给工具，让模型根据已经得到的结果回答（学 smolagents 的 final answer）。

        以前直接返回「没做完」，前面几十步的结果全浪费了。收尾的回答不再过 finish_turn_hook：
        步数已经用完，钩子说「没做完」也没法再继续，有文字就当回答。
        收尾失败（又去调工具、被截断、API 报错）才用兜底那句话，原因放进 StepLimitReached。
        兜底回答也要进历史：让模型下一轮知道上一轮卡住了，也保证历史以 assistant 结尾。
        工具表照样发（去掉工具会断缓存，有的厂商历史里有工具调用时还要求带着工具表），只在提示里说别再调。
        """
        self.context.add(Message.user(self.wrap_up_prompt.format(n=self.max_steps)).with_meta(synthetic=True))
        try:
            response = self._request(self.max_steps + 1)
            failure = ("又去调了工具" if response.tool_calls
                       else f"stop_reason={response.stop_reason}" if not response.finished_normally
                       else "回答是空的" if not response.text else "")
        except Exception as exc:  # noqa: BLE001 —— 收尾是尽力而为，失败就用兜底，不能把这一轮搞没
            failure = f"{type(exc).__name__}: {exc}"
        self.on_event(StepLimitReached(self.max_steps, wrapped_up=not failure, failure=failure))
        if not failure:
            self.context.add(response.to_message())
            return response.text
        fallback = (
            f"已达到最大步数 {self.max_steps} 仍未得出结论。"
            "可以把问题拆小一点，或者调大 max_steps。"
        )
        self.context.add(Message.assistant(fallback).with_meta(synthetic=True))
        return fallback

    # ------------------------------------------------------------------
    def _chat(self, step: int, system: str, tools: list) -> LLMResponse:
        """请求模型。API 报上下文超长时，不看阈值强制整理一次再重试，只试一次。"""
        try:
            return self.llm.chat(messages=self.context.render(), tools=tools, system=system)
        except ContextOverflow:
            self.on_event(ContextOverflowed(step))
            if not self._maintain(system, tools, force=True):
                raise
            return self.llm.chat(messages=self.context.render(), tools=tools, system=system)

    def _maintain(self, system: str, tools: list, *, force: bool = False) -> list[Event]:
        """让上下文整理一次。system / tools 用来量大小，写摘要时原样带上以命中缓存。

        熔断：自动压缩连续失败 MAX_COMPACTION_FAILURES 次，之后只清理不压缩（强制的照常压），
        免得每一步都白花一次写摘要的钱。调模型的整理成功一次就恢复。
        """
        paused = self._compaction_failures >= MAX_COMPACTION_FAILURES
        try:
            events = self.context.maintain(self._measure(system, tools), force=force,
                                           prompt=Prompt(system, tuple(tools)),
                                           model_calls=force or not paused)
        except CompactionFailed as exc:            # 只有强制整理会抛出来
            self.session_usage += exc.usage
            raise
        for event in events:
            self.session_usage += event.usage      # 写摘要也是一次收费的调用，失败了也收
            self.on_event(event)
            if isinstance(event, ContextEditFailed):
                self._compaction_failures += 1
                if self._compaction_failures == MAX_COMPACTION_FAILURES:
                    self.on_event(AutoCompactionPaused(self._compaction_failures))
            elif isinstance(event, ContextEdited) and event.used_model:
                self._compaction_failures = 0
        return events

    def _measure(self, system: str, tools: list) -> Measure:
        overhead = estimate_overhead(system, tools)
        return lambda msgs: estimate_context(msgs, overhead=overhead).tokens

    # ------------------------------------------------------------------
    def _check_stop_reason(self, response: LLMResponse) -> None:
        """被截断时同样没有 tool_calls，不查 stop_reason 就会把半句话当答案返回。"""
        if response.truncated:
            raise OutputTruncated(
                f"模型输出被截断（stop_reason={response.stop_reason}），这不是「完成」。"
                f"已生成 {response.usage.output} 个 token。"
                "解决办法：调大 .env 里的 MAX_TOKENS，或让它分步输出。"
            )

        if response.refused:
            raise ModelRefused(
                f"模型拒绝了这个请求（stop_reason={response.stop_reason}）。"
            )

        if not response.finished_normally:
            # 没见过的值。宁可炸掉，也不要静默当成「完成」。
            raise UnexpectedStopReason(
                f"遇到未处理的 stop_reason='{response.stop_reason}'。"
                "如果这是个正常的结束原因，请把它加进 core/messages.py 的 "
                "NORMAL_STOP_REASONS。"
            )

    def _decide_next(self, outcome: TurnOutcome) -> TurnDecision:
        if self.finish_turn_hook is None:
            return TurnDecision.keep_going() if outcome.requested_tools else TurnDecision.end()
        return self.finish_turn_hook(outcome)

    def _execute(self, call: ToolCall) -> None:
        self.on_event(ToolStarted(name=call.name, arguments=call.arguments))

        if self.approval_hook is not None:
            allowed, reason = self.approval_hook(call)
            if not allowed:
                # 被拒绝也要给模型一条结果，否则 tool_call 没有配对，它也不知道发生了什么
                self.context.add(Message.tool_result(call.id, f"用户拒绝执行：{reason}", is_error=True))
                self.on_event(ToolDenied(name=call.name, reason=reason))
                return

        started = time.perf_counter()
        result = self.tools.invoke(call.name, call.arguments)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        key = _call_key(call)
        self._calls_this_turn[key] += 1
        count = self._calls_this_turn[key]
        content = result.content
        if count >= REPEAT_WARN_AT:
            content += REPEAT_NOTE.format(name=call.name, n=count)

        self.context.add(Message.tool_result(call.id, content, is_error=result.is_error,
                                             summary=result.summary, images=result.images))
        self.on_event(ToolFinished(
            name=call.name, content=content, is_error=result.is_error,
            elapsed_ms=elapsed_ms, details=result.details,
        ))
        if count >= REPEAT_WARN_AT:
            self.on_event(ToolCallRepeated(call.name, count))

    def _render_system_prompt(self) -> str:
        """固定人设 + 会话上下文。一个会话只算一次：系统提示词一变，后面整段缓存都废了。"""
        if self._system is None:
            extra = self.session_context() if self.session_context else ""
            self._system = f"{self.system_prompt}\n\n{extra}" if extra else self.system_prompt
        return self._system

    def context_usage(self) -> ContextEstimate:
        """如果现在发下一次请求，输入大概有多大（按 render() 之后真正会发的那份估）。"""
        overhead = estimate_overhead(self._render_system_prompt(), self.tools.schemas())
        return estimate_context(self.context.render(), overhead=overhead)

    def compact(self) -> list[Event]:
        """手动整理（/compact）：不看阈值。和 run() 一样是事务，失败时历史不变。"""
        snapshot = self.context.snapshot()
        try:
            return self._maintain(self._render_system_prompt(), self.tools.schemas(), force=True)
        except BaseException:
            self.context.restore(snapshot)
            raise

    def reset(self) -> None:
        self.context.clear()
        self._system = None
        self._compaction_failures = 0
        self.interrupted = None


def _call_key(call: ToolCall) -> str:
    return f"{call.name} {json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str)}"
