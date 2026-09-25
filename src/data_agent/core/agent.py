"""Agent 主循环：请求模型 → 有工具调用就执行、把结果放回历史 → 直到模型不再调工具。

只依赖 core 里的三个抽象：LLMProvider、ToolRegistry、BaseContext。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from .context import BaseContext, Context, Measure, Prompt
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
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
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
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.context = context if context is not None else Context()
        self.max_steps = max_steps
        self.approval_hook = approval_hook
        self.finish_turn_hook = finish_turn_hook
        self.on_event = on_event
        # 会话开始时拼到系统提示词末尾的内容（库概览，以后的记忆索引）。只算一次
        self.session_context = session_context
        self._system: str | None = None
        # 本次会话花了多少 token。钱花了就是花了：失败的一轮回滚历史，但不回滚这里
        self.session_usage = Usage()
        # 自动压缩连续失败了几次（熔断用）。和 session_usage 一样不跟着回滚
        self._compaction_failures = 0

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """跑一轮对话，返回最终回答。

        事务：要么完整完成，要么历史回到进来之前。半截的一轮会毒化历史——
        tool_call 没有结果，之后每次请求都 400；只有提问没有回答，下一次提问会和它粘在一起。
        用 finally 而不是 except，连 Ctrl-C 一起兜住。
        """
        snapshot = self.context.snapshot()
        completed = False
        try:
            answer = self._run_turn(user_input)
            completed = True
            return answer
        finally:
            if not completed:
                self.context.restore(snapshot)

    # ------------------------------------------------------------------
    def _run_turn(self, user_input: str) -> str:
        self.context.add(Message.user(user_input))

        for step in range(1, self.max_steps + 1):
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

            # 先查 stop_reason 再进历史：截断、被拒的回复不可信，不能让它进去
            self._check_stop_reason(response)

            self.context.add(response.to_message())

            requested_tools = bool(response.tool_calls)
            for call in response.tool_calls:
                self._execute(call)

            decision = self._decide_next(TurnOutcome(step, response, requested_tools))
            if decision.action == "end":
                return response.text

            if not requested_tools:
                # 历史以 assistant 结尾不能直接再请求（Anthropic 会当成 prefill，新模型直接 400），
                # 补一条 nudge。标成 synthetic：它不是真人的新问题，不算新回合的开头
                nudge = decision.nudge or "请继续完成上面的任务。"
                self.context.add(Message.user(nudge).with_meta(synthetic=True))
                self.on_event(TurnContinued(step=step, nudge=nudge))

        # 步数耗尽：兜底回答也要进历史，让模型下一轮知道上一轮卡住了，
        # 也保证历史以 assistant 结尾（正常返回，事务不会替我们收尾）
        self.on_event(StepLimitReached(self.max_steps))
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

        self.context.add(Message.tool_result(call.id, result.content, is_error=result.is_error,
                                             summary=result.summary, images=result.images))
        self.on_event(ToolFinished(
            name=call.name, content=result.content, is_error=result.is_error,
            elapsed_ms=elapsed_ms, details=result.details,
        ))

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
