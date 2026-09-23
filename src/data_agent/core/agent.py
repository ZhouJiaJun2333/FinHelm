"""Agent 主循环 —— 整个项目的心脏。

一个 Agent 的本质就是这个循环：

    while True:
        response = 模型(历史, 工具表)
        if 没有工具调用:
            结束，返回答案
        执行所有工具调用，把结果塞回历史

就这么简单。剩下的复杂度全在「历史怎么管」「工具怎么写」「提示词怎么写」上，
所以那三块各自是独立模块。

这个文件刻意不 import 任何具体的工具、厂商、数据库。它只依赖三个抽象：
    LLMProvider（llm/base.py）
    ToolRegistry（tools/registry.py）
    BaseContext （core/context.py）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from ..llm.base import LLMProvider
from ..tools.registry import ToolRegistry
from .context import BaseContext, FullContext
from .errors import ModelRefused, OutputTruncated, UnexpectedStopReason
from .events import (
    Event,
    LLMResponded,
    StepLimitReached,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
    noop_sink,
)
from .messages import LLMResponse, Message, ToolCall

# 执行工具前的审批钩子：返回 (是否放行, 拒绝理由)
ApprovalHook = Callable[[ToolCall], "tuple[bool, str]"]


# ---------------------------------------------------------------- stop_reason
# 「这一轮为什么停下来」的分类。不同厂商的叫法不一样，在这里统一识别。
#
# ⚠️ 这是新手最容易漏的一件事：
#    判断「模型完成了没有」不能只看有没有 tool_calls。被 max_tokens 截断时
#    同样没有 tool_calls，但那是**话说到一半被砍了**，不是答完了。
#    不查 stop_reason 的话，你会把半句话当成最终答案返回，而且毫无察觉。
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})
REFUSAL_STOP_REASONS = frozenset({"refusal", "content_filter"})
NORMAL_STOP_REASONS = frozenset({
    "end_turn", "stop", "tool_use", "tool_calls", "function_call", "",
})


# ------------------------------------------------------------------ finishTurn
@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """一轮结束时交给钩子判断的材料。"""

    step: int
    response: LLMResponse

    # ⚠️ 语义是「模型**请求**了工具」，不是「工具真的跑成功了」。
    #    它在执行之前就赋值，所以被审批钩子拒掉的调用也算 True。
    #    想知道工具到底跑没跑、结果如何，订阅 ToolFinished / ToolDenied 事件。
    requested_tools: bool


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """钩子的判断结果：这一轮之后该收工还是继续。"""

    action: Literal["end", "continue"]
    nudge: str = ""          # continue 且本轮没跑工具时，用它推动模型继续

    @staticmethod
    def end() -> "TurnDecision":
        return TurnDecision("end")

    @staticmethod
    def keep_going(nudge: str = "") -> "TurnDecision":
        return TurnDecision("continue", nudge)


# 决定「这一轮之后要不要继续」的钩子。
#
# 不装钩子时用默认规则：有工具调用就继续，没有就结束。
# 装上钩子，你就能表达「模型以为自己答完了，但我检查后觉得没完成，让它接着干」——
# 这正是把「结束条件」从主循环里搬出来的意义（pi 的 finishTurn 是同一个设计）。
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
        dynamic_context: Callable[[], str] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.context = context or FullContext()
        self.max_steps = max_steps
        self.approval_hook = approval_hook
        self.finish_turn_hook = finish_turn_hook
        self.on_event = on_event
        # 每轮动态拼到系统提示词末尾的内容（数据库概览、记忆、RAG 检索结果…）
        self.dynamic_context = dynamic_context

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """跑一轮完整对话（内部可能调用多次工具），返回最终回答。

        **事务语义：要么完整完成，要么历史回到进来之前的样子。**

        为什么必须这样：一轮失败时（截断、模型拒绝、网络错、Ctrl-C），历史里会
        留下半截状态，而这些状态会让**之后每一轮**都失败：

            提问后失败      → [user]，用户再问一次 → [user, user]
                              Anthropic 要求角色交替，直接 400
            工具调用后失败  → assistant 有 tool_calls 却没有对应结果
                              两家都 400

        两者是同一个死亡模式的两个入口：**一次失败升级成整个会话报废**，
        用户只能 /reset 清空所有上下文。

        ⚠️ 特别注意第一种：OpenAI 兼容接口（DeepSeek 等）容忍连续 user 消息，
           所以这个 bug 在本地怎么测都不出来，换到 Anthropic 才炸。
           抽象层的意义之一就是不让这种差异漏到上层 —— 所以这里统一按
           **最严格**的那家的约束来保证。
        """
        snapshot = self.context.snapshot()
        completed = False
        try:
            answer = self._run_turn(user_input)
            completed = True
            return answer
        finally:
            # 用 finally 而不是 except，是为了连 KeyboardInterrupt 一起兜住 ——
            # 用户 Ctrl-C 打断的半截回合同样会毒化历史。
            if not completed:
                self.context.restore(snapshot)

    # ------------------------------------------------------------------
    def _run_turn(self, user_input: str) -> str:
        """run() 的实际循环体。失败时由 run() 负责回滚，这里只管往前跑。"""
        self.context.add(Message.user(user_input))

        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(
                messages=self.context.render(),
                tools=self.tools.schemas(),
                system=self._render_system_prompt(),
            )
            self.on_event(LLMResponded(
                step=step,
                text=response.text,
                tool_calls=[c.name for c in response.tool_calls],
                usage=response.usage,
            ))

            # 先分诊，**再**决定要不要写进历史 —— 顺序很重要。
            #
            # 被截断/被拒绝的回复不可信，绝不能留在历史里：
            #   · 半截话会被模型当成自己已经说过的结论，下一轮接着往下编
            #   · 更致命：截断发生在工具调用中途时，历史里会留下一个
            #     没有结果的 tool_call，下一次请求直接 400 —
            #         An assistant message with 'tool_calls' must be followed
            #         by tool messages responding to each 'tool_call_id'
            #     而且这个错是**永久**的：坏消息一直躺在历史里，此后每一轮
            #     都报同样的错，用户只能 /reset 清空整个对话。
            #     一轮失败变成整个会话报废。
            #
            # 「不让它进去」比「进去了再删」干净：没有回滚逻辑，也不会有中间状态。
            self._check_stop_reason(response)

            self.context.add(response.to_message())

            # 注意是「请求了工具」，不是「工具跑成功了」—— 见 TurnOutcome 的注释。
            # 这里必须在 _execute 之前取值：循环跑完后 response 不会变，
            # 但放在后面容易让人误以为它反映的是执行结果。
            requested_tools = bool(response.tool_calls)
            for call in response.tool_calls:
                self._execute(call)

            decision = self._decide_next(TurnOutcome(step, response, requested_tools))
            if decision.action == "end":
                return response.text

            if not requested_tools:
                # 本轮没有工具结果，历史以 assistant 结尾。
                # 必须补一条 user 消息，否则下一轮请求的角色不交替，API 会报错。
                nudge = decision.nudge or "请继续完成上面的任务。"
                self.context.add(Message.user(nudge))
                self.on_event(TurnContinued(step=step, nudge=nudge))

        self.on_event(StepLimitReached(self.max_steps))
        return (
            f"已达到最大步数 {self.max_steps} 仍未得出结论。"
            "可以把问题拆小一点，或者调大 max_steps。"
        )

    # ------------------------------------------------------------------
    def _check_stop_reason(self, response: LLMResponse) -> None:
        """看模型「为什么停下来」，把不可信的那几种拦掉。

        为什么不能只看 tool_calls：被 max_tokens 截断时同样没有 tool_calls，
        但那是话说到一半被砍了。静默地把半句话当答案返回，是这类系统里
        最难排查的一种 bug —— 不报错、不告警，只是答案莫名其妙地不完整。
        """
        reason = (response.stop_reason or "").lower()

        if reason in TRUNCATED_STOP_REASONS:
            raise OutputTruncated(
                f"模型输出被截断（stop_reason={response.stop_reason}），这不是「完成」。"
                f"已生成 {response.usage.get('output_tokens', '?')} 个 token。"
                "解决办法：调大 .env 里的 MAX_TOKENS，或让它分步输出。"
            )

        if reason in REFUSAL_STOP_REASONS:
            raise ModelRefused(
                f"模型拒绝了这个请求（stop_reason={response.stop_reason}）。"
            )

        if reason not in NORMAL_STOP_REASONS:
            # 没见过的值。宁可炸掉，也不要静默当成「完成」。
            raise UnexpectedStopReason(
                f"遇到未处理的 stop_reason='{response.stop_reason}'。"
                "如果这是个正常的结束原因，请把它加进 core/agent.py 的 "
                "NORMAL_STOP_REASONS。"
            )

    # ------------------------------------------------------------------
    def _decide_next(self, outcome: TurnOutcome) -> TurnDecision:
        """这一轮之后，收工还是继续？

        默认规则（和没加钩子之前的行为完全一致）：
            请求了工具 → 继续（要把结果喂回去让它接着想）
            没请求工具 → 结束（模型认为任务完成了）

        注意「请求了工具」包含被审批钩子拒掉的情况 —— 那时历史里是一条
        「已拒绝」的 tool_result，同样需要再跑一轮让模型看到并改道。

        装上 finish_turn_hook 就能覆盖它。典型用途：
            · 模型说完了但你检查发现少了关键内容 → keep_going("还缺占比，补上")
            · 达到某个业务条件就强制收工       → end()
            · 输出格式不合规就打回重做
        """
        if self.finish_turn_hook is None:
            return TurnDecision.keep_going() if outcome.requested_tools else TurnDecision.end()
        return self.finish_turn_hook(outcome)

    # ------------------------------------------------------------------
    def _execute(self, call: ToolCall) -> None:
        self.on_event(ToolStarted(name=call.name, arguments=call.arguments))

        if self.approval_hook is not None:
            allowed, reason = self.approval_hook(call)
            if not allowed:
                # 被拒绝也要给模型一条结果，否则它不知道发生了什么，会一直重试
                self.context.add(Message.tool_result(call.id, f"用户拒绝执行：{reason}"))
                self.on_event(ToolDenied(name=call.name, reason=reason))
                return

        started = time.perf_counter()
        result = self.tools.invoke(call.name, call.arguments)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        self.context.add(Message.tool_result(call.id, result.content))
        self.on_event(ToolFinished(
            name=call.name, ok=result.ok, content=result.content, elapsed_ms=elapsed_ms,
        ))

    def _render_system_prompt(self) -> str:
        """系统提示词 = 固定人设 + 动态上下文。

        动态内容放在**末尾**：前缀保持稳定才能吃到 prompt 缓存。
        以后接记忆 / RAG，检索到的内容就从 dynamic_context 拼进来。
        """
        if self.dynamic_context is None:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{self.dynamic_context()}"

    def reset(self) -> None:
        self.context.clear()
