"""交付前复核（VERIFY）：主 Agent 第一次打算收工时推它一句，先派 verifier 独立重算一遍，再给最终回答。

做成机制、不靠模型自觉：评测才能拿到「复核前」和「复核后」两份答案，数出复核改对、改错了多少。
每一轮只推一次；离步数上限不到两步时不推（复核一步、再回答一步，不够就直接交）。
"""

from __future__ import annotations

from ..core.agent import FinishTurnHook, TurnDecision, TurnOutcome

VERIFY_NUDGE = ("[交付前复核] 先不要给最终回答。用 delegate 派一个 verifier：任务说明里写上用户的原问题（原文照抄），"
                "你答案里的关键数字、结论和用的口径；不要写你的 SQL 或代码，让它独立算。复核结果回来后，"
                "一致就给出最终回答；不一致就查明是谁错了、改正后再回答。复核也可能错，不要盲从。")


def verify_before_finish(max_steps: int, then: FinishTurnHook | None = None) -> FinishTurnHook:
    """then：原来的收工判断（没有就用 Agent 的默认：调了工具继续，没调结束）。"""
    last_step = 0
    nudged = False

    def hook(outcome: TurnOutcome) -> TurnDecision:
        nonlocal last_step, nudged
        if outcome.step <= last_step:            # 步数变小了 = 新的一轮
            nudged = False
        last_step = outcome.step
        if then is not None:
            decision = then(outcome)
        else:
            decision = TurnDecision.keep_going() if outcome.requested_tools else TurnDecision.end()
        if decision.action == "end" and not nudged and outcome.step <= max_steps - 2:
            nudged = True
            return TurnDecision.keep_going(VERIFY_NUDGE)
        return decision

    return hook
