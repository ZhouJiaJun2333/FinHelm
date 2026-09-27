"""评测里出错了从检查点接着跑；故障注入能复现。"""

from __future__ import annotations

import pytest

from data_agent.core.messages import LLMResponse, ToolCall
from evals.faults import FlakyProvider, InjectedFault
from evals.runner import Trial, answer, digest

from fakes import ScriptedProvider, make_agent


def call(n: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(f"c{n}", "echo", {"text": str(n)})], stop_reason="tool_use")


FINAL = LLMResponse(text="答案是 42", stop_reason="end_turn")


def test_出错了接着跑_有进展就重新计数():
    # 每走一步都断一次：总共断 3 次，但每次都有进展，不算连续失败
    agent, events = make_agent([call(1), RuntimeError("a"), call(2), RuntimeError("b"), call(3),
                                RuntimeError("c"), FINAL])
    assert answer(agent, "算一下", waits=(0, 0)) == "答案是 42"
    t = Trial("x", 1)
    digest(t, events)
    assert t.resumes == 3 and t.steps == 4


def test_同一步上连续失败_次数用完就放弃_抛最后那个异常():
    agent, _ = make_agent([call(1), RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), FINAL])
    with pytest.raises(RuntimeError, match="c"):
        answer(agent, "算一下", waits=(0, 0))
    assert agent.llm.calls == 4, "第 1 步 + 第 2 步上试了 3 次"


def test_关掉接着跑_就是以前的整题作废():
    agent, _ = make_agent([call(1), RuntimeError("a"), FINAL])
    with pytest.raises(RuntimeError):
        answer(agent, "算一下", resume=False)


def test_注入错误_同一题同一次每回都在同一步():
    def trace(seed: str) -> list[bool]:
        llm = FlakyProvider(ScriptedProvider([FINAL]), 0.3, seed)
        out = []
        for _ in range(30):
            try:
                llm.chat([])
                out.append(False)
            except InjectedFault:
                out.append(True)
        return out

    assert trace("dab-1-1") == trace("dab-1-1")
    assert trace("dab-1-1") != trace("dab-1-2")
    assert 3 < sum(trace("dab-1-1")) < 20


def test_注入错误加接着跑_答案和不注入一样():
    llm = FlakyProvider(ScriptedProvider([call(1), call(2), call(3), FINAL]), 0.4, "seed")
    agent, events = make_agent()
    agent.llm = llm
    assert answer(agent, "算一下", waits=(0, 0)) == "答案是 42"
    assert llm.injected > 0 and llm.inner.calls == 4, "注入的错误不花钱、真模型只被调了 4 次"
