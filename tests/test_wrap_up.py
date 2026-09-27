"""步数用完时收尾：不再给工具，让模型根据已经得到的结果回答。收尾失败才用兜底那句话。"""

from __future__ import annotations

from data_agent.core.agent import WRAP_UP
from data_agent.core.events import LLMResponded, StepLimitReached
from data_agent.core.messages import LLMResponse, Message, ToolCall
from data_agent.llm.anthropic_provider import AnthropicProvider
from evals.cases import Case
from evals.report import render, summarize
from evals.runner import Trial, digest, grade

from fakes import ScriptedProvider, make_agent

CALL = LLMResponse(text="", tool_calls=[ToolCall("c", "echo", {"text": "x"})], stop_reason="tool_use")


def limit_event(events) -> StepLimitReached:
    return next(e for e in events if isinstance(e, StepLimitReached))


def test_收尾成功_返回模型的回答():
    agent, events = make_agent([CALL, CALL, LLMResponse(text="算到一半：大约 42。最终答案：42", stop_reason="end_turn")],
                               max_steps=2)
    assert agent.run("算一下") == "算到一半：大约 42。最终答案：42"
    assert limit_event(events).wrapped_up and limit_event(events).failure == ""
    # 收尾那次照样带着工具表（缓存、厂商要求），最后一条是收尾提示
    assert agent.llm.tools_seen[-1] == agent.llm.tools_seen[0]
    assert agent.llm.seen[-1][-1].content == WRAP_UP.format(n=2)
    history = agent.context.render()
    assert [m.role for m in history[-3:]] == ["tool", "user", "assistant"]
    assert history[-2].meta.synthetic and not history[-1].meta.synthetic
    assert sum(isinstance(e, LLMResponded) for e in events) == 3


def test_收尾提示可以换():
    agent, _ = make_agent([CALL, LLMResponse(text="好", stop_reason="end_turn")], max_steps=1,
                          wrap_up_prompt="[{n} 步用完了，给个答案]")
    agent.run("算一下")
    assert agent.llm.seen[-1][-1].content == "[1 步用完了，给个答案]"


def test_组装时按配置选收尾提示():
    from data_agent.app import build_application
    from data_agent.prompts import WRAP_UP_BEST_GUESS
    from data_agent.settings import Settings

    for style, prompt in (("report", WRAP_UP), ("best_guess", WRAP_UP_BEST_GUESS)):
        app = build_application(Settings(provider="openai", openai_api_key="x", wrap_up=style),
                                llm=ScriptedProvider())
        assert app.agent.wrap_up_prompt == prompt


def test_收尾时又去调工具_用兜底_不留没配对的调用():
    agent, events = make_agent([CALL], max_steps=2)          # 剧本永远调工具
    answer = agent.run("算一下")
    assert "最大步数 2" in answer
    assert not limit_event(events).wrapped_up and limit_event(events).failure == "又去调了工具"
    # 事件如实记下收尾那次调了工具
    assert [e.tool_calls for e in events if isinstance(e, LLMResponded)][-1] == ["echo"]
    history = agent.context.render()
    assert history[-1].role == "assistant" and not history[-1].tool_calls
    calls = {c.id for m in history for c in m.tool_calls}
    assert all(m.tool_call_id in calls for m in history if m.role == "tool")


def test_收尾时API报错或被截断_用兜底_记下原因_这一轮不回滚():
    for failure, reason in ((RuntimeError("503"), "RuntimeError: 503"),
                            (LLMResponse(text="写到一半", stop_reason="max_tokens"), "stop_reason=max_tokens")):
        agent, events = make_agent([CALL, CALL, failure], max_steps=2)
        assert "最大步数 2" in agent.run("算一下")
        assert not limit_event(events).wrapped_up and limit_event(events).failure == reason
        assert agent.context.render()[0].content == "算一下", "前两步的历史还在"


# ================================================================ Anthropic：连续的 user 并成一条
def test_anthropic_工具结果后面的提示并进同一条user消息():
    agent, _ = make_agent([CALL, CALL, LLMResponse(text="好", stop_reason="end_turn")], max_steps=2)
    agent.run("算一下")
    out = AnthropicProvider.convert_messages(agent.context.render())
    roles = [m["role"] for m in out]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    last_user = out[-2]["content"]
    assert last_user[0]["type"] == "tool_result" and last_user[-1] == {"type": "text", "text": WRAP_UP.format(n=2)}


def test_anthropic_只并工具结果后面的user_两条文字user照样分开():
    """两条文字 user 挨着是历史里有残留，要让角色交替的检查抓得到（见 test_provider_conversion）。"""
    out = AnthropicProvider.convert_messages([Message.user("a"), Message.user("b")])
    assert out == [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]


# ================================================================ 评测
def test_评测_收尾的回答照常判分_报告里单独数():
    agent, events = make_agent([CALL, CALL, LLMResponse(text="最终答案：NL", stop_reason="end_turn")], max_steps=2)
    answer = agent.run("q")
    c = Case(id="dab-1", question="q", gold_sql=(), official_answer="NL")
    t = Trial(c.id, 1, no_sql=True, official=True, answer=answer)
    digest(t, events)
    grade(t, c, None, None)
    assert t.step_limit and t.wrapped_up and t.answer_ok and t.failure == ""
    s = summarize([c], [t])
    assert (s["步数耗尽"], s["收尾"], s["收尾后答对"]) == (0, 1, 1)
    meta = {"cases": "d", "model": "m", "started": "", "git": "x", "dirty": False,
            "cases_sha1": "", "prompt_sha1": "", "trials": 1}
    assert "步数用完后收尾（其中答对） | 1（1）" in render(meta, s)


def test_评测_收尾没成的原因记进trial():
    agent, events = make_agent([CALL], max_steps=1)
    t = Trial("x", 1, answer=agent.run("q"))
    digest(t, events)
    assert t.step_limit and not t.wrapped_up and t.wrap_up_failure == "又去调了工具"
