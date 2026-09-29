"""ask_user：一轮做到一半问用户，这一轮暂停（复用检查点），回答作为那次调用的结果，同一轮接着跑。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from data_agent.app import build_application
from data_agent.cli import pick_option
from data_agent.core.agent import ANSWER, NO_ANSWER, ONE_QUESTION, AwaitingUser
from data_agent.core.context import turn_starts
from data_agent.core.events import LLMResponded, ToolFinished, TurnResumed, UserAsked
from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage
from data_agent.session import Session
from data_agent.settings import Settings
from data_agent.tools.ask_user import AskUserTool

from fakes import EchoTool, ScriptedProvider, make_agent

FINAL = LLMResponse(text="按 40 万算，一共 18 个", stop_reason="end_turn")


def reply(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(text="", tool_calls=list(calls), stop_reason="tool_use", usage=Usage(input=100, output=10))


def ask(n: int = 1, question: str = "大客户的门槛是多少？", options=("30 万", "40 万")) -> ToolCall:
    return ToolCall(f"a{n}", "ask_user", {"question": question, "options": list(options)})


def echo(n: int) -> ToolCall:
    return ToolCall(f"e{n}", "echo", {"text": str(n)})


def agent_with_ask(script, **kw):
    return make_agent(script, tools=[EchoTool(), AskUserTool()], **kw)


def assert_paired(messages) -> None:
    calls = [c.id for m in messages for c in m.tool_calls]
    results = [m.tool_call_id for m in messages if m.role == "tool"]
    assert sorted(calls) == sorted(results)


def tool_result(messages, call_id: str) -> str:
    return next(m.content for m in messages if m.role == "tool" and m.tool_call_id == call_id)


# ================================================================ 暂停、回答、接着跑
def test_问了就暂停_回答成为那次调用的结果_同一轮接着跑():
    agent, events = agent_with_ask([reply(echo(1)), reply(ask()), reply(echo(2)), FINAL])
    with pytest.raises(AwaitingUser) as info:
        agent.run("有多少大客户？")

    q = info.value.pending
    assert (q.question, q.options, q.name) == ("大客户的门槛是多少？", ("30 万", "40 万"), "ask_user")
    assert agent.context.render() == [], "没答完的一轮不进正式历史"
    turn = agent.interrupted
    assert turn.pending == q and turn.steps == 2 and turn.reason == "等用户回答"
    assert [type(e) for e in events if isinstance(e, UserAsked)] == [UserAsked]

    assert agent.resume("40 万") == "按 40 万算，一共 18 个"
    assert agent.interrupted is None
    history = agent.context.render()
    assert len(turn_starts(history)) == 1, "回答不是新的一轮"
    assert_paired(history)
    assert tool_result(history, "a1") == ANSWER.format(answer="40 万")
    assert agent.llm.calls == 4, "前两步没有重新请求"
    assert [e.step for e in events if isinstance(e, LLMResponded)] == [1, 2, 3, 4], "步数接着数"
    assert next(e for e in events if isinstance(e, TurnResumed)).message == "40 万"
    finished = [e for e in events if isinstance(e, ToolFinished) and e.name == "ask_user"]
    assert [e.content for e in finished] == [ANSWER.format(answer="40 万")], "界面能看到这次调用有了结果"


def test_不回答_让它自己判断():
    agent, _ = agent_with_ask([reply(ask()), FINAL])
    with pytest.raises(AwaitingUser):
        agent.run("有多少大客户？")
    agent.resume("")
    assert tool_result(agent.context.render(), "a1") == NO_ANSWER


def test_不回答问新问题_这一轮作废():
    agent, _ = agent_with_ask([reply(ask()), FINAL])
    with pytest.raises(AwaitingUser):
        agent.run("有多少大客户？")
    agent.run("算了，看看销售额")
    assert agent.interrupted is None
    history = agent.context.render()
    assert [m.content for m in history if m.role == "user"] == ["算了，看看销售额"]
    assert_paired(history)


def test_同一批的其它调用照常执行_第二个问题直接回绝():
    agent, events = agent_with_ask([reply(echo(1), ask(1), ask(2, "还有一个问题"), echo(2)), FINAL])
    with pytest.raises(AwaitingUser) as info:
        agent.run("q")
    assert info.value.pending.call_id == "a1"
    entries = agent.interrupted.entries
    assert tool_result(entries, "e1") == "echo: 1" and tool_result(entries, "e2") == "echo: 2"
    assert tool_result(entries, "a2") == ONE_QUESTION
    assert sum(isinstance(e, UserAsked) for e in events) == 1
    agent.resume("30 万")
    history = agent.context.render()
    assert_paired(history)
    # 回答和同一批的其它结果挨着，都在下一次模型回复之前
    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "tool", "tool", "tool", "tool", "assistant"]


def test_问题是最后一步也能收尾():
    agent, events = agent_with_ask([reply(ask()), FINAL], max_steps=1)
    with pytest.raises(AwaitingUser):
        agent.run("q")
    assert agent.resume("40 万") == "按 40 万算，一共 18 个"


def test_参数不合法不会暂停():
    agent, _ = agent_with_ask([reply(ToolCall("a1", "ask_user", {"question": ""})), FINAL])
    assert agent.run("q") == "按 40 万算，一共 18 个"
    too_many = AskUserTool().execute({"question": "选哪个", "options": list("abcde")})
    assert too_many.is_error


# ================================================================ 存盘：程序重启了还能接着问
def test_停在提问上_存盘读回来接着跑(tmp_path):
    session = Session.create(tmp_path)
    agent, _ = agent_with_ask([reply(echo(1)), reply(ask())], checkpoint_hook=session.save_checkpoint)
    with pytest.raises(AwaitingUser):
        agent.run("有多少大客户？")
    session.save_checkpoint(agent.interrupted)

    reopened = Session.open(tmp_path)
    reopened.load()
    turn = reopened.load_checkpoint()
    assert turn == agent.interrupted and turn.pending.options == ("30 万", "40 万")

    # 重启后补的提醒（沙箱内核是新的）接在进度后面：回答要插在它前面，紧跟着那批工具结果
    note = Message.user("[提醒：沙箱重启过]").with_meta(synthetic=True)
    turn = replace(turn, entries=(*turn.entries, note))
    fresh, _ = agent_with_ask([FINAL])
    fresh.interrupted = turn
    fresh.resume("40 万")
    history = fresh.context.render()
    assert_paired(history)
    at = next(i for i, m in enumerate(history) if m.tool_call_id == "a1")
    assert history[at + 1].content == "[提醒：沙箱重启过]"


# ================================================================ 组装和命令行
def test_开关控制注册():
    base = dict(database_url="", python_sandbox=False, r_sandbox=False)
    assert "ask_user" in build_application(Settings(**base, ask_user=True), llm=ScriptedProvider()).tools
    assert "ask_user" not in build_application(Settings(**base, ask_user=False), llm=ScriptedProvider()).tools


def test_命令行_输入编号就是选那个选项():
    options = ("30 万", "40 万")
    assert pick_option("2", options) == "40 万"
    assert pick_option("3", options) == "3" and pick_option("按 35 万", options) == "按 35 万"
    assert pick_option("1", ()) == "1"


# ================================================================ 评测：没有真人，按 replies 回答
def test_评测_按顺序拿replies回答_用完了回没回答():
    from evals.runner import answer

    script = [reply(ask(1)), reply(ask(2, "还要按渠道拆吗？")), reply(ask(3, "第三个")), FINAL]
    agent, events = agent_with_ask(script)
    assert answer(agent, "有多少大客户？", replies=("40 万", "不用拆")) == "按 40 万算，一共 18 个"
    history = agent.context.render()
    assert [tool_result(history, f"a{n}") for n in (1, 2, 3)] == [
        ANSWER.format(answer="40 万"), ANSWER.format(answer="不用拆"), NO_ANSWER]
    assert len(turn_starts(history)) == 1


def test_评测_该不该问分开检查():
    from evals.cases import Case, load_cases
    from evals.runner import Trial, check_ask, digest_asks

    events = [UserAsked("ask_user", "门槛是多少？")]
    t = Trial("ask-001", 1)
    digest_asks(t, events)
    check_ask(t, Case("ask-001", "q", (), ask=True))
    assert t.asks == ["门槛是多少？"] and t.ask_ok is True
    check_ask(t, Case("ask-004", "q", (), ask=False))
    assert t.ask_ok is False
    quiet = Trial("x", 1)
    check_ask(quiet, Case("x", "q", ()))
    assert quiet.ask_ok is None

    cases = {c.id: c for c in load_cases("ask").cases}
    assert cases["ask-001"].ask is True and cases["ask-001"].replies
    assert cases["ask-004"].ask is False and not cases["ask-004"].replies


def test_评测_回答提问后的接着跑不算出错重跑():
    from evals.runner import _error_resumes

    asked = [UserAsked("ask_user", "q"), TurnResumed(1, "40 万")]
    crashed = [TurnResumed(2)]
    assert _error_resumes(asked) == 0 and _error_resumes(asked + crashed) == 1
