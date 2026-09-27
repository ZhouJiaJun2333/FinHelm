"""检查点落盘：进程被杀了、第二天 --resume，也能接着跑那一轮。"""

from __future__ import annotations

import pytest

from data_agent.app import build_application
from data_agent.cli import _pending_turn
from data_agent.core.agent import InterruptedTurn
from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage
from data_agent.session import Session
from data_agent.settings import Settings

from fakes import ScriptedProvider, make_agent


def call(n: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(f"c{n}", "echo", {"text": str(n)})],
                       stop_reason="tool_use", usage=Usage(input=100 * n, output=10))


FINAL = LLMResponse(text="答案是 42", stop_reason="end_turn")


def crash(tmp_path, script) -> Session:
    """第一轮答完、第二轮跑到一半断了（CLI 里的顺序：每处理完一次输入 sync + 存检查点）。"""
    session = Session.create(tmp_path)
    agent, _ = make_agent(script, checkpoint_hook=session.save_checkpoint)
    agent.run("第一个问题")
    session.sync(agent.context.history)
    session.save_checkpoint(agent.interrupted)
    with pytest.raises(RuntimeError):
        agent.run("第二个问题")
    session.sync(agent.context.history)
    session.save_checkpoint(agent.interrupted)
    return session


def test_存盘读回来_接回去跑完(tmp_path):
    session = crash(tmp_path, [FINAL, call(1), call(2), RuntimeError("429"), FINAL])
    assert session.checkpoint_path.exists()

    reopened = Session.open(tmp_path)
    history = reopened.load()
    turn = reopened.load_checkpoint()
    assert turn.steps == 2 and turn.reason == "RuntimeError: 429" and turn.question == "第二个问题"

    agent, events = make_agent([FINAL], checkpoint_hook=reopened.save_checkpoint)
    agent.context.restore(history)
    agent.interrupted = turn
    assert agent.resume() == "答案是 42"
    assert agent.llm.calls == 1
    reopened.sync(agent.context.history)
    reopened.save_checkpoint(agent.interrupted)
    assert not reopened.checkpoint_path.exists(), "这一轮提交了，检查点删掉"
    assert [m.content for m in Session.open(tmp_path).load() if m.role == "user"] == ["第一个问题", "第二个问题"]


def test_读回来的进度和断之前一模一样(tmp_path):
    session = Session.create(tmp_path)
    saved: list[InterruptedTurn] = []
    agent, _ = make_agent([call(1), RuntimeError("断")],
                          checkpoint_hook=lambda t: (saved.append(t), session.save_checkpoint(t)))
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    assert Session.open(tmp_path).load_checkpoint() == saved[-1]


def test_第一轮就断了_会话也建起来了_resume找得到(tmp_path):
    session = Session.create(tmp_path)
    agent, _ = make_agent([RuntimeError("断网")], checkpoint_hook=session.save_checkpoint)
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    reopened = Session.open(tmp_path)
    assert reopened.load() == []
    assert reopened.load_checkpoint().question == "算一下"


def test_提交之后删检查点之前崩了_读的时候认出过期(tmp_path):
    session = crash(tmp_path, [FINAL, call(1), RuntimeError("断"), FINAL])
    stale = session.checkpoint_path.read_text(encoding="utf-8")
    agent, _ = make_agent([FINAL])
    agent.context.restore(Session.open(tmp_path).load())
    agent.run("第三个问题")
    session.sync(agent.context.history)                    # 历史写进去了……
    session.checkpoint_path.write_text(stale, encoding="utf-8")   # ……检查点还没来得及删
    reopened = Session.open(tmp_path)
    reopened.load()
    assert reopened.load_checkpoint() is None and not reopened.checkpoint_path.exists()


def test_检查点文件坏了_丢掉不报错(tmp_path):
    session = crash(tmp_path, [FINAL, call(1), RuntimeError("断")])
    session.checkpoint_path.write_text("{写了一半", encoding="utf-8")
    reopened = Session.open(tmp_path)
    reopened.load()
    assert reopened.load_checkpoint() is None


def test_存盘不留临时文件(tmp_path):
    session = crash(tmp_path, [FINAL, call(1), RuntimeError("断")])
    assert [p.name for p in session.root.iterdir() if p.suffix == ".tmp"] == []


# ================================================================ 程序重启后沙箱内核是新的
def _app(**kw):
    return build_application(Settings(provider="openai", openai_api_key="x", domain="research", **kw),
                             llm=ScriptedProvider())


def test_这一轮用过沙箱_重启后接着跑要提醒内核是新的(tmp_path):
    app = _app()
    used = InterruptedTurn((Message.user("做个 meta 分析"),
                            Message(role="assistant", tool_calls=[ToolCall("p1", "run_python", {"code": "df = 1"})]),
                            Message.tool_result("p1", "ok")), steps=1)
    assert "run_python" in app.fresh_kernel_note(used)
    assert app.fresh_kernel_note(InterruptedTurn((Message.user("你好"),), steps=0)) == ""

    session = Session.create(tmp_path)
    session.save_checkpoint(used)
    session = Session.open(tmp_path)
    session.load()
    turn = _pending_turn(session, app)
    assert turn.entries[-1].meta.synthetic and "内核是新的" in turn.entries[-1].content
