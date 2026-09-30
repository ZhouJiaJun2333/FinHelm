"""子 Agent（delegate）：类型定义、组装、分派一轮、失败和停止。"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.core.events import (
    LLMResponded,
    SubagentEvent,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    TurnStarted,
    collect_sink,
)
from data_agent.core.messages import LLMResponse, ToolCall, Usage
from data_agent.core.provider import LLMProvider
from data_agent.db.connection import QueryResult
from data_agent.settings import Settings
from data_agent.subagents import BUILTIN, PARENT_ONLY, load_definitions
from data_agent.tools.paths import SandboxPaths
from data_agent.tools.python import PYTHON_KERNEL
from data_agent.tools.sandbox import Sandbox
from data_agent.tools.sql.results import ResultStore

from fakes import ScriptedProvider


def settings(tmp_path, **kw) -> Settings:
    base = dict(provider="openai", openai_api_key="x", database_url="", python_sandbox=False, r_sandbox=False,
                mcp_enabled=False, docs_dirs="", ask_user=True, memory_enabled=True, subagents=True,
                project_dir=str(tmp_path), memory_dir=str(tmp_path / "mem"))
    return Settings(**{**base, **kw})


def delegate(*tasks: tuple[str, str, str], call_id="d1") -> LLMResponse:
    return LLMResponse(text="", stop_reason="tool_use", tool_calls=[ToolCall(call_id, "delegate", {
        "tasks": [{"agent": a, "title": t, "prompt": p} for a, t, p in tasks]})])


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end_turn", usage=Usage(input=100, output=10))


# ================================================================ 类型定义
def test_内置的explore_项目的同名定义盖过内置的_写坏的跳过(tmp_path):
    project = tmp_path / "agents"
    project.mkdir()
    (project / "explore.md").write_text("---\nname: explore\ndescription: 项目自己的\ntools: run_sql\n---\n只查库",
                                        encoding="utf-8")
    (project / "bad.md").write_text("---\nname: bad\ndescription: 没写工具\n---\n", encoding="utf-8")
    (project / "Wrong.md").write_text("---\nname: other\ndescription: x\ntools: *\n---\n", encoding="utf-8")
    found, problems = load_definitions([project, BUILTIN])
    explore = next(d for d in found if d.name == "explore")
    assert explore.description == "项目自己的" and explore.role() == "只查库"
    assert len(problems) == 2 and any("缺 tools" in p for p in problems)

    builtin = next(d for d in load_definitions([BUILTIN])[0] if d.name == "explore")
    assert "run_sql" in builtin.tools and "run_python" not in builtin.tools


def test_子Agent拿不到只属于主Agent的工具(tmp_path):
    (tmp_path / "a.md").write_text("---\nname: a\ndescription: x\ntools: *\n---\n", encoding="utf-8")
    everything = load_definitions([tmp_path])[0][0]
    names = ["run_sql", "delegate", "ask_user", "remember", "read_memory"]
    assert everything.allowed(names) == ["run_sql", "read_memory"]
    assert not PARENT_ONLY & set(everything.allowed(names))


def test_开关关着就没有delegate(tmp_path):
    assert "delegate" not in build_application(settings(tmp_path, subagents=False), llm=ScriptedProvider()).tools
    app = build_application(settings(tmp_path), llm=ScriptedProvider())
    assert "delegate" in app.tools and sorted(d.name for d in app.subagents) == ["analyst", "explore", "verifier"]
    assert "explore：" in app.tools.get("delegate").description


# ================================================================ 分派一轮
def test_分派一个任务_子Agent上下文全新_只交回结论(tmp_path):
    llm = ScriptedProvider([
        delegate(("explore", "找订单表", "订单数据在哪张表？交回表名和金额字段")),
        say("订单在 shop.orders，金额是 amount"),         # 子 Agent 的回答
        say("好的，订单在 shop.orders"),                  # 主 Agent 收到后回答用户
    ])
    events = []
    app = build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events))
    assert app.agent.run("先帮我看看订单数据在哪") == "好的，订单在 shop.orders"

    # 子 Agent 的请求：只有任务说明，没有主对话；系统提示词是子 Agent 的；工具里没有主 Agent 专属的
    child_messages, child_system, child_tools = llm.seen[1], llm.system_seen[1], llm.tools_seen[1]
    assert [m.content for m in child_messages] == ["订单数据在哪张表？交回表名和金额字段"]
    assert "子 Agent" in child_system and "查探子 Agent" in child_system
    assert {t["name"] for t in child_tools} == {"read_memory"}

    # 主 Agent 下一次请求看到的：一条 delegate 结果，带着子 Agent 的结论；子 Agent 的过程不在里面
    result = llm.seen[2][-1]
    assert result.role == "tool" and "t1 · explore · 找订单表：完成" in result.content
    assert "订单在 shop.orders，金额是 amount" in result.content
    assert len(llm.seen[2]) == 3

    # 子 Agent 的事件包一层从主 Agent 发出来
    inner = [e.event for e in events if isinstance(e, SubagentEvent)]
    assert isinstance(inner[0], TurnStarted) and isinstance(inner[-1], TurnEnded)
    assert all(e.task == "t1" and e.agent == "explore" for e in events if isinstance(e, SubagentEvent))
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.name == "delegate")
    assert finished.details[0].ok and finished.details[0].usage.output == 10
    # 子 Agent 花的 token 记在主 Agent 账上：主 Agent 自己请求了 2 次（剧本里分派那条没带用量），子 Agent 1 次
    assert app.agent.session_usage.output == 20


class Routed(LLMProvider):
    """主 Agent 照剧本走；子 Agent 按它的任务说明（第一条消息）查 children 回答，值可以是异常或函数。"""

    model = "routed"
    context_window = None

    def __init__(self, parent: list, children: dict) -> None:
        self.parent = ScriptedProvider(parent)
        self.children = children

    def chat(self, messages, tools=None, system=None, max_tokens=None):
        if system and system.startswith("你是 FinHelm 的子 Agent"):
            reply = self.children[messages[0].content]
            reply = reply() if callable(reply) and not isinstance(reply, type) else reply
            if isinstance(reply, BaseException):
                raise reply
            return reply
        return self.parent.chat(messages, tools, system, max_tokens)


def test_几个任务同时跑(tmp_path):
    both_in = threading.Barrier(2, timeout=5)           # 两个子 Agent 都进了请求才放行：不是一个接一个

    def answer(text):
        def reply():
            both_in.wait()
            return say(text)
        return reply

    llm = Routed([delegate(("explore", "甲", "查甲"), ("explore", "乙", "查乙")), say("汇总")],
                 {"查甲": answer("甲是 1"), "查乙": answer("乙是 2")})
    events = []
    build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events)).agent.run("查甲和乙")
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.name == "delegate")
    assert "t1 · explore · 甲：完成" in finished.content and "甲是 1" in finished.content
    assert "t2 · explore · 乙：完成" in finished.content and "乙是 2" in finished.content
    assert finished.content.index("t1") < finished.content.index("t2")      # 交回的顺序按分派的顺序
    ended = {e.task for e in events if isinstance(e, SubagentEvent) and isinstance(e.event, TurnEnded)}
    assert ended == {"t1", "t2"}


def test_任务号在会话里接着编(tmp_path):
    llm = Routed([delegate(("explore", "甲", "查甲")), say("好"), delegate(("explore", "乙", "查乙"), call_id="d2"),
                  say("好")], {"查甲": say("1"), "查乙": say("2")})
    events = []
    app = build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events))
    app.agent.run("一")
    app.agent.run("二")
    assert [e.task for e in events if isinstance(e, SubagentEvent) and isinstance(e.event, TurnStarted)] == ["t1", "t2"]


def test_一个任务失败不连累别的_全失败才算这次调用出错(tmp_path):
    llm = Routed([delegate(("explore", "甲", "查甲"), ("explore", "乙", "查乙")), say("汇总")],
                 {"查甲": ConnectionError("断网了"), "查乙": say("乙查到了")})
    events = []
    app = build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events))
    app.agent.run("查甲和乙")
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.name == "delegate")
    assert not finished.is_error
    assert "t1 · explore · 甲：失败" in finished.content and "ConnectionError: 断网了" in finished.content
    assert "t2 · explore · 乙：完成" in finished.content and "乙查到了" in finished.content


def test_没有这个类型_告诉模型有哪些(tmp_path):
    llm = ScriptedProvider([delegate(("planner", "x", "y")), say("好")])
    events = []
    build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events)).agent.run("q")
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.is_error and "没有子 Agent 类型 planner" in finished.content and "explore" in finished.content


class Stopped(BaseException):
    pass


def test_停止_在子Agent的事件上抛出_主Agent这一轮也停下(tmp_path):
    llm = ScriptedProvider([delegate(("explore", "x", "y")), say("子"), say("主")])

    def sink(event):
        if isinstance(event, SubagentEvent) and isinstance(event.event, TurnStarted):
            raise Stopped()

    app = build_application(settings(tmp_path), llm=llm, on_event=sink)
    with pytest.raises(Stopped):
        app.agent.run("q")
    assert llm.calls == 1                     # 子 Agent 一次都没请求
    assert app.agent.interrupted is not None and app.agent.interrupted.reason.startswith("Stopped")


def test_停止_一个子Agent被停下_兄弟在下一个事件上也停(tmp_path):
    first_stopped = threading.Event()

    def second():
        first_stopped.wait(5)                  # 乙的请求等甲被停下了才返回：返回后下一个事件就该停
        return LLMResponse(text="", stop_reason="tool_use", tool_calls=[ToolCall("x", "read_memory", {"scope": "user",
                                                                                                        "name": "a"})])

    llm = Routed([delegate(("explore", "甲", "查甲"), ("explore", "乙", "查乙")), say("主")],
                 {"查甲": say("甲"), "查乙": second})
    seen = []

    def sink(event):
        seen.append(event)
        if isinstance(event, SubagentEvent) and event.task == "t1" and isinstance(event.event, LLMResponded):
            first_stopped.set()
            raise Stopped()

    app = build_application(settings(tmp_path), llm=llm, on_event=sink)
    with pytest.raises(Stopped):
        app.agent.run("q")
    inner = [(e.task, e.event) for e in seen if isinstance(e, SubagentEvent)]
    assert not any(isinstance(ev, ToolStarted) for _, ev in inner)          # 乙的工具没执行
    assert {t for t, ev in inner if isinstance(ev, TurnEnded)} == {"t1", "t2"}   # 两个都报了停下
    assert app.agent.interrupted.reason.startswith("Stopped")


# ================================================================ 共用的东西
def test_结果编号_同时存不重号_记得是哪个任务的():
    store = ResultStore()
    table = QueryResult(["n"], [(1,)], False, 1)

    def work(name):
        with store.origin(name):
            for _ in range(50):
                store.add("select 1", table)

    threads = [threading.Thread(target=work, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.add("select 1", table)                        # 主 Agent 的
    assert len(store.refs()) == len(set(store.refs())) == 201
    assert all(len(store.made_by(f"t{i}")) == 50 for i in range(4))
    assert store.refs()[-1] not in {r for i in range(4) for r in store.made_by(f"t{i}")}


def test_子Agent的沙箱_figures挂到自己的目录(tmp_path):
    figures = tmp_path / "figures" / "t1"
    sandbox = Sandbox.docker("img", PYTHON_KERNEL, tmp_path, lambda ref: {}, figures_dir=figures)
    assert f"type=bind,source={figures.resolve()},target=/work/figures" in sandbox.command
    assert sandbox._host_path("figures/a.png") == figures.resolve() / "a.png"
    assert sandbox._host_path("inputs/x.csv") == tmp_path.resolve() / "inputs" / "x.csv"
    assert not figures.exists()                         # 用到沙箱时才建目录

    # 读文件、看图：它说 figures/a.png 指的是自己目录里的；inputs/ 照常共享
    figures.mkdir(parents=True)
    (figures / "a.png").write_bytes(b"x")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "d.csv").write_text("a", encoding="utf-8")
    paths = SandboxPaths(tmp_path, figures_dir=figures)
    for raw in ("figures/a.png", "/work/figures/a.png", str(figures / "a.png")):
        assert paths.resolve(raw) == (figures / "a.png").resolve()
        assert paths.display(paths.resolve(raw)) == "figures/a.png"
    assert paths.resolve("inputs/d.csv") == (tmp_path / "inputs" / "d.csv").resolve()


def test_同一次分派里任务说明一模一样_拒绝(tmp_path):
    llm = ScriptedProvider([delegate(("explore", "甲", "查订单  表"), ("analyst", "乙", "查订单 表")), say("好")])
    events = []
    build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events)).agent.run("q")
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.is_error and "第 1 个和第 2 个任务的说明一模一样" in finished.content
    assert llm.calls == 2                     # 子 Agent 一个都没起


# ================================================================ 存盘和重新打开
def test_子任务的过程存盘_重新打开后任务号接着编_界面能建出子任务(tmp_path):
    from data_agent.session.store import load_transcript
    from data_agent.web.serialize import timeline

    transcripts = tmp_path / "subagents"
    llm = Routed([delegate(("explore", "甲", "查甲")), say("好")], {"查甲": say("甲是 1")})
    app = build_application(settings(tmp_path), llm=llm, subagent_dir=transcripts)
    app.agent.run("q")
    header, entries = load_transcript(transcripts / "t1.jsonl")
    assert header == {"type": "subagent", "version": 1, "task": "t1", "agent": "explore", "title": "甲", "prompt": "查甲"}
    assert [e.content for e in entries] == ["查甲", "甲是 1"]

    # 重新打开：delegate 那条下面建出子任务，它自己的条目和主对话一样的形状
    items = timeline(app.agent.context.history, subagents=transcripts)
    tool = next(i for i in items if i["kind"] == "tool")
    assert [(c["task"], c["agent"], c["title"], c["status"]) for c in tool["children"]] == [("t1", "explore", "甲", "done")]
    assert [(i["kind"], i["text"]) for i in tool["children"][0]["items"]] == [("user", "查甲"), ("assistant", "甲是 1")]

    # 同一个会话再开一次：新任务从 t2 编起，不盖掉 t1 的过程
    llm2 = Routed([delegate(("explore", "乙", "查乙")), say("好")], {"查乙": say("乙")})
    build_application(settings(tmp_path), llm=llm2, subagent_dir=transcripts).agent.run("q2")
    assert sorted(p.name for p in transcripts.iterdir()) == ["t1.jsonl", "t2.jsonl"]


# ================================================================ 交付前复核
def test_复核_每轮第一次收工时推一次_离上限太近不推():
    from data_agent.core.agent import TurnOutcome
    from data_agent.subagents.verify import VERIFY_NUDGE, verify_before_finish

    hook = verify_before_finish(max_steps=6)
    end = lambda step, tools=False: hook(TurnOutcome(step, say("x"), tools))   # noqa: E731
    assert end(1, tools=True).action == "continue"
    first = end(2)
    assert first.action == "continue" and first.nudge == VERIFY_NUDGE
    assert end(3, tools=True).action == "continue"
    assert end(4).action == "end"                 # 这一轮推过了
    assert end(1).nudge == VERIFY_NUDGE           # 新的一轮又推
    hook2 = verify_before_finish(max_steps=6)
    assert hook2(TurnOutcome(5, say("x"), False)).action == "end"   # 只剩 1 步，来不及复核


def test_开了复核_交付前派verifier_评测拿到复核前那一版(tmp_path):
    from data_agent.subagents.verify import VERIFY_NUDGE
    from evals.runner import Trial, digest

    llm = Routed([
        say("答案是 41"),                                            # 第一次打算收工
        delegate(("verifier", "复核", "原问题：q。主 Agent 的答案：41")),
        say("复核发现应该是 42。最终答案：42"),
    ], {"原问题：q。主 Agent 的答案：41": say("不一致：应该是 42")})
    events = []
    app = build_application(settings(tmp_path, verify=True), llm=llm, on_event=collect_sink(events))
    assert app.agent.run("q") == "复核发现应该是 42。最终答案：42"
    assert llm.parent.seen[1][-1].content == VERIFY_NUDGE

    t = Trial("c1", 1)
    digest(t, events)
    assert t.answer_before == "答案是 41" and t.subagents == ["verifier"]
    assert t.subagent_usage.output == 10 and t.peak_context > 0

    # 没开子 Agent 时 VERIFY 不起作用（没有 verifier 可派）
    llm2 = ScriptedProvider([say("答案是 41")])
    assert build_application(settings(tmp_path, subagents=False, verify=True), llm=llm2).agent.run("q") == "答案是 41"
