"""子 Agent（delegate）：类型定义、组装、分派一轮、失败和停止。"""

from __future__ import annotations

import pytest

from data_agent.app import build_application
from data_agent.core.events import SubagentEvent, ToolFinished, TurnEnded, TurnStarted, collect_sink
from data_agent.core.messages import LLMResponse, ToolCall, Usage
from data_agent.settings import Settings
from data_agent.subagents import BUILTIN, PARENT_ONLY, load_definitions

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
    assert "delegate" in app.tools and [d.name for d in app.subagents] == ["explore"]
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


def test_一个任务失败不连累别的_全失败才算这次调用出错(tmp_path):
    llm = ScriptedProvider([
        delegate(("explore", "甲", "查甲"), ("explore", "乙", "查乙")),
        ConnectionError("断网了"),                        # 甲：请求失败
        say("乙查到了"),                                  # 乙
        say("汇总"),
    ])
    events = []
    app = build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events))
    app.agent.run("查甲和乙")
    finished = next(e for e in events if isinstance(e, ToolFinished) and e.name == "delegate")
    assert not finished.is_error
    assert "t1 · explore · 甲：失败" in finished.content and "ConnectionError: 断网了" in finished.content
    assert "t2 · explore · 乙：完成" in finished.content and "乙查到了" in finished.content


def test_没有这个类型_告诉模型有哪些(tmp_path):
    llm = ScriptedProvider([delegate(("analyst", "x", "y")), say("好")])
    events = []
    build_application(settings(tmp_path), llm=llm, on_event=collect_sink(events)).agent.run("q")
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.is_error and "没有子 Agent 类型 analyst" in finished.content and "explore" in finished.content


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
