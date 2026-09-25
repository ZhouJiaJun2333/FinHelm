"""会话目录：日志怎么写、怎么读回来，读回来的上下文和原来一模一样。不需要 key 和数据库。"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from data_agent.cli import open_session, parse_args
from data_agent.core.context import ClearOldToolResults, Context, HistoryCompacted, ToolResultsCleared
from data_agent.core.context.base import Marker
from data_agent.core.messages import LLMResponse, Message, MessageMeta, ToolCall, Usage
from data_agent.db.connection import QueryResult
from data_agent.session import Session
from data_agent.session.codec import _marker_classes, decode, encode
from data_agent.tools.sql.results import ResultStore, markdown_table

from fakes import make_agent

BIG = "| 华东 | 8100531.47 | 3017 |\n" * 60


# ================================================================ 编解码
SAMPLE_MESSAGES = [
    Message.user("各区域销售额"),
    Message(role="assistant", content="我查一下",
            tool_calls=[ToolCall("c1", "run_sql", {"sql": "SELECT 1", "purpose": "试试"})],
            raw={"role": "assistant", "reasoning_content": "先看表", "tool_calls": [{"id": "c1"}]},
            meta=MessageMeta(usage=Usage(input=10, output=5, cache_read=3), measured_on="abc")),
    Message.tool_result("c1", "UndefinedColumn: x", is_error=True),
    Message.tool_result("c2", BIG, summary="结果 r1：60 行 × 3 列"),
    Message.user("请继续完成上面的任务。").with_meta(synthetic=True),
]
SAMPLE_MARKERS = [
    ToolResultsCleared(frozenset({"c2", "c1"})),
    HistoryCompacted(summary="## 用户的目标\n看销售额", kept_turns=1, compacted_turns=3,
                     usage=Usage(input=100, output=50)),
]


@pytest.mark.parametrize("entry", SAMPLE_MESSAGES + SAMPLE_MARKERS, ids=lambda e: type(e).__name__)
def test_写进去读出来一模一样(entry):
    line = json.dumps(encode(entry), ensure_ascii=False)
    assert decode(json.loads(line)) == entry


def test_每种标记都有编解码样例():
    """新写一种 ContextEdit 的标记，得在上面加一个样例 —— 字段类型不认得，测试会炸在这里而不是恢复会话时。"""
    ours = {name for name, cls in _marker_classes().items() if cls.__module__.startswith("data_agent.")}
    assert ours == {type(m).__name__ for m in SAMPLE_MARKERS}
    assert all(isinstance(m, Marker) for m in SAMPLE_MARKERS)


def test_旧日志缺了后来加的字段_用默认值():
    data = encode(SAMPLE_MARKERS[1])
    del data["usage"]
    assert decode(data).usage == Usage()


def test_普通消息写出来一眼能读():
    assert encode(Message.user("你好")) == {"type": "message", "role": "user", "content": "你好"}


# ================================================================ 写
def test_新会话不写就不建目录(tmp_path):
    session = Session.create(tmp_path)
    session.sync([])
    assert not session.root.exists()


def test_只写成功的轮次_失败的一轮已经回滚不会写进去(tmp_path):
    agent, _ = make_agent([LLMResponse(text="答1"), RuntimeError("网络断了"), LLMResponse(text="答3")])
    session = Session.create(tmp_path)

    agent.run("问1")
    session.sync(agent.context.history)
    with pytest.raises(RuntimeError):
        agent.run("问2")
    session.sync(agent.context.history)
    agent.run("问3")
    session.sync(agent.context.history)

    lines = [json.loads(line) for line in session.log_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "session" and lines[0]["id"] == session.id
    assert [d["content"] for d in lines[1:]] == ["问1", "答1", "问3", "答3"]


def test_reset之后只读回reset之后的部分(tmp_path):
    session = Session.create(tmp_path)
    session.sync([Message.user("旧"), Message.assistant("旧答")])
    session.sync([])                                   # /reset
    session.sync([Message.user("新")])
    assert Session.open(tmp_path, session.id).load() == [Message.user("新")]


def test_写到一半崩了_最后一行读不出来就跳过(tmp_path):
    session = Session.create(tmp_path)
    session.sync([Message.user("问1"), Message.assistant("答1")])
    with open(session.log_path, "a", encoding="utf-8") as f:
        f.write('{"type": "message", "role": "us')
    assert len(Session.open(tmp_path, session.id).load()) == 2


def test_接着写不会重复(tmp_path):
    session = Session.create(tmp_path)
    history = [Message.user("问1"), Message.assistant("答1")]
    session.sync(history)

    resumed = Session.open(tmp_path)
    loaded = resumed.load()
    resumed.sync([*loaded, Message.user("问2")])
    assert len(Session.open(tmp_path).load()) == 3


# ================================================================ 恢复
def test_恢复出来的上下文和原来发给模型的完全一样(tmp_path):
    """标记（清理了哪些）、锚点、原生 raw 都要原样回来，否则恢复后第一次请求就和原来不一样。"""
    def context():
        return Context([ClearOldToolResults(trigger_tokens=1_000, keep_recent=1, clear_at_least=100)])

    script = []
    for i in range(3):
        script += [LLMResponse(text="", tool_calls=[ToolCall(f"c{i}", "echo", {"text": BIG})],
                               usage=Usage(input=100 * (i + 1)), raw_content={"n": i}),
                   LLMResponse(text=f"答{i}", usage=Usage(input=150 * (i + 1)))]
    agent, _ = make_agent(script, context=context())
    session = Session.create(tmp_path)
    for i in range(3):
        agent.run(f"问{i}")
        session.sync(agent.context.history)
    assert any(isinstance(e, ToolResultsCleared) for e in agent.context.history), "要测到标记"

    resumed, _ = make_agent(context=context())
    resumed.context.restore(Session.open(tmp_path).load())
    assert resumed.context.render() == agent.context.render()
    assert resumed.context_usage() == agent.context_usage()


# ================================================================ 查询结果
def test_查询结果落盘_重开以后编号接着往下编(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultStore(path)
    rows = [("华东", Decimal("810.05"), date(2024, 3, 1)), (None, Decimal("0.10"), date(2024, 3, 2))]
    shown = markdown_table(["region", "gmv", "day"], rows)
    store.add("SELECT 1", QueryResult(["region", "gmv", "day"], rows, False, 3))
    store.add("SELECT 2", QueryResult(["n"], [(1,)], False, 1))

    reopened = ResultStore(path)
    assert reopened.refs() == ["r1", "r2"]
    assert reopened.get("r2").sql == "SELECT 2"
    # 存的是用户当时看到的：Decimal、日期变成字符串，显示出来一样
    r1 = reopened.get("r1").result
    assert markdown_table(r1.columns, r1.rows) == shown
    assert reopened.add("SELECT 3", QueryResult(["n"], [(2,)], False, 1)).ref == "r3"


def test_不给路径就只在内存里(tmp_path):
    ResultStore().add("SELECT 1", QueryResult(["n"], [(1,)], False, 1))
    assert list(tmp_path.iterdir()) == []


# ================================================================ 命令行
def test_resume参数():
    assert parse_args([]).resume is None
    assert parse_args(["--resume"]).resume == ""
    assert parse_args(["--resume", "20260925-101010-abcd"]).resume == "20260925-101010-abcd"


def test_不写ID就恢复最近的一次_找不到就报错(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_session(tmp_path, "")
    old = Session.create(tmp_path)
    old.sync([Message.user("旧")])
    new = Session(tmp_path / "20990101-000000-ffff")
    new.sync([Message.user("新")])

    session, history = open_session(tmp_path, "")
    assert session.id == new.id and history == [Message.user("新")]
    with pytest.raises(FileNotFoundError):
        open_session(tmp_path, "没有这个")
    session, history = open_session(tmp_path, None)
    assert history == [] and not session.root.exists()
