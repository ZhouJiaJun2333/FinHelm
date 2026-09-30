"""Web 后端：会话 runner（后台线程跑回合、事件推给订阅者）和 HTTP 接口。不起浏览器、不连模型。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage
from data_agent.db.connection import QueryResult
from data_agent.session import Session
from data_agent.settings import Settings
from data_agent.tools.python.run_python import RunPythonTool
from data_agent.tools.sandbox import Execution
from data_agent.tools.sql.results import ResultStore
from data_agent.tools.sql.run_sql import _format
from data_agent.web.serialize import timeline
from data_agent.web.runner import SessionRunner
from data_agent.web.auth import Users
from data_agent.web.server import create_app

from fakes import ScriptedProvider

FINAL = LLMResponse(text="华东最高", stop_reason="end_turn", usage=Usage(input=100, output=10))


def settings(tmp_path, **kw) -> Settings:
    return Settings(provider="openai", openai_api_key="x", database_url="", memory_enabled=False,
                    mcp_enabled=False, python_sandbox=False, r_sandbox=False, docs_dirs="", ask_user=True,
                    project_dir=str(tmp_path), sessions_dir=str(tmp_path / "sessions"),
                    web_dir=str(tmp_path / "web"), memory_dir=str(tmp_path / "mem"), **kw)


def ask(question="门槛多少？", options=("30 万", "40 万")) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall("a1", "ask_user", {"question": question, "options": list(options)})],
                       stop_reason="tool_use")


class Harness:
    """一个 runner + 一个订阅者队列。drain() 等后台线程跑完、把队列里的消息全拿出来。"""

    def __init__(self, tmp_path, script, llm=None) -> None:
        self.loop = asyncio.new_event_loop()
        self.llm = llm or ScriptedProvider(script)
        self.runner = SessionRunner(settings(tmp_path), Session.create(tmp_path / "sessions"), self.loop, self.llm)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.runner.subscribe(self.queue)

    def drain(self) -> list[dict]:
        if self.runner._thread is not None:
            self.runner._thread.join(5)
        self.loop.run_until_complete(asyncio.sleep(0))
        out = []
        while not self.queue.empty():
            out.append(self.queue.get_nowait())
        return out

    def close(self) -> None:
        self.runner.close()
        self.loop.close()


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(script, llm=None):
        h = Harness(tmp_path, script, llm)
        made.append(h)
        return h

    yield make
    for h in made:
        h.close()


def test_先快照_再按顺序收到事件_跑完落盘(harness):
    h = harness([FINAL])
    h.runner.send("哪个大区最高")
    messages = h.drain()

    types = [m["type"] for m in messages]
    assert types[0] == "snapshot" and types[-1] == "idle"
    assert types[1:-1] == ["TurnStarted", "StepStarted", "TextDelta", "LLMResponded", "TurnEnded"]
    assert all("state" in m for m in messages if m["type"] != "TextDelta")
    assert messages[-1]["state"]["status"] == "idle" and messages[-1]["state"]["answer"] == "华东最高"
    assert h.runner.session.log_path.exists()

    items = h.runner.snapshot()["data"]["items"]
    assert [(i["kind"], i["text"]) for i in items] == [("user", "哪个大区最高"), ("assistant", "华东最高")]


def test_问用户_停下来_下一句话就是回答(harness):
    h = harness([ask(), FINAL])
    h.runner.send("有多少大客户")
    idle = h.drain()[-1]
    assert idle["data"]["interrupted"]["pending"]["options"] == ["30 万", "40 万"]
    assert idle["state"]["status"] == "asking"
    # 快照里看得到没跑完的那一轮（刷新页面也能接着答）
    kinds = [i["kind"] for i in h.runner.snapshot()["data"]["items"]]
    assert kinds == ["user", "tool"]

    h.runner.send("2")                       # 编号选第二个
    last = h.drain()
    assert last[-1]["state"]["status"] == "idle"
    tool_results = [m for m in h.llm.seen[-1] if m.role == "tool"]
    assert "40 万" in tool_results[0].content


class StopOnFirstDelta(ScriptedProvider):
    def __init__(self, script, runner_ref) -> None:
        super().__init__(script)
        self.runner_ref = runner_ref

    def stream(self, messages, tools=None, system=None, max_tokens=None, *, on_delta):
        self.runner_ref[0].stop()
        on_delta("华", False)
        on_delta("东", False)
        return self.chat(messages, tools=tools, system=system)


def test_停止_停在下一个事件上_进度留着可以继续(harness):
    ref = []
    h = harness(None, llm=StopOnFirstDelta([FINAL], ref))
    ref.append(h.runner)
    h.runner.send("q")
    messages = h.drain()
    ended = next(m for m in messages if m["type"] == "TurnEnded")
    assert ended["data"]["interrupted"].startswith("Stopped")
    assert messages[-1]["data"]["interrupted"] is not None
    assert not h.runner.busy


def test_上传的文件进inputs_下一条消息告诉模型(harness):
    h = harness([FINAL])
    path = h.runner.upload("../../坏名字.csv", b"a,b\n1,2\n")
    assert path.parent == h.runner.session.work_dir / "inputs" and path.name == "坏名字.csv"
    h.runner.send("看看")
    h.drain()
    assert h.llm.seen[0][-1].content.startswith("[用户上传了文件")


def test_重新打开会话_历史里的图和表都找得回来(tmp_path):
    work = tmp_path / "work"
    (work / "figures").mkdir(parents=True)
    png, pdf = work / "figures" / "forest.png", work / "figures" / "forest.pdf"
    png.write_bytes(b"png")
    pdf.write_bytes(b"pdf")
    results = ResultStore()
    table = results.add("select 1 as n", QueryResult(["n"], [(1,)], truncated=False, elapsed_ms=3))
    sandbox_text = RunPythonTool.render(RunPythonTool.__new__(RunPythonTool),
                                        Execution(output="画好了", figures=[png, pdf, work / "figures" / "gone.png"]))
    entries = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "run_sql", {"sql": "select 1"}),
                                              ToolCall("c2", "run_python", {"code": "plot()"})]),
        Message.tool_result("c1", _format(table.ref, table.result)),
        Message.tool_result("c2", sandbox_text),
    ]
    items = timeline(entries, lambda path: f"/f/{path.name}", results)
    assert items[0]["details"]["kind"] == "table" and items[0]["details"]["rows"] == [[1]]
    # 已经删掉的文件不给
    assert items[1]["details"]["figures"] == ["/f/forest.png", "/f/forest.pdf"]
    # 不给 file_url / results 时和以前一样只有文字
    assert all(i["details"] is None for i in timeline(entries))


# ================================================================ HTTP
@pytest.fixture
def client(tmp_path):
    app = create_app(settings(tmp_path), llm=ScriptedProvider([FINAL]))
    with TestClient(app) as c:
        yield c


def test_新建会话_发消息_会话列表有标题(client):
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "  "}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "哪个大区最高"}).json() == {"ok": True}
    runner = client.app.state.hubs[""].runners[sid]
    runner._thread.join(5)
    sessions = client.get("/api/sessions").json()
    assert sessions[0] == {**sessions[0], "id": sid, "title": "哪个大区最高"}


def test_没有的会话404_文件不能跳出work目录(client):
    assert client.get("/api/sessions/不存在/results/r1").status_code == 404
    sid = client.post("/api/sessions").json()["id"]
    work = client.app.state.hubs[""].runners[sid].session.work_dir
    (work / "figures").mkdir(parents=True)
    (work / "figures" / "a.png").write_bytes(b"png")
    assert client.get(f"/api/sessions/{sid}/files/figures/a.png").content == b"png"
    assert client.get(f"/api/sessions/{sid}/files/../session.jsonl").status_code == 404
    assert client.get(f"/api/sessions/{sid}/files/..%2F..%2F.env").status_code == 404


def test_改名_删除进回收站_恢复(client):
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/messages", json={"text": "第一个问题"})
    client.app.state.hubs[""].runners[sid]._thread.join(5)

    assert client.patch(f"/api/sessions/{sid}", json={"title": "  华东  销售 "}).json() == {"ok": True}
    assert client.get("/api/sessions").json()[0]["title"] == "华东 销售"

    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    assert sid not in [s["id"] for s in client.get("/api/sessions").json()]
    assert [t["id"] for t in client.get("/api/trash").json()] == [sid]
    assert client.get(f"/api/sessions/{sid}/events").status_code == 404

    assert client.post(f"/api/trash/{sid}/restore").json() == {"ok": True}
    assert client.get("/api/sessions").json()[0] == {**client.get("/api/sessions").json()[0], "id": sid, "title": "华东 销售"}
    assert client.get("/api/trash").json() == []


# ================================================================ 登录
def test_账号_密码只存哈希_凭证能验_删号就失效(tmp_path):
    users = Users(tmp_path)
    with pytest.raises(ValueError):
        users.add("../x", "12345678")
    with pytest.raises(ValueError):
        users.add("alice", "short")
    users.add("alice", "correct horse")
    assert "correct horse" not in users.path.read_text(encoding="utf-8")
    assert users.verify("alice", "correct horse") and not users.verify("alice", "wrong one")
    assert not users.verify("nobody", "correct horse")

    token = users.token("alice")
    assert users.check(token) == "alice"
    assert users.check(token.replace("alice", "bob", 1)) is None          # 改了名字签名就对不上
    assert users.check(users.token("alice", now=0)) is None               # 过期
    users.remove("alice")
    assert users.check(token) is None


def test_建了账号就要登录_每个人只看得到自己的会话(tmp_path):
    s = settings(tmp_path)
    users = Users(tmp_path / "web")
    users.add("alice", "alice-password")
    users.add("bob", "bob-password")
    with TestClient(create_app(s, llm=ScriptedProvider([FINAL]))) as c:
        assert c.get("/api/me").json()["auth"] is True
        assert c.get("/api/sessions").status_code == 401
        assert c.post("/api/login", json={"name": "alice", "password": "nope"}).status_code == 401

        assert c.post("/api/login", json={"name": "alice", "password": "alice-password"}).status_code == 200
        assert c.get("/api/me").json()["user"] == "alice"
        sid = c.post("/api/sessions").json()["id"]
        c.post(f"/api/sessions/{sid}/messages", json={"text": "alice 的问题"})
        c.app.state.hubs["alice"].runners[sid]._thread.join(5)
        assert (tmp_path / "sessions" / "alice" / sid / "session.jsonl").exists()

        c.post("/api/logout")
        assert c.get("/api/sessions").status_code == 401
        c.post("/api/login", json={"name": "bob", "password": "bob-password"})
        assert c.get("/api/sessions").json() == []
        assert c.get(f"/api/sessions/{sid}/events").status_code == 404


def test_Excel和CSV按工作表预览(client):
    from openpyxl import Workbook

    sid = client.post("/api/sessions").json()["id"]
    work = client.app.state.hubs[""].runners[sid].session.work_dir
    work.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    book.active.title = "各研究"
    book.active.append([None])                       # 表头前的空行跳过
    book.active.append(["研究", "MD"])
    book.active.append(["Chen 2017", -2.4])
    book.create_sheet("合并结果").append(["项目", "数值"])
    book.save(work / "r.xlsx")
    (work / "a.csv").write_text("x,y\n1,2\n3\n", encoding="utf-8")
    (work / "a.txt").write_text("hi", encoding="utf-8")

    sheets = client.get(f"/api/sessions/{sid}/sheets/r.xlsx").json()["sheets"]
    assert [s["title"] for s in sheets] == ["各研究", "合并结果"]
    assert sheets[0]["columns"] == ["研究", "MD"] and sheets[0]["rows"] == [["Chen 2017", -2.4]]
    assert sheets[1]["row_count"] == 0
    csv = client.get(f"/api/sessions/{sid}/sheets/a.csv").json()["sheets"][0]
    assert csv["rows"] == [["1", "2"], ["3", None]]
    assert client.get(f"/api/sessions/{sid}/sheets/a.txt").status_code == 415
    assert client.get(f"/api/sessions/{sid}/sheets/../session.jsonl").status_code == 404
