"""Web 后端：会话 runner（后台线程跑回合、事件推给订阅者）和 HTTP 接口。不起浏览器、不连模型。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from data_agent.core.messages import LLMResponse, ToolCall, Usage
from data_agent.session import Session
from data_agent.settings import Settings
from data_agent.web.runner import SessionRunner
from data_agent.web.server import create_app

from fakes import ScriptedProvider

FINAL = LLMResponse(text="华东最高", stop_reason="end_turn", usage=Usage(input=100, output=10))


def settings(tmp_path, **kw) -> Settings:
    return Settings(provider="openai", openai_api_key="x", database_url="", memory_enabled=False,
                    mcp_enabled=False, python_sandbox=False, r_sandbox=False, docs_dirs="", ask_user=True,
                    project_dir=str(tmp_path), sessions_dir=str(tmp_path / "sessions"), **kw)


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
    runner = client.app.state.hub.runners[sid]
    runner._thread.join(5)
    sessions = client.get("/api/sessions").json()
    assert sessions[0] == {**sessions[0], "id": sid, "title": "哪个大区最高"}


def test_没有的会话404_文件不能跳出work目录(client):
    assert client.get("/api/sessions/不存在/results/r1").status_code == 404
    sid = client.post("/api/sessions").json()["id"]
    work = client.app.state.hub.runners[sid].session.work_dir
    (work / "figures").mkdir(parents=True)
    (work / "figures" / "a.png").write_bytes(b"png")
    assert client.get(f"/api/sessions/{sid}/files/figures/a.png").content == b"png"
    assert client.get(f"/api/sessions/{sid}/files/../session.jsonl").status_code == 404
    assert client.get(f"/api/sessions/{sid}/files/..%2F..%2F.env").status_code == 404
