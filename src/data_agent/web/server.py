"""Web 界面的后端：FastAPI。事件用 SSE 推给浏览器（单向），发消息、回答、停止用普通 POST。

    python run_web.py            # 默认 http://127.0.0.1:8765

前端在仓库的 web/ 目录（React + Vite），npm run build 之后这里直接托管 web/dist。
只监听本机：没有登录，谁连上谁就能让 Agent 跑 SQL、跑代码。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.messages import Message
from ..core.provider import LLMProvider
from ..session import Session
from ..settings import Settings
from ..tools.sql.export_csv import export_result
from .runner import Busy, SessionRunner
from .serialize import table_json

# 同时开着几个会话。每个会话有自己的沙箱容器、MCP 子进程，多了关掉最久没用的（闲着的）
MAX_OPEN = 4
# SSE 多久没事件发一次注释行，免得代理、浏览器当连接死了
PING_S = 15
DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class Hub:
    """打开的会话。没打开过的会话按需从磁盘读。"""

    def __init__(self, settings: Settings, llm: LLMProvider | None = None) -> None:
        self.settings = settings
        self.llm = llm                    # 测试塞假模型
        self.base = Path(settings.sessions_dir)
        self.runners: dict[str, SessionRunner] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def create(self) -> SessionRunner:
        return self._open(Session.create(self.base))

    def get(self, session_id: str) -> SessionRunner:
        with self._lock:
            runner = self.runners.get(session_id)
        if runner is not None:
            return runner
        root = self.base / session_id
        if "/" in session_id or "\\" in session_id or not (root / "session.jsonl").exists():
            raise HTTPException(404, f"没有这个会话：{session_id}")
        return self._open(Session(root))

    def _open(self, session: Session) -> SessionRunner:
        runner = SessionRunner(self.settings, session, self.loop, self.llm)
        with self._lock:
            self.runners[runner.id] = runner
            idle = [r for r in self.runners.values() if not r.busy and r is not runner]
            for old in idle[: max(0, len(self.runners) - MAX_OPEN)]:
                del self.runners[old.id]
                old.close()
        return runner

    def sessions(self) -> list[dict[str, Any]]:
        """会话列表，最近的在前。标题是第一个问题；还没写过盘的新会话也列上。"""
        out = []
        for log in self.base.glob("*/session.jsonl"):
            title = _title(Session(log.parent))
            if title is not None:
                out.append({"id": log.parent.name, "title": title, "updated": log.stat().st_mtime})
        on_disk = {s["id"] for s in out}
        for runner in list(self.runners.values()):
            if runner.id not in on_disk:
                out.append({"id": runner.id, "title": "", "updated": 1e12})
        return sorted(out, key=lambda s: s["updated"], reverse=True)

    def close(self) -> None:
        for runner in self.runners.values():
            runner.close()


def _title(session: Session) -> str | None:
    """None = 读不了（别的版本写的日志），不列出来。"""
    try:
        entries = session.load()
    except Exception:  # noqa: BLE001 —— 坏掉的日志不能让整个列表出不来
        return None
    first = next((e for e in entries if isinstance(e, Message) and e.role == "user" and not e.meta.synthetic), None)
    if first is not None:
        text = first.content
    elif (turn := session.load_checkpoint()) is not None:
        text = turn.question                  # 第一轮还没跑完，历史里还没有
    else:
        return ""
    if text.startswith("[用户上传了文件") and "\n\n" in text:
        text = text.split("\n\n", 1)[1]
    if text.startswith("[用户指定按技能"):
        text = text.rsplit("\n\n", 1)[-1]
    return " ".join(text.split())[:60]


class Text(BaseModel):
    text: str = ""


class Approval(BaseModel):
    decision: str


def create_app(settings: Settings | None = None, llm: LLMProvider | None = None) -> FastAPI:
    hub = Hub(settings or Settings(), llm)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        hub.loop = asyncio.get_running_loop()
        yield
        await run_in_threadpool(hub.close)

    app = FastAPI(title="FinHelm", lifespan=lifespan)
    app.state.hub = hub

    def act(fn, *args) -> dict[str, bool]:
        try:
            fn(*args)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------------------ 会话
    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return hub.sessions()

    @app.post("/api/sessions")
    def new_session() -> dict[str, str]:
        return {"id": hub.create().id}

    @app.get("/api/sessions/{sid}/events")
    async def events(sid: str, request: Request) -> StreamingResponse:
        runner = await run_in_threadpool(hub.get, sid)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        runner.subscribe(queue)

        async def stream() -> AsyncIterator[str]:
            try:
                while not await request.is_disconnected():
                    try:
                        message = await asyncio.wait_for(queue.get(), PING_S)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(message, ensure_ascii=False, default=str)}\n\n"
            finally:
                runner.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/sessions/{sid}/messages")
    def send(sid: str, body: Text):
        if not body.text.strip():
            raise HTTPException(400, "消息是空的")
        return act(hub.get(sid).send, body.text.strip())

    @app.post("/api/sessions/{sid}/continue")
    def resume(sid: str, body: Text):
        return act(hub.get(sid).resume, body.text.strip())

    @app.post("/api/sessions/{sid}/stop")
    def stop(sid: str):
        return act(hub.get(sid).stop)

    @app.post("/api/sessions/{sid}/reset")
    def reset(sid: str):
        return act(hub.get(sid).reset)

    @app.post("/api/sessions/{sid}/compact")
    def compact(sid: str):
        return act(hub.get(sid).compact)

    @app.post("/api/sessions/{sid}/approvals/{rid}")
    def approve(sid: str, rid: str, body: Approval):
        if body.decision not in ("once", "session", "deny"):
            raise HTTPException(400, "decision 只能是 once / session / deny")
        try:
            hub.get(sid).approve(rid, body.decision)
        except KeyError as exc:
            raise HTTPException(404, "这个审批已经过期了") from exc
        return {"ok": True}

    @app.post("/api/sessions/{sid}/uploads")
    async def upload(sid: str, files: list[UploadFile]):
        runner = await run_in_threadpool(hub.get, sid)
        saved = [runner.upload(f.filename or "upload", await f.read()) for f in files]
        return {"files": [{"name": p.name, "size": p.stat().st_size} for p in saved]}

    # ------------------------------------------------------------ 结果、文件
    @app.get("/api/sessions/{sid}/results/{ref}")
    def result(sid: str, ref: str):
        table = hub.get(sid).results.get(ref)
        if table is None:
            raise HTTPException(404, f"没有结果 {ref}")
        return table_json(table, rows=None)

    @app.get("/api/sessions/{sid}/results/{ref}/csv")
    def result_csv(sid: str, ref: str):
        runner = hub.get(sid)
        table = runner.results.get(ref)
        if table is None:
            raise HTTPException(404, f"没有结果 {ref}")
        try:
            done = export_result(runner.app.db, table, runner.app.export_dir, f"{ref}.csv")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"导出失败：{type(exc).__name__}: {exc}") from exc
        return FileResponse(done.path, filename=done.path.name, media_type="text/csv")

    @app.get("/api/sessions/{sid}/files/{path:path}")
    def file(sid: str, path: str):
        root = hub.get(sid).session.work_dir.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, "没有这个文件")
        return FileResponse(target)

    if DIST.is_dir():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
    return app


def main() -> None:
    import uvicorn

    load_dotenv()
    parser = argparse.ArgumentParser(prog="python run_web.py", description="FinHelm Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="默认只让本机访问（没有登录）")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not DIST.is_dir():
        print(f"⚠️ 没找到前端 {DIST}：先 cd web && npm install && npm run build（开发时用 npm run dev）")
    print(f"FinHelm Web：http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
