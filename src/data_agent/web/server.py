"""Web 界面的后端：FastAPI。事件用 SSE 推给浏览器（单向），发消息、回答、停止用普通 POST。

    python run_web.py                   # 默认 http://127.0.0.1:8765
    python run_web.py adduser 名字      # 建账号之后就要登录（见 auth.py）

前端在仓库的 web/ 目录（React + Vite），npm run build 之后这里直接托管 web/dist。
没建账号时只许监听本机：谁连上谁就能让 Agent 跑 SQL、跑代码。
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.messages import Message
from ..core.provider import LLMProvider
from ..session import Session
from ..settings import Settings
from ..tools.sql.export_csv import export_result
from .auth import COOKIE, TTL_S, Users
from .runner import Busy, SessionRunner
from .serialize import sheets_json, table_json

# 同时开着几个会话。每个会话有自己的沙箱容器、MCP 子进程，多了关掉最久没用的（闲着的）
MAX_OPEN = 4
# 右侧面板能按表格预览的文件
SHEET_SUFFIXES = {".xlsx", ".xlsm", ".csv"}
# SSE 多久没事件发一次注释行，免得代理、浏览器当连接死了
PING_S = 15
DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
SESSION_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def user_settings(settings: Settings, user: str) -> Settings:
    """每个账号自己的会话目录和长期记忆；单用户模式（user 为空）照原样。"""
    if not user:
        return settings
    return settings.model_copy(update={
        "sessions_dir": str(Path(settings.sessions_dir) / user),
        "memory_dir": str(Path(settings.memory_dir).expanduser() / "users" / user),
    })


class Hub:
    """一个人打开的会话。没打开过的会话按需从磁盘读。"""

    def __init__(self, settings: Settings, loop: asyncio.AbstractEventLoop | None,
                 llm: LLMProvider | None = None) -> None:
        self.settings = settings
        self.loop = loop
        self.llm = llm                    # 测试塞假模型
        self.base = Path(settings.sessions_dir)
        self.runners: dict[str, SessionRunner] = {}
        self._lock = threading.Lock()

    def create(self) -> SessionRunner:
        return self._open(Session.create(self.base))

    def get(self, session_id: str) -> SessionRunner:
        with self._lock:
            runner = self.runners.get(session_id)
        if runner is not None:
            return runner
        return self._open(Session(self._root(session_id)))

    def _root(self, session_id: str) -> Path:
        root = self.base / session_id
        if not SESSION_ID.fullmatch(session_id) or not (root / "session.jsonl").exists():
            raise HTTPException(404, f"没有这个会话：{session_id}")
        return root

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
        """会话列表，最近的在前。还没写过盘的新会话也列上。"""
        out = []
        busy = {r.id for r in list(self.runners.values()) if r.busy}
        for log in self.base.glob("*/session.jsonl"):
            title = _title(Session(log.parent))
            if title is not None:
                out.append({"id": log.parent.name, "title": title, "updated": log.stat().st_mtime,
                            "busy": log.parent.name in busy})
        on_disk = {s["id"] for s in out}
        for runner in list(self.runners.values()):
            if runner.id not in on_disk:
                out.append({"id": runner.id, "title": runner.session.title(), "updated": time.time(),
                            "busy": runner.busy})
        return sorted(out, key=lambda s: s["updated"], reverse=True)

    def rename(self, session_id: str, title: str) -> None:
        with self._lock:
            runner = self.runners.get(session_id)
        session = runner.session if runner is not None else Session(self._root(session_id))
        session.rename(title)

    def delete(self, session_id: str) -> None:
        with self._lock:
            runner = self.runners.get(session_id)
            if runner is not None and runner.busy:
                raise Busy("正在跑，先停下来再删除")
            self.runners.pop(session_id, None)
        if runner is not None:
            runner.close()
            session = runner.session
        else:
            session = Session(self._root(session_id))
        session.trash()

    def trashed(self) -> list[dict[str, Any]]:
        out = [{"id": s.id, "title": _title(s) or "", "deleted": s.root.stat().st_mtime}
               for s in Session.trashed(self.base)]
        return sorted(out, key=lambda s: s["deleted"], reverse=True)

    def restore(self, session_id: str) -> None:
        if not SESSION_ID.fullmatch(session_id):
            raise HTTPException(404, "没有这个会话")
        try:
            Session.restore(self.base, session_id)
        except (FileNotFoundError, FileExistsError) as exc:
            raise HTTPException(404, str(exc)) from exc

    def close(self) -> None:
        for runner in self.runners.values():
            runner.close()


def _title(session: Session) -> str | None:
    """改过名就用改的；否则是第一个问题。None = 读不了（别的版本写的日志），不列出来。"""
    try:
        entries = session.load()
    except Exception:  # noqa: BLE001 —— 坏掉的日志不能让整个列表出不来
        return None
    if named := session.title():
        return named
    first = next((e for e in entries if isinstance(e, Message) and e.role == "user" and not e.meta.synthetic), None)
    # 第一轮还没跑完的，问题只在检查点里
    text = first.content if first is not None else session.checkpoint_question()
    if text.startswith("[用户上传了文件") and "\n\n" in text:
        text = text.split("\n\n", 1)[1]
    if text.startswith("[用户指定按技能"):
        text = text.rsplit("\n\n", 1)[-1]
    return " ".join(text.split())[:60]


class Text(BaseModel):
    text: str = ""


class Approval(BaseModel):
    decision: str


class Login(BaseModel):
    name: str
    password: str


class Title(BaseModel):
    title: str


def create_app(settings: Settings | None = None, llm: LLMProvider | None = None) -> FastAPI:
    settings = settings or Settings()
    users = Users(Path(settings.web_dir).expanduser())
    # 启动时有账号就开登录。之后再建第一个账号要重启才生效，免得正在用的人突然被踢出去
    auth = bool(users.names())
    hubs: dict[str, Hub] = {}
    hubs_lock = threading.Lock()
    loop: dict[str, asyncio.AbstractEventLoop] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        loop["main"] = asyncio.get_running_loop()
        yield
        await run_in_threadpool(lambda: [h.close() for h in hubs.values()])

    app = FastAPI(title="FinHelm", lifespan=lifespan)
    app.state.hubs = hubs

    def current_user(request: Request) -> str:
        if not auth:
            return ""
        name = users.check(request.cookies.get(COOKIE))
        if name is None:
            raise HTTPException(401, "请先登录")
        return name

    def hub(user: str = Depends(current_user)) -> Hub:
        with hubs_lock:
            if user not in hubs:
                hubs[user] = Hub(user_settings(settings, user), loop.get("main"), llm)
            return hubs[user]

    def act(fn, *args) -> dict[str, bool]:
        try:
            fn(*args)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------------------ 登录
    @app.get("/api/me")
    def me(request: Request) -> dict[str, Any]:
        user = users.check(request.cookies.get(COOKIE)) if auth else ""
        return {"auth": auth, "user": user, "project": Path(settings.project_dir).resolve().name,
                "model": settings.openai_model if settings.provider == "openai" else settings.anthropic_model}

    @app.post("/api/login")
    def login(body: Login, response: Response):
        if not users.verify(body.name, body.password):
            time.sleep(0.5)                        # 慢一点，猜密码划不来
            raise HTTPException(401, "用户名或密码不对")
        response.set_cookie(COOKIE, users.token(body.name), max_age=TTL_S, httponly=True, samesite="strict")
        return {"ok": True}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE)
        return {"ok": True}

    # ------------------------------------------------------------ 会话
    @app.get("/api/sessions")
    def list_sessions(h: Hub = Depends(hub)) -> list[dict[str, Any]]:
        return h.sessions()

    @app.post("/api/sessions")
    def new_session(h: Hub = Depends(hub)) -> dict[str, str]:
        return {"id": h.create().id}

    @app.patch("/api/sessions/{sid}")
    def rename(sid: str, body: Title, h: Hub = Depends(hub)):
        title = " ".join(body.title.split())[:80]
        if not title:
            raise HTTPException(400, "标题是空的")
        return act(h.rename, sid, title)

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str, h: Hub = Depends(hub)):
        return act(h.delete, sid)

    @app.get("/api/trash")
    def trash(h: Hub = Depends(hub)) -> list[dict[str, Any]]:
        return h.trashed()

    @app.post("/api/trash/{sid}/restore")
    def restore(sid: str, h: Hub = Depends(hub)):
        return act(h.restore, sid)

    @app.get("/api/sessions/{sid}/events")
    async def events(sid: str, request: Request, h: Hub = Depends(hub)) -> StreamingResponse:
        runner = await run_in_threadpool(h.get, sid)
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
    def send(sid: str, body: Text, h: Hub = Depends(hub)):
        if not body.text.strip():
            raise HTTPException(400, "消息是空的")
        return act(h.get(sid).send, body.text.strip())

    @app.post("/api/sessions/{sid}/continue")
    def resume(sid: str, body: Text, h: Hub = Depends(hub)):
        return act(h.get(sid).resume, body.text.strip())

    @app.post("/api/sessions/{sid}/stop")
    def stop(sid: str, h: Hub = Depends(hub)):
        return act(h.get(sid).stop)

    @app.post("/api/sessions/{sid}/reset")
    def reset(sid: str, h: Hub = Depends(hub)):
        return act(h.get(sid).reset)

    @app.post("/api/sessions/{sid}/compact")
    def compact(sid: str, h: Hub = Depends(hub)):
        return act(h.get(sid).compact)

    @app.post("/api/sessions/{sid}/approvals/{rid}")
    def approve(sid: str, rid: str, body: Approval, h: Hub = Depends(hub)):
        if body.decision not in ("once", "session", "deny"):
            raise HTTPException(400, "decision 只能是 once / session / deny")
        try:
            h.get(sid).approve(rid, body.decision)
        except KeyError as exc:
            raise HTTPException(404, "这个审批已经过期了") from exc
        return {"ok": True}

    @app.post("/api/sessions/{sid}/uploads")
    async def upload(sid: str, files: list[UploadFile], h: Hub = Depends(hub)):
        runner = await run_in_threadpool(h.get, sid)
        saved = [runner.upload(f.filename or "upload", await f.read()) for f in files]
        return {"files": [{"name": p.name, "size": p.stat().st_size} for p in saved]}

    # ------------------------------------------------------------ 结果、文件
    @app.get("/api/sessions/{sid}/results/{ref}")
    def result(sid: str, ref: str, h: Hub = Depends(hub)):
        table = h.get(sid).results.get(ref)
        if table is None:
            raise HTTPException(404, f"没有结果 {ref}")
        return table_json(table, rows=None)

    @app.get("/api/sessions/{sid}/results/{ref}/csv")
    def result_csv(sid: str, ref: str, h: Hub = Depends(hub)):
        runner = h.get(sid)
        table = runner.results.get(ref)
        if table is None:
            raise HTTPException(404, f"没有结果 {ref}")
        try:
            done = export_result(runner.app.db, table, runner.app.export_dir, f"{ref}.csv")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"导出失败：{type(exc).__name__}: {exc}") from exc
        return FileResponse(done.path, filename=done.path.name, media_type="text/csv")

    def work_file(h: Hub, sid: str, path: str) -> Path:
        root = h.get(sid).session.work_dir.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, "没有这个文件")
        return target

    @app.get("/api/sessions/{sid}/files/{path:path}")
    def file(sid: str, path: str, h: Hub = Depends(hub)):
        return FileResponse(work_file(h, sid, path))

    @app.get("/api/sessions/{sid}/sheets/{path:path}")
    def sheets(sid: str, path: str, h: Hub = Depends(hub)):
        target = work_file(h, sid, path)
        if target.suffix.lower() not in SHEET_SUFFIXES:
            raise HTTPException(415, "只能预览 xlsx、xlsm、csv")
        try:
            return {"sheets": sheets_json(target)}
        except Exception as exc:          # 坏文件、加密的工作簿
            raise HTTPException(422, f"读不了这个文件：{exc}") from exc

    if DIST.is_dir():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
    return app


# ------------------------------------------------------------------ 命令行
def main() -> None:
    import uvicorn

    load_dotenv()
    parser = argparse.ArgumentParser(prog="python run_web.py", description="FinHelm Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="默认只让本机访问")
    parser.add_argument("--port", type=int, default=8765)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("adduser", help="建账号或改密码").add_argument("name")
    sub.add_parser("deluser", help="删账号（会话和记忆留在磁盘上）").add_argument("name")
    sub.add_parser("users", help="列出账号")
    args = parser.parse_args()

    settings = Settings()
    users = Users(Path(settings.web_dir).expanduser())
    if args.command == "adduser":
        password = getpass.getpass(f"{args.name} 的密码：")
        if password != getpass.getpass("再输一遍："):
            sys.exit("两次不一样")
        try:
            users.add(args.name, password)
        except ValueError as exc:
            sys.exit(str(exc))
        print(f"好了。账号在 {users.path}，重启 run_web.py 后生效")
        return
    if args.command == "deluser":
        print("删了" if users.remove(args.name) else "没有这个账号")
        return
    if args.command == "users":
        print("\n".join(users.names()) or "还没有账号（单用户模式，只许本机访问）")
        return

    if args.host not in LOOPBACK and not users.names():
        sys.exit("对外监听之前先建账号：python run_web.py adduser 名字（不然谁连上都能让 Agent 跑代码）")
    if not DIST.is_dir():
        print(f"⚠️ 没找到前端 {DIST}：先 cd web && npm install && npm run build（开发时用 npm run dev）")
    print(f"FinHelm Web：http://{args.host}:{args.port}" + ("（要登录）" if users.names() else ""))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")
