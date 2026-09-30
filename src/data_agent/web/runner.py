"""一个打开的会话：一套 Application + 一个跑回合的后台线程 + 一群订阅事件的浏览器页面。

Agent 是同步的，一轮放在后台线程里跑；事件从那个线程出来，经 call_soon_threadsafe 交给各个页面的 asyncio 队列。
页面连上时先拿一份快照（历史 + agent.state），之后接着收事件（学 pi：状态 = 快照 + 之后的事件）。
"""

from __future__ import annotations

import asyncio
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Callable

from ..app import Application, build_application
from ..cli import pick_option
from ..core.agent import AwaitingUser
from ..core.events import Event, TextDelta, ToolFinished, TurnEnded
from ..core.messages import ToolCall
from ..core.provider import LLMProvider
from ..mcp import Decision, McpTool
from ..session import Session
from ..settings import Settings
from ..tools.sql.results import ResultStore
from .serialize import event_json, interrupted_json, plain, state_json, table_json, timeline

Message = dict[str, Any]

# MCP 审批等多久没人点就当拒绝
APPROVAL_TIMEOUT_S = 600


class Stopped(BaseException):
    """用户点了停止。和 Ctrl-C 一样是 BaseException：不能被「尽力而为」的 except Exception 吞掉。"""


class Busy(RuntimeError):
    """这个会话正在跑一轮。"""


class SessionRunner:
    def __init__(self, settings: Settings, session: Session, loop: asyncio.AbstractEventLoop,
                 llm: LLMProvider | None = None) -> None:
        self.session = session
        self.loop = loop
        self._subscribers: set[asyncio.Queue[Message]] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._approvals: dict[str, dict[str, Any]] = {}
        self._result_count = 0

        history = session.load() if session.log_path.exists() else []
        self.results = ResultStore(session.results_path)
        self.app: Application = build_application(
            settings, on_event=self._on_event, results=self.results, export_dir=session.exports_dir,
            work_dir=session.work_dir, ask_mcp=self._ask_mcp, llm=llm)
        self.app.restore(history, session.load_checkpoint())
        self.app.agent.checkpoint_hook = session.save_checkpoint
        self._result_count = len(self.results.refs())

    @property
    def id(self) -> str:
        return self.session.id

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ 订阅
    def subscribe(self, queue: asyncio.Queue[Message]) -> None:
        # 和 _publish 同一把锁：快照之前的事件已经在快照里，之后的一定进队列
        with self._lock:
            queue.put_nowait(self.snapshot())
            self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue[Message]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def _publish(self, message: Message) -> None:
        with self._lock:
            for queue in self._subscribers:
                self.loop.call_soon_threadsafe(queue.put_nowait, message)

    def snapshot(self) -> Message:
        agent = self.app.agent
        entries = list(agent.context.history)
        turn = agent.interrupted
        if turn is not None and not self.busy:
            entries += turn.entries           # 没跑完的那一轮不在正式历史里，也要看得到
        return {"type": "snapshot", "data": {
            "session": self.id,
            "items": timeline(entries),
            "interrupted": interrupted_json(turn),
            "results": self.results_json(),
            "approvals": [a["request"] for a in self._approvals.values()],
            "info": self.info(),
            "busy": self.busy,
        }, "state": self._state_with_context()}

    def _state_with_context(self) -> dict[str, Any]:
        """刚打开的会话还没请求过模型，状态里没有上下文用量，先用估算的顶上。"""
        state = state_json(self.app.agent.state)
        if not state["context_tokens"] and self.app.agent.context.history:
            state["context_tokens"] = self.app.agent.context_usage().tokens
            state["context_window"] = self.app.llm.context_window
        return state

    def info(self) -> dict[str, Any]:
        app = self.app
        return {"model": app.llm.model, "context_window": app.llm.context_window,
                "tools": [t.name for t in app.tools], "skills": [s.name for s in app.skills],
                "mcp": sorted({t.server for t in app.mcp_tools}), "sandbox": bool(app.sandboxes),
                "database": app.db is not None}

    def results_json(self) -> list[dict[str, Any]]:
        return [table_json(t) for ref in self.results.refs() if (t := self.results.get(ref))]

    # ------------------------------------------------------------ 事件
    def _on_event(self, event: Event) -> None:
        if self._stop and not isinstance(event, TurnEnded):
            self._stop = False
            raise Stopped("用户停止")
        message = event_json(event, self.file_url)
        if not isinstance(event, TextDelta):          # 每个增量都带状态太重，前端自己把字拼上
            message["state"] = state_json(self.app.agent.state)
        self._publish(message)
        if isinstance(event, ToolFinished) and len(self.results.refs()) != self._result_count:
            self._result_count = len(self.results.refs())
            self._publish({"type": "results", "data": self.results_json()})

    def file_url(self, path: Path) -> str | None:
        """work 目录下的文件（图）给浏览器的地址。目录外的不给。"""
        try:
            rel = path.resolve().relative_to(self.session.work_dir.resolve())
        except ValueError:
            return None
        # 同名的图会被重画覆盖，带上修改时间，浏览器才不会拿缓存里的旧图
        version = path.stat().st_mtime_ns if path.exists() else 0
        return f"/api/sessions/{self.id}/files/{rel.as_posix()}?v={version}"

    # ------------------------------------------------------------ 跑一轮
    def _start(self, body: Callable[[], Any]) -> None:
        with self._lock:
            if self.busy:
                raise Busy("这个会话正在跑，先等它结束或者点停止")
            self._stop = False
            self._thread = threading.Thread(target=self._work, args=(body,), daemon=True)
            self._thread.start()

    def _work(self, body: Callable[[], Any]) -> None:
        try:
            outcome = body()
            if outcome == []:                         # compact() 什么都没做
                self._notice("没有可整理的内容（压缩至少要保留当前这一轮，对话还太短）")
        except (AwaitingUser, Stopped):
            pass                                      # 问题 / 停止都已经由 TurnEnded 带出去了
        except BaseException as exc:  # noqa: BLE001
            # 有进度的话「没跑完」卡片已经说了原因，不重复
            if self.app.agent.interrupted is None:
                self._publish({"type": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}})
        finally:
            # 和 CLI 一样：先写历史再写检查点
            self.session.sync(self.app.agent.context.history)
            self.session.save_checkpoint(self.app.agent.interrupted)
            self._publish({"type": "idle", "data": {"interrupted": interrupted_json(self.app.agent.interrupted)},
                           "state": state_json(self.app.agent.state)})

    def _notice(self, text: str) -> None:
        self._publish({"type": "notice", "data": {"text": text}})

    def send(self, text: str) -> None:
        """用户发了一句话：停在提问上就是回答，否则是新问题（上一轮没跑完的进度作废）。"""
        agent = self.app.agent
        turn = agent.interrupted
        if turn is not None and turn.pending is not None:
            answer = pick_option(text, turn.pending.options)
            self._start(lambda: agent.resume(answer))
        else:
            self._start(lambda: agent.run(self.app.with_uploads(text)))

    def resume(self, text: str = "") -> None:
        if self.app.agent.interrupted is None:
            raise ValueError("没有暂停或出错的回合可以接着跑")
        self._start(lambda: self.app.agent.resume(text))

    def compact(self) -> None:
        self._start(self.app.agent.compact)

    def stop(self) -> None:
        """停在下一个事件上（模型每吐一个字就是一个事件）。正在跑的工具能取消的就取消
        （沙箱杀内核、SQL 取消查询、MCP 不再等），它的结果是一条「用户中断」，之后照常停下。"""
        if self.busy:
            self._stop = True
            for approval in list(self._approvals.values()):
                approval["event"].set()               # 在等审批的直接当拒绝
            self.app.agent.cancel_tool()

    def reset(self) -> None:
        if self.busy:
            raise Busy("正在跑，先停下来再清空")
        self.app.reset()
        self.session.sync(self.app.agent.context.history)
        self.session.save_checkpoint(None)
        self._result_count = len(self.results.refs())

    def upload(self, name: str, data: bytes) -> Path:
        """上传的文件放进 work/inputs/，下一条消息告诉模型（同 CLI 的 /attach）。"""
        inputs = self.session.work_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        path = inputs / _safe_name(name)
        path.write_bytes(data)
        self.app.pending_uploads.append(path)
        return path

    # ------------------------------------------------------------ MCP 审批
    def _ask_mcp(self, call: ToolCall, tool: McpTool) -> Decision:
        """在跑回合的线程里等：发请求给页面，页面点了再放行。和 ask_user 不同，这里不结束这一轮。"""
        rid = secrets.token_hex(4)
        request = {"id": rid, "server": tool.server, "tool": tool.remote_name, "arguments": plain(call.arguments)}
        approval = {"event": threading.Event(), "decision": "deny", "request": request}
        self._approvals[rid] = approval
        self._publish({"type": "approval", "data": request})
        approval["event"].wait(APPROVAL_TIMEOUT_S)
        del self._approvals[rid]
        decision: Decision = approval["decision"]
        self._publish({"type": "approval_done", "data": {"id": rid, "decision": decision}})
        return decision

    def approve(self, rid: str, decision: Decision) -> None:
        approval = self._approvals.get(rid)
        if approval is None:
            raise KeyError(rid)
        approval["decision"] = decision
        approval["event"].set()

    def close(self) -> None:
        self.stop()
        self.app.close()


def _safe_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    return re.sub(r'[<>:"/|?*\x00-\x1f]', "_", name).strip(". ") or "upload"
