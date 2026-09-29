"""把事件、状态、历史换成浏览器要的 JSON。

历史和事件最后都变成同一种「时间线条目」（user / assistant / tool / notice），前端只认这一种：
打开会话时用历史建出条目，之后来的事件接着改这些条目。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ..core.agent import InterruptedTurn
from ..core.context import Entry, Marker
from ..core.events import Event, ToolFinished
from ..core.messages import Message
from ..core.state import AgentState
from ..tools.sandbox import Execution
from ..tools.sql.results import StoredResult

# 界面上一条工具结果最多带多少字、一张表的预览带几行
CONTENT_CHARS = 4000
PREVIEW_ROWS = 20

FileUrl = Callable[[Path], str | None]


def plain(obj: Any) -> Any:
    """递归换成 JSON 能直接写的东西。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (dt.date, dt.datetime, dt.time)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [plain(v) for v in obj]
    return str(obj)


def table_json(table: StoredResult, rows: int | None = PREVIEW_ROWS) -> dict[str, Any]:
    r = table.result
    return {"ref": table.ref, "title": table.title, "source": table.source, "sql": table.sql,
            "columns": r.columns, "rows": plain(r.rows if rows is None else r.rows[:rows]),
            "row_count": r.row_count, "truncated": r.truncated}


def details_json(details: Any, file_url: FileUrl) -> dict[str, Any] | None:
    """ToolOutput.details（只给界面的结构化结果）。认不得的就不发，界面显示模型看到的那份文字。"""
    if isinstance(details, StoredResult):
        return {"kind": "table", **table_json(details)}
    if isinstance(details, Execution):
        return {"kind": "execution", "output": details.output, "value": details.value, "error": details.error,
                "figures": [u for p in details.figures if (u := file_url(p))]}
    return None


def event_json(event: Event, file_url: FileUrl) -> dict[str, Any]:
    data = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
    if isinstance(event, ToolFinished):
        data["details"] = details_json(event.details, file_url)
        data["content"] = _cap(event.content)
    return {"type": type(event).__name__, "data": plain(data)}


def state_json(state: AgentState) -> dict[str, Any]:
    return {**plain(state), "running": state.running}


def interrupted_json(turn: InterruptedTurn | None) -> dict[str, Any] | None:
    if turn is None:
        return None
    pending = turn.pending
    return {"reason": turn.reason, "steps": turn.steps, "question": turn.question,
            "pending": None if pending is None else
            {"call_id": pending.call_id, "question": pending.question, "options": list(pending.options)}}


def timeline(entries: list[Entry] | tuple[Entry, ...]) -> list[dict[str, Any]]:
    """历史 → 时间线条目。工具调用和它的结果并成一条 tool。"""
    items: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    for e in entries:
        if isinstance(e, Marker):
            items.append({"kind": "notice", "text": e.describe(), "level": "info"})
            continue
        if not isinstance(e, Message):
            continue
        if e.role == "user":
            # Agent 自己补的（推动、收尾提示、重启提醒）不是用户说的，不显示
            if not e.meta.synthetic:
                items.append({"kind": "user", "text": e.content})
        elif e.role == "assistant":
            if e.content:
                items.append({"kind": "assistant", "text": e.content, "thinking": _thinking(e.raw),
                              "streaming": False})
            for c in e.tool_calls:
                tool = {"kind": "tool", "call_id": c.id, "name": c.name, "arguments": plain(c.arguments),
                        "status": "running", "content": "", "details": None, "elapsed_ms": 0}
                tools[c.id] = tool
                items.append(tool)
        elif e.role == "tool" and (tool := tools.get(e.tool_call_id or "")) is not None:
            tool.update(status="error" if e.is_error else "done", content=_cap(e.content))
    return items


def _thinking(raw: Any) -> str:
    """原生消息里的思考过程（DeepSeek 的 reasoning_content、Anthropic 的 thinking 块）。"""
    if isinstance(raw, dict):
        return raw.get("reasoning_content") or ""
    if isinstance(raw, list):
        return "\n".join(b.get("thinking", "") for b in raw if isinstance(b, dict) and b.get("type") == "thinking")
    return ""


def _cap(text: str) -> str:
    return text if len(text) <= CONTENT_CHARS else text[:CONTENT_CHARS] + f"\n…（共 {len(text)} 字）"
