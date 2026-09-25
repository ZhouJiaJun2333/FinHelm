"""历史条目（消息 + 标记）↔ JSON。decode(encode(x)) == x，tests/test_session.py 对每种标记都验。

新标记不用在这里登记，但字段类型要在 _to_json / _from_json 认得的范围里（str、数字、bool、frozenset、Usage）。
"""

from __future__ import annotations

import sys
from dataclasses import fields
from typing import Any, get_origin, get_type_hints

from ..core.context import Entry, Marker
from ..core.messages import Message, MessageMeta, ToolCall, Usage


def encode(entry: Entry) -> dict[str, Any]:
    if isinstance(entry, Message):
        return _encode_message(entry)
    return {"type": "marker", "kind": type(entry).__name__,
            **{f.name: _to_json(getattr(entry, f.name)) for f in fields(entry)}}


def decode(data: dict[str, Any]) -> Entry:
    if data["type"] == "message":
        return _decode_message(data)
    if data["type"] == "marker":
        cls = _marker_classes().get(data["kind"])
        if cls is None:
            raise ValueError(f"不认识的标记：{data['kind']}（日志是更新的版本写的？）")
        hints = get_type_hints(cls)
        # 旧日志里没有的字段（标记后来加的）用默认值
        return cls(**{f.name: _from_json(data[f.name], hints[f.name])
                      for f in fields(cls) if f.name in data})
    raise ValueError(f"不认识的记录类型：{data['type']}")


# ------------------------------------------------------------------ 消息
# 只写非默认值，日志打开来能直接读
def _encode_message(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "message", "role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.is_error:
        d["is_error"] = True
    if m.raw is not None:
        d["raw"] = m.raw
    meta = {f.name: _to_json(getattr(m.meta, f.name)) for f in fields(MessageMeta)
            if getattr(m.meta, f.name) != f.default}
    if meta:
        d["meta"] = meta
    return d


def _decode_message(d: dict[str, Any]) -> Message:
    hints = get_type_hints(MessageMeta)
    meta = MessageMeta(**{k: _from_json(v, hints[k]) for k, v in d.get("meta", {}).items()})
    return Message(
        role=d["role"],
        content=d["content"],
        tool_calls=[ToolCall(c["id"], c["name"], c["arguments"]) for c in d.get("tool_calls", [])],
        tool_call_id=d.get("tool_call_id"),
        is_error=d.get("is_error", False),
        raw=d.get("raw"),
        meta=meta,
    )


# ------------------------------------------------------------ 字段类型
def _to_json(value: Any) -> Any:
    if isinstance(value, Usage):
        return {f.name: getattr(value, f.name) for f in fields(Usage)}
    if isinstance(value, frozenset):
        return sorted(value)          # 排序：同样的集合写出同样的一行
    return value


def _from_json(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    if hint is Usage or (isinstance(value, dict) and Usage in getattr(hint, "__args__", ())):
        return Usage(**value)
    if get_origin(hint) is frozenset:
        return frozenset(value)
    return value


def _marker_classes() -> dict[str, type[Marker]]:
    """Marker 的所有子类（按类名）。

    只认模块里真正叫这个名字的类：@dataclass(slots=True) 会另造一个新类，旧类还挂在
    __subclasses__() 里，拿它解码出来的标记 isinstance 认不出来。
    """
    found: dict[str, type[Marker]] = {}
    todo = list(Marker.__subclasses__())
    while todo:
        cls = todo.pop()
        todo += cls.__subclasses__()
        if getattr(sys.modules.get(cls.__module__), cls.__name__, None) is cls:
            found[cls.__name__] = cls
    return found
