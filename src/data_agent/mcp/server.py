"""FinHelm 的 MCP 服务器（stdio）：把知识库的 list_docs / search_docs / read_doc 给任何 MCP 客户端用。

    python -m data_agent.mcp.server --docs-dir data/financebench/pdfs
    Claude Code：claude mcp add finhelm -- python -m data_agent.mcp.server --docs-dir D:/.../pdfs

索引、模型的设置（RAG_DIR、RAG_EMBEDDER、RAG_RERANKER）和 Agent 一样从 .env 读，用的是同一份索引。
stdout 只能写协议消息：别的输出（加载模型的进度条、print）都转到 stderr。请求按顺序一个个处理。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import IO, Any

from ..core.tools import Tool
from .client import PROTOCOL_VERSION

SERVER_INFO = {"name": "finhelm", "version": "0.1"}


def serve(tools: list[Tool], instructions: str, stdin: IO[bytes], stdout: IO[bytes]) -> None:
    by_name = {t.name: t for t in tools}
    for line in stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _write(stdout, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        if "method" not in msg or "id" not in msg:
            continue                                  # 通知（initialized、cancelled）和客户端的响应：不用回
        try:
            reply = {"result": _handle(msg["method"], msg.get("params") or {}, by_name, instructions)}
        except _RpcError as exc:
            reply = {"error": {"code": exc.code, "message": str(exc)}}
        _write(stdout, {"jsonrpc": "2.0", "id": msg["id"], **reply})


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _handle(method: str, params: dict[str, Any], tools: dict[str, Tool], instructions: str) -> dict[str, Any]:
    if method == "initialize":
        return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO,
                **({"instructions": instructions} if instructions else {})}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.schema()["parameters"],
                           "annotations": {"readOnlyHint": t.rerunnable}} for t in tools.values()]}
    if method == "tools/call":
        tool = tools.get(params.get("name", ""))
        if tool is None:
            raise _RpcError(-32602, f"没有叫 {params.get('name')} 的工具，有：{', '.join(tools)}")
        out = tool.execute(params.get("arguments") or {})
        return {"content": [{"type": "text", "text": out.content}], "isError": out.is_error}
    raise _RpcError(-32601, f"Method not found: {method}")


def _write(stdout: IO[bytes], msg: dict[str, Any]) -> None:
    stdout.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
    stdout.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", action="append", required=True, help="文档目录，可以写几次")
    args = ap.parse_args(argv)

    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr                           # 之后谁 print 都进 stderr，不弄脏协议
    from dotenv import load_dotenv

    load_dotenv()
    from ..app import _collections
    from ..prompts import docs_step
    from ..rag import SearchSpec
    from ..settings import Settings
    from ..tools.docs import ListDocsTool, ReadDocTool, SearchDocsTool

    settings = Settings(docs_dirs=";".join(args.docs_dir))
    collections = _collections(settings)
    retrievers = ("bm25", "dense") if settings.rag_embedder else ("bm25",)
    tools = [ListDocsTool(collections), SearchDocsTool(collections, SearchSpec(retrievers, reranker=settings.rag_reranker)),
             ReadDocTool(collections)]
    for c in collections:                             # 后台先把索引加载好：第一次调用不用等十几秒
        threading.Thread(target=c.index, daemon=True).start()
    instructions = docs_step([(c.name, len(c.files())) for c in collections])
    serve(tools, instructions, sys.stdin.buffer, protocol_out)


if __name__ == "__main__":
    main()
