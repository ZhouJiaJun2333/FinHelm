"""MCP 客户端（stdio 传输）：起一个子进程，用 JSON-RPC 2.0 和它说话，每行一条消息。

    initialize → notifications/initialized → tools/list（分页）→ tools/call …

一个读线程收 stdout：响应按 id 交给等它的那次调用；服务器反过来发的请求（ping 等）就地回；通知忽略。
stderr 也要有线程读掉：管道缓冲满了，子进程会卡在写日志上。几个 Agent 可以共用一个客户端：写有锁，响应按 id 各回各家。
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "finhelm", "version": "0.1"}


class McpError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    auto_approve: frozenset[str] = frozenset()      # 不用问用户就能调的工具（服务器里的原名）
    cwd: str | None = None


class McpClient:
    def __init__(self, config: ServerConfig, timeout: float = 60.0) -> None:
        self.config = config
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self.instructions = ""                       # 服务器在握手时给的用法说明
        self._proc: subprocess.Popen | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, queue.Queue] = {}
        self._write_lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=30)

    # ------------------------------------------------------------ 启动、关闭
    def start(self) -> "McpClient":
        command = shutil.which(self.config.command) or self.config.command     # Windows 上 npx 实际是 npx.cmd
        try:
            self._proc = subprocess.Popen(
                [command, *self.config.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, **self.config.env}, cwd=self.config.cwd)
        except OSError as exc:
            raise McpError(f"MCP 服务器 {self.config.name} 起不来：{exc}") from exc
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        result = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                             "clientInfo": CLIENT_INFO})
        self.server_info = result.get("serverInfo") or {}
        self.instructions = result.get("instructions") or ""
        self.notify("notifications/initialized")
        return self

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.close()                  # 关掉 stdin 是 stdio 服务器的正常退出信号
            self._proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()

    # ------------------------------------------------------------ 协议
    def list_tools(self) -> list[dict[str, Any]]:
        tools, cursor = [], None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += result.get("tools") or []
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout)

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> dict:
        request_id = next(self._ids)
        box: queue.Queue = queue.Queue(maxsize=1)
        self._pending[request_id] = box
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
            try:
                msg = box.get(timeout=timeout or self.timeout)
            except queue.Empty:
                self.notify("notifications/cancelled", {"requestId": request_id, "reason": "timeout"})
                raise McpError(f"MCP 服务器 {self.config.name} 的 {method} 超时（{timeout or self.timeout:g} 秒）") from None
        finally:
            self._pending.pop(request_id, None)
        if "error" in msg:
            err = msg["error"]
            raise McpError(f"MCP 服务器 {self.config.name} 报错（{err.get('code')}）：{err.get('message')}")
        return msg.get("result") or {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise McpError(f"MCP 服务器 {self.config.name} 没在运行。{self._stderr_tail()}")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except OSError as exc:
                raise McpError(f"写不进 MCP 服务器 {self.config.name}：{exc}。{self._stderr_tail()}") from exc

    # ------------------------------------------------------------ 读线程
    def _read_stdout(self) -> None:
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(f"[stdout 不是 JSON] {line[:200]!r}")     # 服务器往 stdout 打了日志
                continue
            if "method" in msg:
                if "id" in msg:
                    self._answer_server_request(msg)
                continue                              # 通知（进度、日志、列表变了）先不处理
            box = self._pending.get(msg.get("id"))
            if box is not None:
                box.put(msg)
        self._stderr_reader.join(timeout=1)           # 让 stderr 读完：崩溃原因常在最后一行
        gone = {"error": {"code": -32000, "message": f"服务器退出了。{self._stderr_tail()}"}}
        for box in list(self._pending.values()):      # 还在等的调用别干等到超时
            box.put(gone)

    def _answer_server_request(self, msg: dict[str, Any]) -> None:
        """服务器也能发请求。我们只回 ping；采样、roots 这些没声明支持，回「没有这个方法」。"""
        if msg["method"] == "ping":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not found"}}
        try:
            self._send(reply)
        except McpError:
            pass

    def _read_stderr(self) -> None:
        for line in self._proc.stderr:
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    def _stderr_tail(self) -> str:
        tail = "\n".join(list(self._stderr)[-5:])
        return f"stderr 最后几行：\n{tail}" if tail else ""
