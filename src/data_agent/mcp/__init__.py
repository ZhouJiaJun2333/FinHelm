"""MCP（Model Context Protocol）：外部工具服务器接进来当工具用，也能把我们的知识库当服务器给别人用。

    client.py   手写的 stdio 客户端（JSON-RPC 2.0，一行一条消息）
    tools.py    外部工具 → 我们的 Tool：名字 mcp__服务器__工具、描述截断、参数按 JSON Schema 校验
    server.py   FinHelm 的 MCP 服务器：list_docs / search_docs / read_doc

配置沿用 Claude Code 的格式，放在项目目录的 .mcp.json（或者 MCP_CONFIG 指定的文件）：

    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/data"],
                           "env": {"TOKEN": "${MY_TOKEN}"}, "autoApprove": ["read_file"]}}}

外部工具第一次调用时要用户点头（autoApprove 里的不用问）；没人能问的场合（评测、测试）不在 autoApprove 里的一律拒绝。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from ..core.messages import ToolCall
from .client import McpClient, McpError, ServerConfig
from .tools import PREFIX, McpTool

__all__ = ["Decision", "McpApproval", "McpClient", "McpError", "McpTool", "ServerConfig", "connect", "load_config"]

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def load_config(path: Path) -> list[ServerConfig]:
    """读 .mcp.json。env 和 args 里的 ${VAR}、${VAR:-默认值} 换成环境变量（密钥不用写进文件）。"""
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    expand = lambda s: _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), str(s))  # noqa: E731
    servers = []
    for name, c in (data.get("mcpServers") or {}).items():
        if c.get("disabled"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"{path}：MCP 服务器名只能用字母、数字、_、-：{name!r}")
        if "command" not in c:
            raise ValueError(f"{path}：{name} 没有 command（现在只支持 stdio 服务器，url / http 的还不支持）")
        servers.append(ServerConfig(name, expand(c["command"]), tuple(expand(a) for a in c.get("args") or ()),
                                    {k: expand(v) for k, v in (c.get("env") or {}).items()},
                                    frozenset(c.get("autoApprove") or ()), c.get("cwd")))
    return servers


def connect(configs: list[ServerConfig], timeout: float, shared: dict[str, McpClient] | None = None
            ) -> tuple[dict[str, McpClient], list[McpTool], list[str]]:
    """连上每个服务器、拉工具列表。返回 (这次新起的客户端, 工具, 连不上的说明)。
    shared 里已经有的（评测里几个 Agent 共用一个服务器）直接用，不重起、也不归这次关。"""
    started: dict[str, McpClient] = {}
    tools: list[McpTool] = []
    problems: list[str] = []
    for config in configs:
        client = (shared or {}).get(config.name)
        try:
            if client is None:
                client = McpClient(config, timeout).start()
                started[config.name] = client
            tools += [McpTool(client, spec) for spec in client.list_tools()]
        except McpError as exc:
            problems.append(str(exc))
            if config.name in started:
                started.pop(config.name).close()
    return started, tools, problems


Decision = Literal["once", "session", "deny"]


@dataclass(slots=True)
class McpApproval:
    """MCP 工具第一次调用时问用户：这次允许 / 本会话都允许 / 拒绝。我们自己的工具不管。"""
    tools: dict[str, McpTool]
    ask: Callable[[ToolCall, McpTool], Decision] | None = None
    allowed: set[str] = field(default_factory=set)       # 本会话都允许的（完整工具名）

    def __post_init__(self) -> None:
        self.allowed |= {name for name, t in self.tools.items() if t.remote_name in t.client.config.auto_approve}

    def __call__(self, call: ToolCall) -> tuple[bool, str]:
        tool = self.tools.get(call.name)
        if not call.name.startswith(PREFIX) or tool is None or call.name in self.allowed:
            return True, ""
        if self.ask is None:
            return False, (f"没有人能审批外部工具 {call.name}。要自动放行，在 MCP 配置里服务器 {tool.server} 的 "
                           f"autoApprove 里加上 {tool.remote_name}")
        decision = self.ask(call, tool)
        if decision == "session":
            self.allowed.add(call.name)
        return (True, "") if decision in ("once", "session") else (False, "用户不允许调用这个外部工具")
