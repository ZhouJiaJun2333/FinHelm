"""MCP：手写的 stdio 客户端、工具包装和参数校验、配置、审批、组装，以及我们自己的服务器。"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.core.messages import ToolCall
from data_agent.core.tools import Tool
from data_agent.mcp import McpApproval, McpClient, McpError, McpTool, ServerConfig, connect, load_config
from data_agent.mcp.server import serve
from data_agent.mcp.tools import MAX_DESCRIPTION, validate
from data_agent.settings import Settings

from fakes import ScriptedProvider

FAKE = Path(__file__).with_name("mcp_fake_server.py")


def _config(**kw) -> ServerConfig:
    return ServerConfig("fake", sys.executable, (str(FAKE),), **kw)


@pytest.fixture
def client():
    c = McpClient(_config(), timeout=10).start()
    yield c
    c.close()


# ================================================================ 客户端
def test_握手_说明_工具列表分页(client):
    assert client.server_info == {"name": "fake"} and client.instructions == "Use echo to repeat text."
    assert [t["name"] for t in client.list_tools()] == ["echo", "fail", "picture", "crash", "slow"]


def test_调用_中间服务器反过来ping客户端(client):
    result = client.call_tool("echo", {"text": "ab", "times": 2})
    assert result["content"][0]["text"] == "abab"


def test_服务器崩了_等着的调用马上报错_带上stderr(client):
    with pytest.raises(McpError, match="(?s)退出.*fatal: crashing"):
        client.call_tool("crash", {})


def test_超时(client):
    with pytest.raises(McpError, match="超时"):
        client.call_tool("slow", {}, timeout=0.5)


def test_命令不存在():
    with pytest.raises(McpError, match="起不来"):
        McpClient(ServerConfig("nope", "definitely-not-a-command-xyz")).start()


# ================================================================ 工具包装
def _tools(client) -> dict[str, McpTool]:
    return {t.remote_name: t for t in (McpTool(client, spec) for spec in client.list_tools())}


def test_名字加前缀_描述标来源并截断_只读才能被清理(client):
    echo = _tools(client)["echo"]
    assert echo.name == "mcp__fake__echo" and echo.rerunnable
    assert echo.description.startswith("[外部 MCP 服务器 fake 的工具] Echo the text back.")
    assert len(echo.description) < MAX_DESCRIPTION + 40 and echo.description.endswith("…")
    assert echo.schema()["parameters"]["required"] == ["text"]
    assert not _tools(client)["fail"].rerunnable


def test_执行_校验参数_报错和图片(client):
    tools = _tools(client)
    assert tools["echo"].execute({"text": "hi"}).content == "hi"
    missing = tools["echo"].execute({})
    assert missing.is_error and "缺少必填参数 text" in missing.content
    typo = tools["echo"].execute({"text": "hi", "time": 2})
    assert typo.is_error and "没有参数 time，可用的：text、times" in typo.content, "拼错的参数拦下来，不交给服务器悄悄忽略"
    assert "不能小于 1" in tools["echo"].execute({"text": "hi", "times": 0}).content
    failed = tools["fail"].execute({})
    assert failed.is_error and failed.content == "boom"
    pic = tools["picture"].execute({})
    assert pic.content == "a chart" and pic.images[0].media_type == "image/png"


def test_JSON_Schema校验():
    schema = {"type": "object", "properties": {
        "mode": {"enum": ["a", "b"]}, "n": {"type": ["integer", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "opt": {"type": "object", "properties": {"x": {"type": "number"}}}}}
    assert validate(schema, {"mode": "a", "n": None, "tags": ["x"], "opt": {"x": 1.5, "extra": 1}}) == [], \
        "嵌套对象没写 additionalProperties 就允许多的"
    errors = validate(schema, {"mode": "c", "n": True, "tags": ["x", 2, "z"]})
    assert any("只能是" in e for e in errors) and any("n 应该是 integer/null" in e for e in errors)
    assert any("tags[1] 应该是 string" in e for e in errors) and any("tags 太长" in e for e in errors)
    assert validate({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 1.5) == ["参数 不符合任何一种允许的格式"]
    assert validate({"type": "object", "additionalProperties": True, "properties": {"a": {}}}, {"b": 1}) == []


# ================================================================ 配置、审批
def test_读配置_环境变量_跳过停用的_不支持http(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TOKEN", "secret")
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": {
        "a": {"command": "python", "args": ["x.py", "${MISSING:-dflt}"], "env": {"T": "${FAKE_TOKEN}"},
              "autoApprove": ["echo"]},
        "off": {"command": "python", "disabled": True}}}), encoding="utf-8")
    [a] = load_config(path)
    assert a.args == ("x.py", "dflt") and a.env == {"T": "secret"} and a.auto_approve == {"echo"}
    assert load_config(tmp_path / "nope.json") == []
    path.write_text(json.dumps({"mcpServers": {"web": {"url": "https://x"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="只支持 stdio"):
        load_config(path)


def test_审批_自动放行_没人问就拒绝_本会话记住(client):
    tools = {t.name: t for t in (McpTool(McpClient(_config(auto_approve=frozenset({"echo"}))), s)
                                 for s in client.list_tools())}
    call = lambda name: ToolCall("c1", name, {})  # noqa: E731
    nobody = McpApproval(tools)
    assert nobody(call("mcp__fake__echo")) == (True, ""), "autoApprove 里的"
    assert nobody(call("run_sql")) == (True, ""), "我们自己的工具不管"
    allowed, reason = nobody(call("mcp__fake__fail"))
    assert not allowed and "autoApprove 里加上 fail" in reason

    answers = iter(["once", "session", "deny"])
    asked = []
    approval = McpApproval(tools, ask=lambda c, t: asked.append(t.remote_name) or next(answers))
    assert approval(call("mcp__fake__fail"))[0] and approval(call("mcp__fake__picture"))[0]
    assert approval(call("mcp__fake__picture"))[0], "本会话都允许：不再问"
    assert not approval(call("mcp__fake__fail"))[0], "只允许了一次：再问，这次拒绝"
    assert asked == ["fail", "picture", "fail"]


# ================================================================ 组装
def test_项目的mcp配置_工具注册_说明进提示词_审批_关掉进程(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"fake": {
        "command": sys.executable, "args": [str(FAKE)], "autoApprove": ["echo"]}}}), encoding="utf-8")
    settings = Settings(project_dir=str(tmp_path), database_url="", python_sandbox=False, r_sandbox=False,
                        mcp_enabled=True)
    app = build_application(settings, llm=ScriptedProvider())
    try:
        assert "mcp__fake__echo" in app.tools and len(app.mcp_tools) == 5
        assert "## 外部工具（MCP）的说明\n" in app.agent.system_prompt
        assert "### fake\nUse echo to repeat text." in app.agent.system_prompt
        hook = app.agent.approval_hook
        assert hook(ToolCall("1", "mcp__fake__echo", {}))[0] and not hook(ToolCall("2", "mcp__fake__fail", {}))[0]
        proc = app.mcp_clients["fake"]._proc
    finally:
        app.close()
    assert proc.poll() is not None, "close() 把服务器进程关掉"


def test_服务器连不上_程序照样起来_记下原因(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"bad": {"command": "definitely-not-xyz"}}}),
                                        encoding="utf-8")
    settings = Settings(project_dir=str(tmp_path), database_url="", python_sandbox=False, r_sandbox=False,
                        mcp_enabled=True)
    app = build_application(settings, llm=ScriptedProvider())
    assert app.mcp_problems and "起不来" in app.mcp_problems[0] and "外部工具" not in app.agent.system_prompt


def test_共用的客户端不重起_也不归这次关(client):
    started, tools, problems = connect([client.config], 10, shared={"fake": client})
    assert started == {} and len(tools) == 5 and not problems


# ================================================================ 我们的服务器
class _Echo(Tool):
    from pydantic import BaseModel as _B

    class Args(_B):
        text: str

    name, description, rerunnable = "say", "Say it.", True

    def run(self, args):
        return f"said {args.text}"


def _rpc(*msgs) -> list[dict]:
    out = io.BytesIO()
    serve([_Echo()], "how to use", io.BytesIO("".join(json.dumps(m) + "\n" for m in msgs).encode()), out)
    return [json.loads(line) for line in out.getvalue().decode().splitlines()]


def test_我们的服务器_握手_列工具_调用_未知方法():
    replies = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "x"}},
                   {"jsonrpc": "2.0", "method": "notifications/initialized"},
                   {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                   {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "say", "arguments": {"text": "hi"}}},
                   {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "say", "arguments": {}}},
                   {"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5], "通知不回"
    assert replies[0]["result"]["instructions"] == "how to use"
    [tool] = replies[1]["result"]["tools"]
    assert tool["name"] == "say" and tool["inputSchema"]["required"] == ["text"] and tool["annotations"]["readOnlyHint"]
    assert replies[2]["result"] == {"content": [{"type": "text", "text": "said hi"}], "isError": False}
    assert replies[3]["result"]["isError"], "参数错了是工具结果里的错，不是协议错"
    assert replies[4]["error"]["code"] == -32601
