"""工具执行到一半取消（界面点停止）：沙箱杀内核、SQL 取消查询、MCP 不再等。结果是一条错误，Agent 接着走。"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from data_agent.core.messages import LLMResponse, ToolCall
from data_agent.core.tools import Tool
from data_agent.mcp import McpClient, ServerConfig
from data_agent.tools.python import PYTHON_KERNEL
from data_agent.tools.sandbox import Sandbox
from data_agent.tools.sql.results import ResultStore, result_resolver

from fakes import make_agent

FINAL = LLMResponse(text="好", stop_reason="end_turn")


def later(seconds: float, fn) -> threading.Timer:
    timer = threading.Timer(seconds, fn)
    timer.start()
    return timer


class Waits(Tool):
    """一直等到被取消。"""

    name = "waits"
    description = "等"

    class Args(BaseModel):
        pass

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()

    def run(self, args) -> str:
        self.started.set()
        if not self.cancelled.wait(5):
            return "没被取消"
        raise RuntimeError("用户中断了执行")


def test_agent取消正在跑的工具_结果是错误_这一轮照常往下走():
    tool = Waits()
    agent, _ = make_agent([LLMResponse(text="", tool_calls=[ToolCall("w1", "waits", {})], stop_reason="tool_use"),
                           FINAL], tools=[tool])
    threading.Thread(target=lambda: (tool.started.wait(5), agent.cancel_tool()), daemon=True).start()
    assert agent.run("q") == "好"
    result = next(m for m in agent.context.render() if m.role == "tool")
    assert result.is_error and "用户中断" in result.content
    agent.cancel_tool()                      # 没有在跑的工具：什么都不发生


def test_沙箱取消_杀掉内核_返回用户中断_下次是新内核(tmp_path):
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path, result_resolver(ResultStore()), timeout_s=30)
    try:
        sandbox.run("x = 1")
        started = time.monotonic()
        later(0.5, sandbox.cancel)
        ex = sandbox.run("import time; time.sleep(20)")
        assert time.monotonic() - started < 10
        assert ex.restarted and "用户中断" in ex.error
        assert "NameError" in (sandbox.run("x").error or "")
    finally:
        sandbox.close()


def test_mcp取消_不再等_报用户取消():
    fake = Path(__file__).with_name("mcp_fake_server.py")
    client = McpClient(ServerConfig("fake", sys.executable, (str(fake),)), timeout=10).start()
    try:
        started = time.monotonic()
        later(0.3, client.cancel)
        with pytest.raises(Exception, match="用户取消"):
            client.call_tool("slow", {})
        assert time.monotonic() - started < 3
    finally:
        client.close()


@pytest.mark.skipif(not os.environ.get("FINHELM_DB_TESTS"), reason="要连 docker 里的库：FINHELM_DB_TESTS=1")
def test_sql取消_正在跑的查询马上结束():
    import psycopg

    from data_agent.db.connection import Database
    from data_agent.settings import Settings

    db = Database(Settings().database_url)
    started = time.monotonic()
    later(0.5, db.cancel)
    with pytest.raises(psycopg.errors.QueryCanceled):
        db.query("SELECT pg_sleep(20)")
    assert time.monotonic() - started < 5
