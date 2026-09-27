"""Agent 主循环的测试 —— 不需要 API key，也不需要数据库。

这正是抽象层带来的好处：
    塞一个假 Provider 进去，整个循环照跑。
    Agent 不关心背后是 Claude、DeepSeek 还是一段写死的剧本。

运行：pytest
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from data_agent.core.context import Context
from data_agent.core.events import LLMResponded, ToolFinished
from data_agent.core.messages import INVALID_JSON_ARGS, LLMResponse, Message, ToolCall
from data_agent.core.tools import Tool, ToolOutput, ToolRegistry

from fakes import EchoTool, make_agent


# ------------------------------------------------------------------ 测试替身
class BoomTool(Tool):
    name = "boom"
    description = "总是抛异常，用来测试错误处理"

    class Args(BaseModel):
        pass

    def run(self, args: Args) -> str:
        raise RuntimeError("故意炸的")


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([EchoTool(), BoomTool()])


# ------------------------------------------------------------------ 测试用例
def test_没有工具调用时直接返回(registry):
    agent, _ = make_agent([LLMResponse(text="直接回答")], tools=registry)
    assert agent.run("你好") == "直接回答"


def test_调用工具后再回答(registry):
    script = [
        LLMResponse(text="我查一下", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        LLMResponse(text="查完了"),
    ]
    agent, events = make_agent(script, tools=registry)

    assert agent.run("回显 hi") == "查完了"

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert len(finished) == 1
    assert finished[0].is_error is False
    assert finished[0].content == "echo: hi"


class TableTool(Tool):
    name = "table"
    description = "模型看摘要，界面拿完整数据"

    class Args(BaseModel):
        pass

    def run(self, args: Args) -> ToolOutput:
        return ToolOutput("共 3 行，看前 1 行：a", details={"rows": ["a", "b", "c"]})


def test_details只给界面_不进历史():
    """一个工具结果两个读者：content 给模型，details 随事件交给界面。"""
    script = [LLMResponse(text="", tool_calls=[ToolCall("c1", "table", {})]), LLMResponse(text="好")]
    agent, events = make_agent(script, tools=[TableTool()])
    agent.run("查")

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.details == {"rows": ["a", "b", "c"]}
    sent = agent.llm.seen[-1]
    assert [m.content for m in sent if m.role == "tool"] == ["共 3 行，看前 1 行：a"]
    assert "'b'" not in repr(sent)


def test_截断只截content_不动details():
    out = ToolOutput("x" * 100, details=[1, 2, 3]).capped(10)
    assert out.content.startswith("x" * 10) and out.details == [1, 2, 3]


def test_工具抛异常不会中断Agent(registry):
    """核心行为：工具炸了要把错误喂回模型，而不是让整个 run() 崩掉。"""
    script = [
        LLMResponse(text="试试", tool_calls=[ToolCall("c1", "boom", {})]),
        LLMResponse(text="知道了，换个方式"),
    ]
    agent, events = make_agent(script, tools=registry)

    assert agent.run("跑一下") == "知道了，换个方式"

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].is_error is True
    assert "故意炸的" in finished[0].content


def test_参数不合法时返回错误而不是抛异常(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"wrong_field": 1})]),
        LLMResponse(text="改好了"),
    ]
    agent, events = make_agent(script, tools=registry)
    agent.run("x")

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].is_error is True
    assert "参数不合法" in finished[0].content


class PageTool(Tool):
    """参数全是可选的：多传、拼错的字段被静默丢掉的话，调用会带着默认值「成功」。"""
    name = "page"
    description = "分页"

    class Args(BaseModel):
        limit: int = 10

    def run(self, args: Args) -> str:
        return f"limit={args.limit}"


def test_拼错的可选参数不会被静默忽略():
    out = PageTool().execute({"limt": 5})
    assert out.is_error and "没有参数 limt" in out.content and "可用的参数：limit" in out.content
    assert PageTool().execute({"limit": 5}).content == "limit=5"


def test_参数不是合法JSON时明确报错_不带着默认值执行():
    out = PageTool().execute({INVALID_JSON_ARGS: '{"limit": 5'})
    assert out.is_error and "不是合法的 JSON" in out.content and '{"limit": 5' in out.content


def test_schema里写明不收多余参数():
    assert PageTool().schema()["parameters"]["additionalProperties"] is False


def test_调用不存在的工具(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "不存在的工具", {})]),
        LLMResponse(text="好吧"),
    ]
    agent, events = make_agent(script, tools=registry)
    agent.run("x")

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].is_error is True
    assert "不存在" in finished[0].content


def test_审批钩子拒绝时模型仍能收到结果(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        LLMResponse(text="那算了"),
    ]
    agent, _ = make_agent(
        script, tools=registry, approval_hook=lambda call: (False, "用户不同意"),
    )
    assert agent.run("x") == "那算了"

    # 被拒绝也必须往历史里塞一条 tool 消息，否则 tool_call 没有对应结果
    history = agent.context.render()
    tool_msgs = [m for m in history if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "用户不同意" in tool_msgs[0].content
    assert tool_msgs[0].is_error, "被拒绝也是失败，要让模型知道"


def test_工具失败的标记一路带进历史(registry):
    script = [
        LLMResponse(text="试试", tool_calls=[ToolCall("c1", "boom", {})]),
        LLMResponse(text="知道了"),
    ]
    agent, _ = make_agent(script, tools=registry)
    agent.run("x")
    [result] = [m for m in agent.llm.seen[-1] if m.role == "tool"]
    assert result.is_error


def test_达到步数上限会停下(registry):
    # 剧本只有一条：永远要求调工具，会一直循环
    script = [LLMResponse(text="", tool_calls=[ToolCall("c", "echo", {"text": "x"})])]
    agent, _ = make_agent(script, tools=registry, max_steps=3)

    answer = agent.run("死循环")
    assert "最大步数 3" in answer


def test_每轮都会把完整工具表发给模型(registry):
    """模型是无状态的，工具表每轮都要重发。"""
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="done"),
    ]
    agent, events = make_agent(script, tools=registry)
    agent.run("x")

    responded = [e for e in events if isinstance(e, LLMResponded)]
    assert len(responded) == 2      # 两轮请求，每轮都带着 2 个工具的 schema
    assert len(registry.schemas()) == 2


# ------------------------------------------------------------------ 上下文
def test_全量上下文不裁剪():
    ctx = Context()
    for i in range(10):
        ctx.add(Message.user(f"m{i}"))
    assert len(ctx.render()) == 10


def test_系统提示词一个会话只算一次_reset后重算():
    """会话上下文（库概览）每步都重算的话，每步都要查一次库；内容一变，缓存也跟着废。"""
    calls = []

    def overview() -> str:
        calls.append(1)
        return f"库概览 v{len(calls)}"

    ok = LLMResponse(text="答", stop_reason="end_turn")
    tool = LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "x"})], stop_reason="tool_use")
    agent, _ = make_agent([tool, ok, ok], session_context=overview)
    agent.run("问题1")
    agent.run("问题2")
    assert len(calls) == 1
    assert set(agent.llm.system_seen) == {"测试\n\n库概览 v1"}

    agent.reset()
    agent.run("问题3")
    assert agent.llm.system_seen[-1] == "测试\n\n库概览 v2"
