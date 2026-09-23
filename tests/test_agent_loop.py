"""Agent 主循环的测试 —— 不需要 API key，也不需要数据库。

这正是抽象层带来的好处：
    塞一个假 Provider 进去，整个循环照跑。
    Agent 不关心背后是 Claude、DeepSeek 还是一段写死的剧本。

运行：pytest
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from data_agent.core.agent import Agent
from data_agent.core.context import FullContext, TurnWindowContext
from data_agent.core.events import Event, LLMResponded, ToolFinished, collect_sink
from data_agent.core.messages import LLMResponse, Message, ToolCall
from data_agent.llm.base import LLMProvider
from data_agent.tools.base import Tool
from data_agent.tools.registry import ToolRegistry


# ------------------------------------------------------------------ 测试替身
class ScriptedProvider(LLMProvider):
    """照剧本走的假模型：第 n 次被调用就返回剧本第 n 条。"""

    model = "scripted"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script
        self.seen_messages: list[list[Message]] = []

    def chat(self, messages, tools=None, system=None) -> LLMResponse:
        self.seen_messages.append(list(messages))
        idx = min(len(self.seen_messages) - 1, len(self.script) - 1)
        return self.script[idx]


class EchoTool(Tool):
    name = "echo"
    description = "把输入原样返回"

    class Args(BaseModel):
        text: str = Field(description="要回显的文本")

    def run(self, args: Args) -> str:
        return f"echo: {args.text}"


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


def make_agent(script: list[LLMResponse], registry: ToolRegistry, **kw) -> tuple[Agent, list[Event]]:
    events: list[Event] = []
    agent = Agent(
        llm=ScriptedProvider(script),
        tools=registry,
        system_prompt="测试用",
        on_event=collect_sink(events),
        **kw,
    )
    return agent, events


# ------------------------------------------------------------------ 测试用例
def test_没有工具调用时直接返回(registry):
    agent, _ = make_agent([LLMResponse(text="直接回答")], registry)
    assert agent.run("你好") == "直接回答"


def test_调用工具后再回答(registry):
    script = [
        LLMResponse(text="我查一下", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        LLMResponse(text="查完了"),
    ]
    agent, events = make_agent(script, registry)

    assert agent.run("回显 hi") == "查完了"

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert len(finished) == 1
    assert finished[0].ok is True
    assert finished[0].content == "echo: hi"


def test_工具抛异常不会中断Agent(registry):
    """核心行为：工具炸了要把错误喂回模型，而不是让整个 run() 崩掉。"""
    script = [
        LLMResponse(text="试试", tool_calls=[ToolCall("c1", "boom", {})]),
        LLMResponse(text="知道了，换个方式"),
    ]
    agent, events = make_agent(script, registry)

    assert agent.run("跑一下") == "知道了，换个方式"

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].ok is False
    assert "故意炸的" in finished[0].content


def test_参数不合法时返回错误而不是抛异常(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"wrong_field": 1})]),
        LLMResponse(text="改好了"),
    ]
    agent, events = make_agent(script, registry)
    agent.run("x")

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].ok is False
    assert "参数不合法" in finished[0].content


def test_调用不存在的工具(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "不存在的工具", {})]),
        LLMResponse(text="好吧"),
    ]
    agent, events = make_agent(script, registry)
    agent.run("x")

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].ok is False
    assert "不存在" in finished[0].content


def test_审批钩子拒绝时模型仍能收到结果(registry):
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        LLMResponse(text="那算了"),
    ]
    agent, _ = make_agent(
        script, registry, approval_hook=lambda call: (False, "用户不同意"),
    )
    assert agent.run("x") == "那算了"

    # 被拒绝也必须往历史里塞一条 tool 消息，否则 tool_call 没有对应结果
    history = agent.context.render()
    tool_msgs = [m for m in history if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "用户不同意" in tool_msgs[0].content


def test_达到步数上限会停下(registry):
    # 剧本只有一条：永远要求调工具，会一直循环
    script = [LLMResponse(text="", tool_calls=[ToolCall("c", "echo", {"text": "x"})])]
    agent, _ = make_agent(script, registry, max_steps=3)

    answer = agent.run("死循环")
    assert "最大步数 3" in answer


def test_每轮都会把完整工具表发给模型(registry):
    """模型是无状态的，工具表每轮都要重发。"""
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="done"),
    ]
    agent, events = make_agent(script, registry)
    agent.run("x")

    responded = [e for e in events if isinstance(e, LLMResponded)]
    assert len(responded) == 2      # 两轮请求，每轮都带着 2 个工具的 schema
    assert len(registry.schemas()) == 2


# ------------------------------------------------------------------ 上下文
def test_按回合裁剪不会拆散工具调用():
    """裁剪的单位必须是「回合」，不能是「消息」，否则 tool_call 和 tool_result 会被拆开。"""
    ctx = TurnWindowContext(max_turns=1)
    # 第 1 轮
    ctx.add(Message.user("问题1"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("a", "echo", {})]))
    ctx.add(Message.tool_result("a", "结果1"))
    ctx.add(Message.assistant("答案1"))
    # 第 2 轮
    ctx.add(Message.user("问题2"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("b", "echo", {})]))
    ctx.add(Message.tool_result("b", "结果2"))

    rendered = ctx.render()
    assert rendered[0].content == "问题2"        # 第 1 轮整体被裁掉
    # 保留下来的这轮里，tool_call 和它的结果都还在
    call_ids = {c.id for m in rendered for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in rendered if m.role == "tool"}
    assert call_ids == result_ids


def test_全量上下文不裁剪():
    ctx = FullContext()
    for i in range(10):
        ctx.add(Message.user(f"m{i}"))
    assert len(ctx.render()) == 10
