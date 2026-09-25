"""测试共用的替身：假模型、假工具、组装 Agent 的快捷函数。

不需要 key，不需要数据库 —— Agent 只依赖抽象，塞假的进去整个循环照跑。

用法：
    from fakes import ScriptedProvider, EchoTool, make_agent

pytest 会把 tests/ 目录加进 sys.path，所以测试文件里直接 import 就行。
（这里不用 conftest.py 的 fixture，是因为这些是普通的类和函数，
 显式 import 比 fixture 的「按参数名魔法注入」更好读。）
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from data_agent.core.agent import Agent
from data_agent.core.events import Event, collect_sink
from data_agent.core.messages import LLMResponse, Message
from data_agent.core.provider import LLMProvider
from data_agent.core.tools import Tool, ToolRegistry


class ScriptedProvider(LLMProvider):
    """照剧本走的假模型：第 n 次被调用就返回剧本第 n 条，剧本用完就一直返回最后一条。
    剧本里也可以放异常，那一次调用就抛出它（模拟 API 报错）。

    同时把每次收到的消息拍照存进 seen —— 测试「模型到底看到了什么」全靠它。
    """

    model = "scripted"

    def __init__(self, script: Iterable[LLMResponse | Exception] = (),
                 context_window: int | None = None) -> None:
        self.script = list(script)
        self.context_window = context_window
        self.calls = 0                          # 可以手动清零，让剧本从头再来
        self.seen: list[list[Message]] = []
        self.max_tokens_seen: list[int | None] = []
        self.system_seen: list[str | None] = []     # 每次请求的系统提示词和工具定义：
        self.tools_seen: list[list | None] = []     # 测「写摘要和平时的请求共用前缀」要用

    def chat(self, messages, tools=None, system=None, max_tokens=None) -> LLMResponse:
        self.calls += 1
        self.seen.append(list(messages))
        self.max_tokens_seen.append(max_tokens)
        self.system_seen.append(system)
        self.tools_seen.append(tools)
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step                          # 剧本里放异常 = 这一次调用失败
        return step


class EchoTool(Tool):
    name = "echo"
    description = "把输入原样返回"

    class Args(BaseModel):
        # 必填：「参数不合法」的测试靠传错字段来触发校验失败
        text: str = Field(description="要回显的文本")

    def run(self, args: Args) -> str:
        return f"echo: {args.text}"


def make_agent(
    script: Iterable[LLMResponse | Exception] = (),
    *,
    tools: Iterable[Tool] | None = None,
    **agent_kw,
) -> tuple[Agent, list[Event]]:
    """组装一个跑剧本的 Agent，返回 (agent, 事件列表)。

    模型是 ScriptedProvider，要看它收到了什么就用 agent.llm.seen。
    工具默认只有一个 EchoTool；其余参数（context、max_steps、钩子…）原样传给 Agent。
    """
    events: list[Event] = []
    agent = Agent(
        llm=ScriptedProvider(script),
        tools=ToolRegistry(tools if tools is not None else [EchoTool()]),
        system_prompt="测试",
        on_event=collect_sink(events),
        **agent_kw,
    )
    return agent, events
