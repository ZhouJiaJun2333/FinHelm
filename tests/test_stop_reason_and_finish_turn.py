"""两个新增能力的测试：stop_reason 分诊 + finish_turn 钩子。

都不需要 API key 和数据库 —— 又一次验证了抽象层的价值。
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from data_agent.core.agent import Agent, TurnDecision, TurnOutcome
from data_agent.core.errors import (
    ModelRefused,
    OutputTruncated,
    UnexpectedStopReason,
)
from data_agent.core.events import Event, TurnContinued, collect_sink
from data_agent.core.messages import LLMResponse, ToolCall
from data_agent.llm.base import LLMProvider
from data_agent.tools.base import Tool
from data_agent.tools.registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    model = "scripted"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script
        self.calls = 0

    def chat(self, messages, tools=None, system=None) -> LLMResponse:
        self.calls += 1
        return self.script[min(self.calls - 1, len(self.script) - 1)]


class EchoTool(Tool):
    name = "echo"
    description = "回显"

    class Args(BaseModel):
        text: str = Field(default="x")

    def run(self, args: Args) -> str:
        return f"echo: {args.text}"


def make_agent(script, **kw) -> tuple[Agent, list[Event]]:
    events: list[Event] = []
    agent = Agent(
        llm=ScriptedProvider(script),
        tools=ToolRegistry([EchoTool()]),
        system_prompt="测试",
        on_event=collect_sink(events),
        **kw,
    )
    return agent, events


# ====================================================== stop_reason 分诊
def test_正常结束不报错():
    agent, _ = make_agent([LLMResponse(text="答完了", stop_reason="end_turn")])
    assert agent.run("x") == "答完了"


@pytest.mark.parametrize("reason", ["end_turn", "stop", "tool_use", "tool_calls", ""])
def test_各厂商的正常stop_reason都认识(reason):
    agent, _ = make_agent([LLMResponse(text="ok", stop_reason=reason)])
    assert agent.run("x") == "ok"


@pytest.mark.parametrize("reason", ["max_tokens", "length"])
def test_被截断时抛异常而不是当成完成(reason):
    """核心行为：被 max_tokens 砍断时同样没有 tool_calls，
    但那是半句话，绝不能当最终答案返回。"""
    agent, _ = make_agent([
        LLMResponse(text="华东大区的销售额是", stop_reason=reason,
                    usage={"output_tokens": 8192}),
    ])
    with pytest.raises(OutputTruncated, match="截断"):
        agent.run("x")


def test_截断异常里带了可操作的提示():
    agent, _ = make_agent([
        LLMResponse(text="半句", stop_reason="max_tokens",
                    usage={"output_tokens": 4096}),
    ])
    with pytest.raises(OutputTruncated) as exc:
        agent.run("x")
    assert "MAX_TOKENS" in str(exc.value)
    assert "4096" in str(exc.value)


@pytest.mark.parametrize("reason", ["refusal", "content_filter"])
def test_模型拒绝时抛专门的异常(reason):
    agent, _ = make_agent([LLMResponse(text="", stop_reason=reason)])
    with pytest.raises(ModelRefused):
        agent.run("x")


def test_没见过的stop_reason宁可炸掉也不静默():
    agent, _ = make_agent([LLMResponse(text="?", stop_reason="某个没见过的值")])
    with pytest.raises(UnexpectedStopReason, match="NORMAL_STOP_REASONS"):
        agent.run("x")


def test_大小写不敏感():
    agent, _ = make_agent([LLMResponse(text="半句", stop_reason="MAX_TOKENS")])
    with pytest.raises(OutputTruncated):
        agent.run("x")


# ====================================================== finish_turn 钩子
def test_不装钩子时行为和以前完全一致():
    script = [
        LLMResponse(text="调工具", stop_reason="tool_use",
                    tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="完成", stop_reason="end_turn"),
    ]
    agent, _ = make_agent(script)
    assert agent.run("x") == "完成"


def test_钩子可以让模型继续干():
    """模型以为自己答完了，钩子判定没完成，把它推回去接着干。"""
    script = [
        LLMResponse(text="第一版答案", stop_reason="end_turn"),
        LLMResponse(text="补充完整的答案", stop_reason="end_turn"),
    ]

    seen: list[TurnOutcome] = []

    def hook(outcome: TurnOutcome) -> TurnDecision:
        seen.append(outcome)
        if outcome.step == 1:
            return TurnDecision.keep_going("还缺占比，补上")
        return TurnDecision.end()

    agent, events = make_agent(script, finish_turn_hook=hook)

    assert agent.run("x") == "补充完整的答案"
    assert [o.step for o in seen] == [1, 2]
    assert [o.requested_tools for o in seen] == [False, False]

    # 推动消息必须以一条 user 消息进历史，否则下一轮角色不交替
    history = agent.context.render()
    assert history[-2].role == "user"
    assert history[-2].content == "还缺占比，补上"

    continued = [e for e in events if isinstance(e, TurnContinued)]
    assert len(continued) == 1
    assert continued[0].nudge == "还缺占比，补上"


def test_钩子可以提前收工():
    """即使模型还想调工具，钩子说停就停。"""
    script = [
        LLMResponse(text="我还想查", stop_reason="tool_use",
                    tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="不该走到这里", stop_reason="end_turn"),
    ]
    agent, _ = make_agent(script, finish_turn_hook=lambda o: TurnDecision.end())

    assert agent.run("x") == "我还想查"
    assert agent.llm.calls == 1          # 只请求了一次


def test_钩子能看到是否跑过工具():
    script = [
        LLMResponse(text="", stop_reason="tool_use",
                    tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="done", stop_reason="end_turn"),
    ]
    seen: list[bool] = []

    def hook(outcome: TurnOutcome) -> TurnDecision:
        seen.append(outcome.requested_tools)
        return TurnDecision.keep_going() if outcome.requested_tools else TurnDecision.end()

    agent, _ = make_agent(script, finish_turn_hook=hook)
    agent.run("x")
    assert seen == [True, False]


def test_continue但没给推动消息时用默认的():
    script = [
        LLMResponse(text="a", stop_reason="end_turn"),
        LLMResponse(text="b", stop_reason="end_turn"),
    ]
    calls = {"n": 0}

    def hook(outcome: TurnOutcome) -> TurnDecision:
        calls["n"] += 1
        return TurnDecision.keep_going() if calls["n"] == 1 else TurnDecision.end()

    agent, _ = make_agent(script, finish_turn_hook=hook)
    agent.run("x")

    history = agent.context.render()
    assert history[-2].role == "user"
    assert "请继续" in history[-2].content


def test_钩子不会让死循环逃过步数上限():
    script = [LLMResponse(text="永远说没完", stop_reason="end_turn")]
    agent, _ = make_agent(
        script, max_steps=3, finish_turn_hook=lambda o: TurnDecision.keep_going("继续"),
    )
    assert "最大步数 3" in agent.run("x")


# ============================ run() 的事务语义（历史不能留半截状态）
#
# 一轮失败时历史里可能留下两种半截状态，都会让**之后每一轮**都 400：
#     [user] 没有回复      → 用户再问 → [user, user]，Anthropic 角色不交替
#     assistant 有 tool_calls 却没有结果  → 两家都拒
# 所以 run() 是原子的：要么完整完成，要么历史回到进来之前。

def test_截断后整轮回滚():
    """不只是 assistant 那条 —— 连用户的提问一起回滚。

    只删 assistant 会留下 [user]，用户重试就变成 [user, user]。
    """
    agent, _ = make_agent([
        LLMResponse(text="华东大区的销售额是", stop_reason="max_tokens"),
    ])
    with pytest.raises(OutputTruncated):
        agent.run("华东卖了多少？")

    assert agent.context.render() == [], "失败的一轮必须不留任何痕迹"


def test_截断在工具调用中途时不留悬空的tool_call():
    """最致命的一种：没有结果的 tool_call 会让之后每一轮都 400。"""
    agent, _ = make_agent([
        LLMResponse(
            text="", stop_reason="max_tokens",
            tool_calls=[ToolCall("c1", "echo", {"__invalid_json__": '{"tex'})],
        ),
    ])
    with pytest.raises(OutputTruncated):
        agent.run("查一下")

    assert agent.context.render() == []


def test_失败后重试不会产生连续的user消息():
    """核心回归：OpenAI 兼容接口容忍连续 user，Anthropic 不容忍。
    在本地（DeepSeek）怎么测都不出来，换厂商才炸 —— 所以必须有测试守着。
    """
    script = [
        LLMResponse(text="被砍断的半句", stop_reason="max_tokens",
                    tool_calls=[ToolCall("c1", "echo", {"text": "x"})]),
        LLMResponse(text="这次答完了", stop_reason="end_turn"),
    ]
    agent, _ = make_agent(script)

    with pytest.raises(OutputTruncated):
        agent.run("第一个问题")
    assert agent.run("第二个问题") == "这次答完了"

    roles = [m.role for m in agent.context.render()]
    assert roles == ["user", "assistant"]
    assert not any(
        a == b == "user" for a, b in zip(roles, roles[1:])
    ), f"出现了连续的 user 消息：{roles}"


def test_只回滚失败的那一轮():
    """之前已经成功的对话必须保住 —— 别把澡盆和孩子一起倒了。"""
    ok = LLMResponse(text="第一轮答完了", stop_reason="end_turn")
    bad = LLMResponse(text="半截", stop_reason="max_tokens")

    agent, _ = make_agent([ok])
    agent.run("问题一")
    before = list(agent.context.render())
    assert len(before) == 2

    agent.llm.script = [bad]
    agent.llm.calls = 0
    with pytest.raises(OutputTruncated):
        agent.run("问题二")

    assert agent.context.render() == before


def test_被拒绝的回复同样整轮回滚():
    agent, _ = make_agent([LLMResponse(text="", stop_reason="refusal")])
    with pytest.raises(ModelRefused):
        agent.run("x")
    assert agent.context.render() == []


def test_用户中断也回滚():
    """Ctrl-C 打断的半截回合同样会毒化历史，所以用 finally 而不是 except。"""
    class Interrupting(LLMProvider):
        model = "interrupting"
        def chat(self, messages, tools=None, system=None):
            raise KeyboardInterrupt

    agent, _ = make_agent([])
    agent.llm = Interrupting()
    with pytest.raises(KeyboardInterrupt):
        agent.run("x")
    assert agent.context.render() == []


def test_任意异常都回滚():
    """网络错、SDK 报错……都一样。历史不该因为一次失败而残缺。"""
    class Exploding(LLMProvider):
        model = "exploding"
        def chat(self, messages, tools=None, system=None):
            raise RuntimeError("503 Service is too busy")

    agent, _ = make_agent([])
    agent.llm = Exploding()
    with pytest.raises(RuntimeError):
        agent.run("x")
    assert agent.context.render() == []


def test_正常回复照常进历史():
    """别把好的也拦掉了。"""
    agent, _ = make_agent([LLMResponse(text="正常回答", stop_reason="end_turn")])
    agent.run("x")
    assert [m.role for m in agent.context.render()] == ["user", "assistant"]


def test_requested_tools_包含被审批拒掉的调用():
    """名字是「请求了工具」不是「跑成功了」—— 被拒的也算 True。

    这不是疏忽：被拒时历史里会留一条「已拒绝」的 tool_result，
    循环必须再跑一轮让模型看到拒绝理由并改道，所以语义上就该是 True。
    """
    script = [
        LLMResponse(text="我要调工具", stop_reason="tool_use",
                    tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="好吧我换个办法", stop_reason="end_turn"),
    ]
    seen: list[bool] = []

    def hook(outcome: TurnOutcome) -> TurnDecision:
        seen.append(outcome.requested_tools)
        return TurnDecision.keep_going() if outcome.requested_tools else TurnDecision.end()

    agent, events = make_agent(
        script,
        approval_hook=lambda call: (False, "测试拒绝"),
        finish_turn_hook=hook,
    )
    agent.run("x")

    assert seen == [True, False], "工具没真的执行，但 requested_tools 仍是 True"
    assert any(type(e).__name__ == "ToolDenied" for e in events)

    # 被拒也要有 tool_result，否则历史形状不合法（悬空的 tool_call）
    history = agent.context.render()
    assert len([m for m in history if m.role == "tool"]) == 1
