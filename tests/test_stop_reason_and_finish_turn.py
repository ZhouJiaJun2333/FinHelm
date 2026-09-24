"""两个新增能力的测试：stop_reason 分诊 + finish_turn 钩子。

都不需要 API key 和数据库 —— 又一次验证了抽象层的价值。
"""

from __future__ import annotations

import pytest

from data_agent.core.agent import TurnDecision, TurnOutcome
from data_agent.core.errors import (
    ModelRefused,
    OutputTruncated,
    UnexpectedStopReason,
)
from data_agent.core.events import TurnContinued
from data_agent.core.messages import LLMResponse, ToolCall, Usage
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.base import LLMProvider

from fakes import make_agent


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
                    usage=Usage(output=8192)),
    ])
    with pytest.raises(OutputTruncated, match="截断"):
        agent.run("x")


def test_截断异常里带了可操作的提示():
    agent, _ = make_agent([
        LLMResponse(text="半句", stop_reason="max_tokens",
                    usage=Usage(output=4096)),
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
# 一轮失败时历史里可能留下两种半截状态：
#     assistant 有 tool_calls 却没有结果  → 两家都 400，之后每一轮都 400
#     [user] 没有回复      → 用户再问 → [user, user]，不报错但被合并成一条，
#                            模型看到的是同一个问题问了两遍
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
    """连续 user 两家都不报错（会被合并成一条），所以真实调用永远发现不了，
    只会表现为答案莫名其妙地变怪 —— 必须有测试守着。
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


# ====================== 步数耗尽：第三个「半截状态」入口（正常返回路径）
#
# 前两个入口走异常路径，被 run() 的事务兜住了。这个是**正常返回**，
# 事务照常提交 —— 所以必须自己保证提交的历史形状合法。

class _NeverStops(LLMProvider):
    """永远调工具，永不收尾 —— 模拟模型陷在循环里。"""
    model = "never-stops"

    def chat(self, messages, tools=None, system=None) -> LLMResponse:
        return LLMResponse(text="再查一次", stop_reason="tool_use",
                           tool_calls=[ToolCall("c", "echo", {"text": "x"})])


def _has_consecutive_user(messages) -> bool:
    roles = [m.role for m in messages]
    return any(a == b == "user" for a, b in zip(roles, roles[1:]))


def test_步数耗尽时历史以assistant收尾():
    """不补收尾的话历史会以 tool 结果结尾，下一轮追加 user 就非法了。"""
    agent, _ = make_agent([], max_steps=2)
    agent.llm = _NeverStops()

    answer = agent.run("停不下来的问题")
    history = agent.context.render()

    assert "最大步数 2" in answer
    assert history[-1].role == "assistant"
    assert history[-1].content == answer, "兜底文案必须进历史，不能只返回给用户"


def test_步数耗尽后再提问不会产生连续user():
    """注意要验**转换后**的形状。

    这一路历史以 tool 消息结尾，中立结构里 [tool, user] 并不算连续 user ——
    只有 Anthropic 把 tool_result 包进 user 消息之后才暴露。
    光验中立结构的话这个测试没牙，撤掉修复也是绿的。
    """
    agent, _ = make_agent([], max_steps=2)
    agent.llm = _NeverStops()
    agent.run("问题一")
    agent.run("问题二")

    assert not _has_consecutive_user(agent.context.render())

    roles = [m["role"] for m in AnthropicProvider.convert_messages(agent.context.render())]
    assert not any(a == b for a, b in zip(roles, roles[1:])), roles


def test_finish_turn撞上限时同样以assistant收尾():
    """另一个变体：历史以 nudge 的 user 消息结尾，中立结构里就已经连续了。"""
    agent, _ = make_agent(
        [LLMResponse(text="我觉得答完了", stop_reason="end_turn")],
        max_steps=2,
        finish_turn_hook=lambda o: TurnDecision.keep_going("继续"),
    )
    agent.run("问题一")
    assert agent.context.render()[-1].role == "assistant"

    agent.run("问题二")
    assert not _has_consecutive_user(agent.context.render())


def test_模型能看到上一轮卡住了():
    """兜底消息进历史的第二个理由：不告诉模型，它下一轮会原样再试一遍死路。"""
    agent, _ = make_agent([], max_steps=2)
    agent.llm = _NeverStops()
    agent.run("停不下来的问题")

    seen_by_model = [m.content for m in agent.context.render()]
    assert any("最大步数" in c for c in seen_by_model)


# ============================================ 综合哨兵：所有路径的历史都合法
def test_所有退出路径产生的历史在anthropic格式下都合法():
    """不手写形状，而是让 Agent 真跑一遍各条路径，再统一验。

    这样以后新增退出路径（新的钩子、新的错误类型）会自动被这个测试覆盖，
    不用记得回来补形状。守的是「不变量」而不是「某个已知 bug」。

    不变量有两条：
        1. user / assistant 严格交替 —— API 不强制（会合并），是我们自己要的，
           保证每个问题有且只有一个回答、没有上一轮的残留粘在新问题上
        2. 每个 tool_use 必须有对应的 tool_result —— 这条是 API 硬约束，违反就 400
    第 1 条违反了不报错，所以只靠真实调用永远发现不了。
    """
    def normal():
        a, _ = make_agent([LLMResponse(text="答完了", stop_reason="end_turn")])
        a.run("问题")
        return a

    def with_tools():
        a, _ = make_agent([
            LLMResponse(text="查一下", stop_reason="tool_use",
                        tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
            LLMResponse(text="答完了", stop_reason="end_turn"),
        ])
        a.run("问题")
        return a

    def step_limit():
        a, _ = make_agent([], max_steps=2)
        a.llm = _NeverStops()
        a.run("停不下来")
        a.run("再问一个")
        return a

    def nudged():
        a, _ = make_agent(
            [LLMResponse(text="第一版", stop_reason="end_turn"),
             LLMResponse(text="补全了", stop_reason="end_turn")],
            finish_turn_hook=lambda o: (
                TurnDecision.keep_going("补上占比") if o.step == 1 else TurnDecision.end()
            ),
        )
        a.run("问题")
        return a

    def after_truncation():
        a, _ = make_agent([
            LLMResponse(text="半截", stop_reason="max_tokens"),
            LLMResponse(text="这次好了", stop_reason="end_turn"),
        ])
        with pytest.raises(OutputTruncated):
            a.run("问题一")
        a.run("问题二")
        return a

    def after_denial():
        a, _ = make_agent(
            [LLMResponse(text="要调工具", stop_reason="tool_use",
                         tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
             LLMResponse(text="换个办法", stop_reason="end_turn")],
            approval_hook=lambda call: (False, "测试拒绝"),
        )
        a.run("问题")
        return a

    scenarios = {
        "普通问答": normal, "带工具": with_tools, "步数耗尽": step_limit,
        "钩子推动继续": nudged, "截断后重试": after_truncation, "工具被拒": after_denial,
    }

    for name, build in scenarios.items():
        history = build().context.render()
        converted = AnthropicProvider.convert_messages(history)

        roles = [m["role"] for m in converted]
        dup = [(i, r) for i, (r, nxt) in enumerate(zip(roles, roles[1:])) if r == nxt]
        assert not dup, f"{name}: 角色没交替 {roles}（重复在 {dup}）"

        uses, results = set(), set()
        for m in converted:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "tool_use":
                        uses.add(b["id"])
                    elif b.get("type") == "tool_result":
                        results.add(b["tool_use_id"])
        assert uses == results, f"{name}: 悬空的工具调用 {uses ^ results}"
