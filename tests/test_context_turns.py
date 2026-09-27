"""KeepRecentTurns：按回合裁剪。

回合 = 一次真人提问 + 它引出的所有东西（模型回复、工具调用和结果、Agent 补的 nudge）。
"""

from __future__ import annotations

from data_agent.core.agent import TurnDecision
from data_agent.core.context import Context, KeepRecentTurns
from data_agent.core.messages import LLMResponse, Message, ToolCall
from data_agent.llm.anthropic_provider import AnthropicProvider

from fakes import make_agent


def test_按回合裁剪不会拆散工具调用():
    """裁剪的单位必须是「回合」，不能是「消息」，否则 tool_call 和 tool_result 会被拆开。"""
    ctx = Context([KeepRecentTurns(max_turns=1)])
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


def test_nudge不算新回合_不会把用户的问题切掉():
    """finish_turn 让模型继续时补的 nudge 也是 role="user"。

    以前按 role 找回合起点，nudge 被当成新回合：只留 1 轮时，窗口切在 nudge 上，
    模型看到的只剩「还缺占比，补上」，不知道要补的是什么问题的占比。
    """
    script = [
        LLMResponse(text="华东 810 万", stop_reason="end_turn"),
        LLMResponse(text="华东 810 万，占 32%", stop_reason="end_turn"),
    ]
    hook = lambda o: TurnDecision.keep_going("还缺占比，补上") if o.step == 1 else TurnDecision.end()
    agent, _ = make_agent(script, finish_turn_hook=hook,
                          context=Context([KeepRecentTurns(max_turns=1)]))
    agent.run("华东卖了多少")

    second_request = agent.llm.seen[1]
    assert [m.content for m in second_request] == [
        "华东卖了多少", "华东 810 万", "还缺占比，补上",
    ]


def test_下一次真人提问才开始新回合():
    """nudge 所在的那轮，在下一次真人提问之后整体被裁掉。"""
    script = [
        LLMResponse(text="答1", stop_reason="end_turn"),
        LLMResponse(text="答1（补全）", stop_reason="end_turn"),
        LLMResponse(text="答2", stop_reason="end_turn"),
    ]
    hook = lambda o: TurnDecision.keep_going("补上") if o.response.text == "答1" else TurnDecision.end()
    agent, _ = make_agent(script, finish_turn_hook=hook,
                          context=Context([KeepRecentTurns(max_turns=1)]))
    agent.run("问题1")
    agent.run("问题2")

    assert [m.content for m in agent.llm.seen[-1]] == ["问题2"]


def test_Agent补的消息都标了synthetic_但发出去和普通消息一样():
    """nudge 和步数用完时的收尾提示是 Agent 自己补的；真人提问和模型回复不是。"""
    agent, _ = make_agent(
        [LLMResponse(text="还没完", stop_reason="end_turn")],
        finish_turn_hook=lambda o: TurnDecision.keep_going("继续"),
        max_steps=2,
    )
    agent.run("问题")

    flags = [(m.role, m.content, m.meta.synthetic) for m in agent.context.history]
    assert flags == [
        ("user", "问题", False),
        ("assistant", "还没完", False),
        ("user", "继续", True),
        ("assistant", "还没完", False),           # 最后一步：不补 nudge，接着收尾
        ("user", flags[4][1], True),              # 收尾提示
        ("assistant", "还没完", False),           # 收尾的回答是模型真说的
    ]
    assert flags[4][1].startswith("[步数用完了（2 步）")
    # synthetic 在 meta 里，provider 不看 meta —— 请求体和普通 user 消息完全一样
    nudge = agent.context.history[2]
    assert (AnthropicProvider.convert_messages([nudge])
            == AnthropicProvider.convert_messages([Message.user("继续")]))
