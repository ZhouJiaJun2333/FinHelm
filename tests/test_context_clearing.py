"""第 2 步：清理较早的工具结果。

守五件事：
1. 什么时候清（阈值、保留最近几条、排除名单、省得不够就不清）
2. 清的是视图不是原件，而且 tool_call / tool_result 仍然配对
3. 锚点：清理之前量的作废，清理之后量的有效
4. 缓存友好：除了清理那一步，每次请求都是上一次请求的纯追加
5. 事务：一轮失败回滚时，清理状态跟着回滚
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from data_agent.core.agent import Agent
from data_agent.core.context import ClearOldToolResults, Context
from data_agent.core.errors import OutputTruncated
from data_agent.core.events import ContextEdited
from data_agent.core.messages import LLMResponse, Message, MessageMeta, ToolCall, Usage
from data_agent.core.tokens import estimate_context, estimate_message
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.openai_provider import OpenAICompatibleProvider
from data_agent.tools.base import Tool, ToolOutput
from data_agent.tools.registry import ToolRegistry

from fakes import ScriptedProvider, make_agent

# 一张像样的 SQL 结果表：约 1.4k token
BIG_RESULT = "| 华东 | 8100531.47 | 3017 |\n" * 60


def measure(msgs: list[Message]) -> int:
    return estimate_context(msgs).tokens


def add_tool_round(ctx, call_id: str, name: str = "run_sql", result: str = BIG_RESULT,
                   usage: Usage | None = None) -> None:
    ctx.add(Message(role="assistant", content="查一下",
                    tool_calls=[ToolCall(call_id, name, {"sql": f"SELECT {call_id}"})],
                    meta=MessageMeta(usage=usage)))
    ctx.add(Message.tool_result(call_id, result))


def make_ctx(n_calls: int, **kw) -> Context:
    kw.setdefault("trigger_tokens", 3_000)
    kw.setdefault("keep_recent", 2)
    kw.setdefault("clear_at_least", 500)
    ctx = Context([ClearOldToolResults(**kw)])
    ctx.add(Message.user("各区域销售额"))
    for i in range(n_calls):
        add_tool_round(ctx, f"c{i}")
    return ctx


def tool_contents(entries) -> list[str]:
    """工具结果的内容。传 render() 的结果或 ctx.history 都行（后者里的标记会被跳过）。"""
    return [e.content for e in entries if isinstance(e, Message) and e.role == "tool"]


def cleared(content: str) -> bool:
    return content.startswith(ClearOldToolResults.CLEARED_PREFIX)


# =============================================================== 什么时候清
def test_没超阈值什么都不做():
    ctx = make_ctx(1)
    assert ctx.maintain(measure) == []
    assert ctx.render() == ctx.history


def test_超阈值时清掉较早的_保留最近keep条():
    ctx = make_ctx(5, keep_recent=2)
    [edit] = ctx.maintain(measure)

    assert edit.description == "清理了 3 条较早的工具结果"
    assert edit.tokens_after < edit.tokens_before
    contents = tool_contents(ctx.render())
    assert [cleared(c) for c in contents] == [True, True, True, False, False]
    assert contents[3:] == [BIG_RESULT, BIG_RESULT]


def test_排除名单里的工具永远不清():
    ctx = Context([ClearOldToolResults(trigger_tokens=3_000, keep_recent=0,
                                       clear_at_least=500, exclude_tools=["describe_table"])])
    ctx.add(Message.user("q"))
    add_tool_round(ctx, "d1", name="describe_table")
    for i in range(3):
        add_tool_round(ctx, f"c{i}")

    ctx.maintain(measure)
    contents = tool_contents(ctx.render())
    assert contents[0] == BIG_RESULT
    assert [cleared(c) for c in contents] == [False, True, True, True]


def test_省得不够clear_at_least就不清():
    """清理会让缓存失效一次。省得太少，不如不动。"""
    ctx = make_ctx(5, clear_at_least=1_000_000)
    assert ctx.maintain(measure) == []
    assert not any(cleared(c) for c in tool_contents(ctx.render()))


def test_小结果不清_断点挪到第一条大结果():
    """最早的往往是几百字的汇总：清它省几十 token，缓存却从开头断掉。"""
    ctx = Context([ClearOldToolResults(trigger_tokens=3_000, keep_recent=2, clear_at_least=500,
                                       min_result_tokens=500)])
    ctx.add(Message.user("各区域销售额"))
    small = "| 华东 | 8100531.47 |\n" * 8                    # 两三百 token，比占位长
    add_tool_round(ctx, "s0", result=small)
    for i in range(4):
        add_tool_round(ctx, f"c{i}")

    ctx.maintain(measure)
    contents = tool_contents(ctx.render())
    assert contents[0] == small
    assert [cleared(c) for c in contents] == [False, True, True, False, False]


def test_清完降不到低水位就不清():
    """清理跟不上（清完还贴着门槛）时不清：隔几步又会过门槛、又断一次缓存，不如交给压缩。"""
    size = measure(make_ctx(5).render())
    freed = size - measure(_cleared_view(make_ctx(5)))
    after = size - freed

    too_high = make_ctx(5, low_water_ratio=(after - 1) / 3_000)
    assert too_high.maintain(measure) == []

    low_enough = make_ctx(5, low_water_ratio=(after + 1) / 3_000)
    assert low_enough.maintain(measure) != []


def test_强制整理时不看低水位():
    """API 已经报超长了，能省一点是一点。"""
    ctx = make_ctx(5, low_water_ratio=0.01)
    assert ctx.maintain(measure, force=True) != []


def _cleared_view(ctx: Context) -> list[Message]:
    ctx.maintain(measure)
    return ctx.render()


def test_清完远低于阈值_下一次请求不会再清():
    """一次清一批，之后是纯追加 —— 否则每次请求都改历史，缓存永远命中不了。"""
    ctx = make_ctx(5)
    assert ctx.maintain(measure) != []
    ctx.add(Message.assistant("好的"))
    assert ctx.maintain(measure) == []


# ====================================================== 视图 vs 原件、配对
def test_只改视图_原件一个字不动():
    ctx = make_ctx(5)
    ctx.maintain(measure)
    assert tool_contents(ctx.history) == [BIG_RESULT] * 5


def test_清理后tool_call和tool_result仍然配对():
    """清的是结果内容，不是消息本身 —— 删消息会留下悬空的 tool_call，直接 400。"""
    ctx = make_ctx(5)
    ctx.maintain(measure)
    converted = AnthropicProvider.convert_messages(ctx.render())

    uses, results = set(), set()
    for m in converted:
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    uses.add(b["id"])
                elif b.get("type") == "tool_result":
                    results.add(b["tool_use_id"])
    assert uses == results == {f"c{i}" for i in range(5)}


# ========================================================= 占位里的线索
def test_占位里留着工具给的摘要():
    """可恢复的压缩：内容可以丢，找回它的线索要留下。"""
    ctx = make_ctx(0, keep_recent=0)
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("s1", "run_sql", {"sql": "..."})]))
    ctx.add(Message.tool_result("s1", BIG_RESULT, summary="60 行 × 3 列（region, gmv, orders）"))
    for i in range(3):
        add_tool_round(ctx, f"c{i}")

    ctx.maintain(measure)
    placeholder = tool_contents(ctx.render())[0]
    assert cleared(placeholder)
    assert "60 行 × 3 列（region, gmv, orders）" in placeholder


def test_工具没给摘要时_至少说明原来有多大():
    ctx = make_ctx(4, keep_recent=0)
    ctx.maintain(measure)
    assert f"约 {len(BIG_RESULT)} 字符" in tool_contents(ctx.render())[0]


def test_摘要本身不发给模型():
    """summary 只在本地用。没被清理的时候，模型看到的是原文，不是原文加摘要。"""
    msg = Message.tool_result("x", "原文", summary="线索")
    for converted in (
        AnthropicProvider.convert_messages([msg]),
        OpenAICompatibleProvider.convert_messages([msg], system=None),
    ):
        assert "线索" not in str(converted)


def test_run_sql的摘要_大结果给形状_小结果直接给值():
    from data_agent.db.connection import QueryResult
    from data_agent.tools.sql.run_sql import _summarize

    table = QueryResult(["region", "gmv", "orders", "avg_discount"],
                        [("华东", 1, 2, 0.1)] * 42, truncated=False, elapsed_ms=5)
    assert _summarize(table) == "42 行 × 4 列（region, gmv, orders, avg_discount）"

    # 一行的聚合结果，线索本身就几乎等于原件，模型不用再查
    agg = QueryResult(["total", "null_region"], [(200, None)], truncated=False, elapsed_ms=1)
    assert _summarize(agg) == "1 行：total=200, null_region=NULL"

    truncated = QueryResult(["id"], [(1,), (2,)], truncated=True, elapsed_ms=1)
    assert _summarize(truncated).endswith("当时已被截断")


def test_输出过长被截断时摘要还在():
    out = ToolOutput(True, "x" * 10_000, summary="线索").capped(100)
    assert out.summary == "线索"


def test_同一条消息每次生成的占位完全一样():
    """占位一变，前缀就变，缓存就废。"""
    msg = Message.tool_result("x", BIG_RESULT, summary="60 行")
    assert ClearOldToolResults.placeholder(msg) == ClearOldToolResults.placeholder(msg)


# ================================================================== 锚点
def test_清理位置之后的旧锚点作废_之前的保留():
    ctx = Context([ClearOldToolResults(trigger_tokens=3_000, keep_recent=1, clear_at_least=500)])
    ctx.add(Message.user("q"))
    ctx.add(Message(role="assistant", content="先看看", meta=MessageMeta(usage=Usage(input=900, output=10))))
    ctx.add(Message.user("继续"))
    add_tool_round(ctx, "c0", usage=Usage(input=1_000, output=10))
    add_tool_round(ctx, "c1", usage=Usage(input=2_500, output=10))
    add_tool_round(ctx, "c2", usage=Usage(input=4_000, output=10))

    ctx.maintain(measure)
    usages = [m.meta.usage for m in ctx.render() if m.role == "assistant"]

    # 第 1 条在被清理的 c0 结果之前，量它的时候视图和现在一样 → 有效
    # 第 2 条（发起 c0 的那条）也在 c0 结果之前 → 有效
    # 第 3、4 条量的时候 c0 还是原文，现在变成了占位 → 作废
    assert usages == [Usage(input=900, output=10), Usage(input=1_000, output=10), None, None]
    assert estimate_context(ctx.render()).anchor_index == 3


def test_清理之后才加进来的锚点有效():
    ctx = make_ctx(5)
    ctx.maintain(measure)
    fresh = Usage(input=2_000, output=30)
    ctx.add(Message(role="assistant", content="答完了", meta=MessageMeta(usage=fresh)))

    rendered = ctx.render()
    assert rendered[-1].meta.usage == fresh
    assert estimate_context(rendered).usage_tokens == fresh.context_tokens


# ============================================================ 接进 Agent
class BigTool(Tool):
    name = "run_sql"
    description = "返回一张大表"

    class Args(BaseModel):
        sql: str = "SELECT 1"

    def run(self, args: Args) -> ToolOutput:
        return ToolOutput(True, BIG_RESULT, summary=f"60 行 × 3 列（{args.sql}）")


def calls(n: int) -> list[LLMResponse]:
    return [
        LLMResponse(text="", stop_reason="tool_use",
                    tool_calls=[ToolCall(f"c{i}", "run_sql", {"sql": f"SELECT {i}"})])
        for i in range(n)
    ]


def make_clearing_agent(script, **ctx_kw):
    """带清理上下文、工具是一张大表的 Agent。返回 (agent, 事件)。"""
    ctx_kw.setdefault("trigger_tokens", 3_000)
    ctx_kw.setdefault("keep_recent", 1)
    ctx_kw.setdefault("clear_at_least", 500)
    return make_agent(script, tools=[BigTool()],
                      context=Context([ClearOldToolResults(**ctx_kw)]), max_steps=20)


def test_传进去的空上下文不会被悄悄换掉():
    """回归：Agent 曾经写成 `context or 默认上下文()`。当时的上下文类定义了
    __len__，空的上下文是 False，配好的清理策略被换成了全量保留 —— 不报错，就是不生效。
    """
    ctx = Context()
    agent = Agent(llm=ScriptedProvider(), tools=ToolRegistry([]),
                  system_prompt="", context=ctx)
    assert agent.context is ctx


def test_agent在请求前清理_模型看到的是占位():
    agent, events = make_clearing_agent(calls(4) + [LLMResponse(text="完", stop_reason="end_turn")])
    agent.run("各区域销售额")

    assert any(isinstance(e, ContextEdited) for e in events)
    assert any(cleared(c) for c in tool_contents(agent.llm.seen[-1]))
    # 工具给的摘要一路穿过 ToolOutput → Agent → Message，最后出现在占位里
    assert "60 行 × 3 列（SELECT 0）" in tool_contents(agent.llm.seen[-1])[0]


def test_除了清理那一步_每次请求都是上一次请求的纯追加():
    """这是「缓存友好」的可测定义：前缀不变，缓存才能命中。

    如果每次请求都改历史（比如「永远只留最近 3 条」的滑动窗口），
    这个测试里违规次数会等于请求次数。
    """
    agent, events = make_clearing_agent(calls(8) + [LLMResponse(text="完", stop_reason="end_turn")])
    agent.run("各区域销售额")

    def shape(msgs):
        # 只比较真正发给 API 的内容。usage 不会发出去，不算。
        return [(m.role, m.content, m.tool_call_id, tuple(c.id for c in m.tool_calls))
                for m in msgs]

    rewrites = [
        i for i, (prev, cur) in enumerate(zip(agent.llm.seen, agent.llm.seen[1:]))
        if shape(cur)[: len(prev)] != shape(prev)
    ]
    clearings = sum(1 for e in events if isinstance(e, ContextEdited))
    assert clearings >= 2, "测试没测到东西：至少应该触发两次清理"
    # 每一次改历史都是一次有记录的清理，没有别的地方偷偷改
    assert len(rewrites) == clearings
    # 清完之后至少要有一次纯追加的请求，缓存才有机会命中。
    # 滑动窗口式的「每次清一条」会在这里挂掉：它连续每一步都在改历史。
    assert all(b - a > 1 for a, b in zip(rewrites, rewrites[1:])), rewrites


def test_一轮失败回滚时清理状态也回滚():
    agent, _ = make_clearing_agent(
        calls(4) + [LLMResponse(text="写到一半", stop_reason="max_tokens")]
    )
    with pytest.raises(OutputTruncated):
        agent.run("问题")

    assert agent.context.render() == []
    assert agent.context.status() == []


def test_清理后历史在anthropic格式下仍然合法():
    agent, _ = make_clearing_agent(calls(6) + [LLMResponse(text="完", stop_reason="end_turn")])
    agent.run("问题一")
    agent.run("问题二")

    converted = AnthropicProvider.convert_messages(agent.context.render())
    roles = [m["role"] for m in converted]
    assert not any(a == b for a, b in zip(roles, roles[1:])), roles
    for m in converted:
        if isinstance(m["content"], list):
            for b in m["content"]:
                assert b.get("type") != "tool_result" or b["content"], "tool_result 不能是空的"


# ============================================ 占位比原文还长的，不换
def _history_with(result: str) -> Context:
    ctx = Context([ClearOldToolResults(trigger_tokens=1, keep_recent=0, clear_at_least=1)])
    ctx.add(Message.user("q"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "run_sql", {})]))
    ctx.add(Message.tool_result("c1", result))
    return ctx


def test_审批拒绝的结果不会被换成_重新调用即可():
    """拒绝消息只有一句话，换成占位反而更长；更糟的是占位会说「重新调用一次即可」，
    等于鼓励模型再试一次被拒绝的操作。"""
    ctx = _history_with("用户拒绝执行：这张表有敏感字段")
    assert ctx.maintain(measure, force=True) == []
    assert ctx.render()[-1].content == "用户拒绝执行：这张表有敏感字段"


def test_很小的结果不换_大结果照常换():
    small = _history_with("| total |\n| 4242 |")
    assert small.maintain(measure, force=True) == []

    big = _history_with("| 华东 | 8100531.47 |\n" * 80)
    [event] = big.maintain(measure, force=True)
    assert big.render()[-1].content.startswith(ClearOldToolResults.CLEARED_PREFIX)
    assert event.tokens_after < event.tokens_before
