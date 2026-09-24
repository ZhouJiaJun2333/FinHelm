"""CompactHistory：清理之后还太大，就把较早的回合换成摘要。

写摘要用假函数（FakeSummarizer），不打真实 API；它会记下每次收到了什么。
"""

from __future__ import annotations

import pytest

from data_agent.core.agent import TurnDecision
from data_agent.core.context import (
    ClearOldToolResults,
    CompactHistory,
    Context,
    HistoryCompacted,
    Summary,
    llm_summarizer,
)
from data_agent.core.context.compaction import TOOL_RESULT_CLIP, serialize
from data_agent.core.errors import CompactionFailed
from data_agent.core.events import ContextEdited
from data_agent.core.messages import LLMResponse, Message, MessageMeta, ToolCall, Usage
from data_agent.core.tokens import estimate_context

from fakes import ScriptedProvider, make_agent


def measure(msgs: list[Message]) -> int:
    return estimate_context(msgs).tokens


class FakeSummarizer:
    """第 n 次调用返回「摘要n」，并记下每次收到的消息。"""

    def __init__(self, usage: Usage = Usage()) -> None:
        self.inputs: list[list[Message]] = []
        self.usage = usage

    def __call__(self, messages: list[Message]) -> Summary:
        self.inputs.append(messages)
        return Summary(f"摘要{len(self.inputs)}", self.usage)


def compactor(fake: FakeSummarizer, **kw) -> CompactHistory:
    """阈值调到极低：只要有东西就压；只保留当前这一轮。"""
    kw.setdefault("trigger_tokens", 1)
    kw.setdefault("keep_recent_tokens", 1)
    return CompactHistory(summarize=fake, **kw)


def add_turn(ctx: Context, n: int, *, with_tool: bool = False) -> None:
    ctx.add(Message.user(f"问题{n}"))
    if with_tool:
        ctx.add(Message(role="assistant", tool_calls=[ToolCall(f"c{n}", "run_sql", {"sql": f"SELECT {n}"})]))
        ctx.add(Message.tool_result(f"c{n}", f"结果{n}"))
    ctx.add(Message.assistant(f"答案{n}"))


def contents(ctx: Context) -> list[str]:
    return [m.content for m in ctx.render()]


# ============================================================== 什么时候压
def test_没超阈值不压缩():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake, trigger_tokens=10_000)])
    for n in range(3):
        add_turn(ctx, n)
    assert ctx.maintain(measure) == []
    assert fake.inputs == []


def test_只有当前这一轮时没有可压的():
    """至少保留当前这一轮 —— 只有一轮，就不该花钱去写摘要。"""
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    add_turn(ctx, 0, with_tool=True)
    assert ctx.maintain(measure) == []
    assert fake.inputs == []


# ============================================================== 压成什么样
def test_较早的回合换成摘要_最近的保留原文():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    for n in range(3):
        add_turn(ctx, n)
    ctx.add(Message.user("问题3"))                  # 当前这一轮，刚提问

    [event] = ctx.maintain(measure)
    assert event.description == "把较早的 3 轮对话压缩成了摘要（保留最近 1 轮原文）"

    [only] = ctx.render()
    assert only.role == "user"
    assert only.content.startswith(CompactHistory.SUMMARY_HEADER + "摘要1")
    assert only.content.endswith("问题3"), "摘要并进了第一条保留的提问前面"
    # 交给模型写摘要的，正好是被压掉的那 3 轮
    assert [m.content for m in fake.inputs[0]] == [
        "问题0", "答案0", "问题1", "答案1", "问题2", "答案2",
    ]


def test_压缩之后上下文确实变小():
    ctx = Context([compactor(FakeSummarizer())])
    for n in range(3):
        ctx.add(Message.user(f"问题{n}"))
        ctx.add(Message.assistant("华东大区的销售额分析如下……" * 50))
    ctx.add(Message.user("问题3"))
    [event] = ctx.maintain(measure)
    assert event.tokens_after < event.tokens_before / 10


def test_保留多少按token攒_按回合取整():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake, keep_recent_tokens=10_000)])   # 够装下好几轮
    for n in range(3):
        add_turn(ctx, n)
    assert ctx.maintain(measure) == [], "全部都在保留范围内，没有可压的"

    ctx2 = Context([compactor(FakeSummarizer(), keep_recent_tokens=6)])
    for n in range(4):
        add_turn(ctx2, n)                            # 每轮估算约 4 token
    [event] = ctx2.maintain(measure)
    assert "保留最近 1 轮" in event.description     # 第二轮就超了 6，只留 1 轮


def test_压缩后角色严格交替_工具调用成对():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    for n in range(3):
        add_turn(ctx, n, with_tool=True)
    ctx.add(Message.user("问题3"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c3", "run_sql", {})]))
    ctx.add(Message.tool_result("c3", "结果3"))
    ctx.maintain(measure)

    view = ctx.render()
    assert [m.role for m in view] == ["user", "assistant", "tool"]
    assert view[2].tool_call_id == "c3"


def test_nudge不会被当成切口():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    add_turn(ctx, 0)
    ctx.add(Message.user("问题1"))
    ctx.add(Message.assistant("答了一半"))
    ctx.add(Message.user("还缺占比，补上").with_meta(synthetic=True))

    ctx.maintain(measure)
    view = contents(ctx)
    assert view[0].endswith("问题1"), "切在真人提问上，这一轮的问题还在"
    assert view[1:] == ["答了一半", "还缺占比，补上"]


# ============================================================ 标记和历史
def test_摘要是历史里的一个标记_原文一条不少():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    for n in range(3):
        add_turn(ctx, n)
    before = ctx.history
    ctx.maintain(measure)

    after = ctx.history
    assert after[:-1] == before
    assert after[-1] == HistoryCompacted(summary="摘要1", kept_turns=1, compacted_turns=2)
    assert all(isinstance(m, Message) for m in ctx.render())


def test_压缩之后新来的消息跟在后面():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake, trigger_tokens=10_000)])
    for n in range(3):
        add_turn(ctx, n)
    ctx.add(HistoryCompacted(summary="摘要", kept_turns=1, compacted_turns=2))
    add_turn(ctx, 3)
    view = contents(ctx)
    assert view[0].endswith("问题2")
    assert view[1:] == ["答案2", "问题3", "答案3"]


def test_第二次压缩的输入里带着上一份摘要():
    """滚动摘要：新摘要在旧摘要基础上写，不是只看最近几轮从零写。"""
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    for n in range(3):
        add_turn(ctx, n)
    ctx.maintain(measure)                            # 摘要1 = 问题0、1
    add_turn(ctx, 3)
    ctx.maintain(measure)                            # 摘要2 = 摘要1 + 第 2 轮

    second = fake.inputs[1]
    assert second[0].content.startswith(CompactHistory.SUMMARY_HEADER + "摘要1")
    assert second[0].content.endswith("问题2")
    assert contents(ctx)[0].startswith(CompactHistory.SUMMARY_HEADER + "摘要2")
    assert contents(ctx)[0].endswith("问题3")


def test_回滚会连摘要标记一起撤掉():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    for n in range(3):
        add_turn(ctx, n)
    snap = ctx.snapshot()
    ctx.maintain(measure)
    ctx.restore(snap)
    assert contents(ctx)[0] == "问题0"
    assert ctx.status() == []


def test_压缩之后之前的锚点作废():
    """视图变了，锚点量的就不是现在这份了 —— 这是 Context 统一处理的，工序不用管。"""
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    ctx.add(Message.user("问题0"))
    ctx.add(Message(role="assistant", content="答案0", meta=MessageMeta(usage=Usage(input=500))))
    add_turn(ctx, 1)
    assert ctx.render()[1].meta.usage == Usage(input=500)

    ctx.maintain(measure)
    assert all(m.meta.usage is None for m in ctx.render())


def test_状态里能看到压缩了几次():
    ctx = Context([compactor(FakeSummarizer())])
    assert ctx.status() == []
    ctx.add(HistoryCompacted(summary="一二三", kept_turns=1, compacted_turns=2))
    assert ctx.status() == ["已压缩 1 次，当前摘要约 3 字（原文还在历史里，只是不再发给模型）"]


# ============================================================ 和清理的配合
def test_排在清理后面_写摘要看到的是清理过的视图():
    fake = FakeSummarizer()
    big = "| 华东 | 8100531.47 |\n" * 80
    ctx = Context([
        ClearOldToolResults(trigger_tokens=1, keep_recent=0, clear_at_least=1),
        compactor(fake),
    ])
    ctx.add(Message.user("问题0"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c0", "run_sql", {})]))
    ctx.add(Message.tool_result("c0", big))
    ctx.add(Message.assistant("答案0"))
    ctx.add(Message.user("问题1"))

    events = ctx.maintain(measure)
    assert len(events) == 2                            # 先清理，再压缩
    tool_input = [m for m in fake.inputs[0] if m.role == "tool"][0]
    assert tool_input.content.startswith(ClearOldToolResults.CLEARED_PREFIX)


# ============================================================ 写摘要的请求
def test_对话序列化成文本_工具结果太长就截断():
    text = serialize([
        Message.user("华东卖了多少"),
        Message(role="assistant", content="我查一下", tool_calls=[
            ToolCall("c1", "run_sql", {"sql": "SELECT sum(gmv) FROM orders"}),
        ]),
        Message.tool_result("c1", "x" * (TOOL_RESULT_CLIP + 500)),
        Message.assistant("810 万"),
    ])
    assert "[用户]\n华东卖了多少" in text
    assert '[助手调用工具 run_sql]\n{"sql": "SELECT sum(gmv) FROM orders"}' in text
    assert "后面省略 500 字符" in text
    assert text.endswith("[助手]\n810 万")


def test_llm写摘要_不带工具_只发一条user():
    llm = ScriptedProvider([LLMResponse(text="  摘要正文  ", stop_reason="end_turn",
                                        usage=Usage(input=300, output=50))])
    summary = llm_summarizer(llm)([Message.user("问题"), Message.assistant("答案")])
    assert summary == Summary("摘要正文", Usage(input=300, output=50))
    [request] = llm.seen
    assert [m.role for m in request] == ["user"]
    assert "[用户]\n问题" in request[0].content


@pytest.mark.parametrize("response", [
    LLMResponse(text="写到一半", stop_reason="max_tokens"),
    LLMResponse(text="   ", stop_reason="end_turn"),
])
def test_摘要被截断或为空时报错_不静默跳过(response):
    with pytest.raises(CompactionFailed):
        llm_summarizer(ScriptedProvider([response]))([Message.user("q")])


# ============================================================ 放进 Agent
def test_Agent里跑起来_摘要的花费记进会话用量():
    fake = FakeSummarizer(usage=Usage(input=1000, output=200))
    agent, events = make_agent(
        [LLMResponse(text="答", stop_reason="end_turn", usage=Usage(input=10, output=1))],
        context=Context([compactor(fake)]),
    )
    agent.run("问题1")
    agent.run("问题2")                               # 请求前发现超标，压掉第 1 轮

    assert agent.llm.seen[-1][0].content.startswith(CompactHistory.SUMMARY_HEADER + "摘要1")
    assert [e for e in events if isinstance(e, ContextEdited)][0].usage == Usage(input=1000, output=200)
    assert agent.session_usage == Usage(input=1020, output=202)


def test_摘要失败时这一轮回滚():
    def broken(messages):
        raise CompactionFailed("写摘要的请求返回了空内容。")

    agent, _ = make_agent([LLMResponse(text="答", stop_reason="end_turn")],
                          context=Context([CompactHistory(broken, trigger_tokens=1, keep_recent_tokens=1)]))
    agent.run("问题1")
    before = agent.context.history
    with pytest.raises(CompactionFailed):
        agent.run("问题2")
    assert agent.context.history == before


def test_finish_turn继续时_压缩不会把当前这轮的问题切掉():
    fake = FakeSummarizer()
    hook = lambda o: TurnDecision.keep_going("补上") if o.step == 1 else TurnDecision.end()
    agent, _ = make_agent(
        [LLMResponse(text="答", stop_reason="end_turn")],
        finish_turn_hook=hook, context=Context([compactor(fake)]),
    )
    agent.run("问题1")
    agent.run("问题2")
    last_request = [m.content for m in agent.llm.seen[-1]]
    assert last_request[0].endswith("问题2")
    assert last_request[1:] == ["答", "补上"]
