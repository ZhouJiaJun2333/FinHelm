"""Context 本身的机制：工序组合、通用的锚点失效、状态、事务。

具体某一种工序（比如清理工具结果）的测试在各自的文件里；这里只测
「任何工序插进来都成立」的东西。
"""

from __future__ import annotations

from data_agent.core.context import (
    ClearOldToolResults,
    Context,
    ContextEdit,
    KeepRecentTurns,
    Marker,
    ToolResultsCleared,
)
from data_agent.core.messages import Message, MessageMeta, ToolCall, Usage
from data_agent.core.tokens import estimate_context


def measure(msgs: list[Message]) -> int:
    return estimate_context(msgs).tokens


def asst(text: str, tokens: int, calls: list[ToolCall] = ()) -> Message:
    return Message(role="assistant", content=text, tool_calls=list(calls),
                   meta=MessageMeta(usage=Usage(input=tokens)))


def usages(ctx: Context) -> list[Usage | None]:
    return [m.meta.usage for m in ctx.render() if m.role == "assistant"]


class Shout(ContextEdit):
    """测试用的自定义工序：把工具结果改成大写。

    它完全不知道锚点的存在 —— 用来证明锚点失效是 Context 统一处理的，
    写新工序的人不用操心。
    """

    def apply(self, entries):
        return [
            Message.tool_result(e.tool_call_id, e.content.upper())
            if isinstance(e, Message) and e.role == "tool" else e
            for e in entries
        ]


class Copy(ContextEdit):
    """什么都不改、但每条消息都换成新对象的工序。"""

    def apply(self, entries):
        return [e.with_meta() if isinstance(e, Message) else e for e in entries]


# =================================================================== 组合
def test_不传工序就是全量保留():
    ctx = Context()
    for i in range(10):
        ctx.add(Message.user(f"m{i}"))
    assert [m.content for m in ctx.render()] == [f"m{i}" for i in range(10)]


def test_自定义工序直接插进来就能用():
    ctx = Context([Shout()])
    ctx.add(Message.user("q"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {})]))
    ctx.add(Message.tool_result("c1", "abc"))
    assert ctx.render()[-1].content == "ABC"
    assert ctx.history[-1].content == "abc", "原件不动"


def test_工序按顺序套用_后一道看到的是前一道的输出():
    """先裁剪再清理：被裁掉的回合，清理工序根本看不见，也就不会去清它。"""
    big = "| 华东 | 8100531.47 |\n" * 80
    clear = ClearOldToolResults(trigger_tokens=100, keep_recent=0, clear_at_least=1)
    ctx = Context([KeepRecentTurns(max_turns=1), clear])
    for turn in range(2):
        ctx.add(Message.user(f"问题{turn}"))
        ctx.add(Message(role="assistant", tool_calls=[ToolCall(f"c{turn}", "run_sql", {})]))
        ctx.add(Message.tool_result(f"c{turn}", big))
        ctx.add(Message.assistant(f"答案{turn}"))

    [event] = ctx.maintain(measure)
    assert event.description == "清理了 1 条较早的工具结果"   # 只清了留在窗口里的 c1
    assert markers(ctx) == [ToolResultsCleared(frozenset({"c1"}))]


# ========================================================== 通用的锚点失效
def test_任何工序改了前面的内容_后面的旧锚点都自动作废():
    ctx = Context([Shout()])
    ctx.add(Message.user("q"))
    ctx.add(asst("查", 100, [ToolCall("c1", "echo", {})]))
    ctx.add(Message.tool_result("c1", "abc"))   # 这条被 Shout 改过
    ctx.add(asst("答", 200))

    # 两条锚点都是在「已经被 Shout 过」的视图上量的 —— 视图没再变，都有效
    assert usages(ctx) == [Usage(input=100), Usage(input=200)]

    # 换一道会改前面内容的工序（模拟以后调整了策略），第二条锚点量的是旧视图
    ctx.edits = []
    assert usages(ctx) == [Usage(input=100), None]


def test_按回合裁剪后_量的时候包含被裁掉内容的锚点作废():
    """顺带修正：以前的 TurnWindowContext 不处理锚点，裁掉前面的回合后
    还拿「含被裁内容时」量的 usage 当基准，估出来的上下文偏大。
    """
    window = KeepRecentTurns(max_turns=5)
    ctx = Context([window])
    ctx.add(Message.user("问题0"))
    ctx.add(asst("答0", 100))
    ctx.add(Message.user("问题1"))
    ctx.add(asst("答1", 200))
    assert usages(ctx) == [Usage(input=100), Usage(input=200)]

    window.max_turns = 1                         # 第 0 轮被裁掉
    assert usages(ctx) == [None]


def test_锚点失效看的是内容_不是对象身份():
    """工序每次都生成新对象、但内容没变 —— 发出去的请求一样，锚点就该有效。"""
    ctx = Context([Copy()])
    ctx.add(Message.user("q"))
    ctx.add(asst("答", 100))
    ctx.add(Message.user("再问"))
    assert usages(ctx) == [Usage(input=100)]


def test_meta变化不影响锚点():
    """meta 不发给模型，改了它不算「前面的内容变了」。"""
    ctx = Context()
    ctx.add(Message.tool_result("x", "结果", summary="线索一"))
    ctx.add(asst("答", 100))
    ctx._history[0] = ctx._history[0].with_meta(summary="线索二")   # 直接改内部，模拟 meta 变化
    assert usages(ctx) == [Usage(input=100)]


# ====================================================== 状态记在历史里
def markers(ctx: Context) -> list[Marker]:
    return [e for e in ctx.history if isinstance(e, Marker)]


def test_工序的决定作为标记记进历史_标记不发给模型():
    big = "| 华东 | 8100531.47 |\n" * 80
    ctx = Context([ClearOldToolResults(trigger_tokens=100, keep_recent=0, clear_at_least=1)])
    ctx.add(Message.user("q"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "run_sql", {})]))
    ctx.add(Message.tool_result("c1", big))
    ctx.maintain(measure)

    assert markers(ctx) == [ToolResultsCleared(frozenset({"c1"}))]
    assert isinstance(ctx.history[-1], ToolResultsCleared), "标记追加在它影响的消息之后"
    assert all(isinstance(m, Message) for m in ctx.render())


def test_工序对象不持有状态_可以给多个会话共用():
    """以前清理记录存在工序对象里，同一个对象给两个会话用，清理会串到另一个会话去。"""
    shared = ClearOldToolResults()
    a, b = Context([shared]), Context([shared])
    a.add(ToolResultsCleared(frozenset({"c1"})))
    assert a.status() != []
    assert b.status() == []


def test_快照就是一份历史_标记跟着一起回滚():
    ctx = Context([ClearOldToolResults()])
    ctx.add(Message.user("q"))
    snap = ctx.snapshot()

    ctx.add(Message.user("q2"))
    ctx.add(ToolResultsCleared(frozenset({"c1"})))
    ctx.restore(snap)

    assert ctx.history == [Message.user("q")]
    assert ctx.status() == []


def test_状态由各道工序从标记里数出来_没事可说时为空():
    ctx = Context([KeepRecentTurns(), ClearOldToolResults()])
    assert ctx.status() == []

    ctx.add(ToolResultsCleared(frozenset({"a", "b"})))
    ctx.add(ToolResultsCleared(frozenset({"c"})))
    assert ctx.status() == ["已清理的旧工具结果：3 条（原件还在，只是不再发给模型）"]


def test_清空历史就清空了一切():
    ctx = Context([ClearOldToolResults()])
    ctx.add(ToolResultsCleared(frozenset({"c1"})))
    ctx.clear()
    assert ctx.status() == [] and ctx.history == []


def test_按回合裁剪时_标记和它影响的消息一起被切掉():
    ctx = Context([KeepRecentTurns(max_turns=1), ClearOldToolResults()])
    ctx.add(Message.user("问题0"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c0", "run_sql", {})]))
    ctx.add(Message.tool_result("c0", "旧结果"))
    ctx.add(ToolResultsCleared(frozenset({"c0"})))
    ctx.add(Message.assistant("答0"))
    ctx.add(Message.user("问题1"))

    assert [m.content for m in ctx.render()] == ["问题1"]


def test_不改变视图的标记不会让锚点作废():
    """锚点看的是发出去的内容，不是历史里多了什么。"""
    ctx = Context([ClearOldToolResults()])
    ctx.add(Message.user("q"))
    ctx.add(asst("答", 100))
    ctx.add(ToolResultsCleared(frozenset({"不存在的调用"})))
    assert usages(ctx) == [Usage(input=100)]
