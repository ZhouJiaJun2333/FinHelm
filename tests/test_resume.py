"""轮内检查点：一轮断了照样回滚，但进度留在 agent.interrupted，resume() 从下一步接着跑。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from data_agent.core.agent import CONTINUE_NUDGE, INTERRUPTED_RESULT, REPEAT_NOTE, WRAP_UP, InterruptedTurn
from data_agent.core.context import turn_starts
from data_agent.core.events import LLMResponded, StepLimitReached, ToolFinished, TurnResumed
from data_agent.core.messages import LLMResponse, Message, ToolCall, Usage
from data_agent.core.tools import Tool

from fakes import EchoTool, make_agent


def call(n: int, text: str | None = None) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(f"c{n}", "echo", {"text": text or str(n)})],
                       stop_reason="tool_use", usage=Usage(input=100 * n, output=10))


FINAL = LLMResponse(text="答案是 42", stop_reason="end_turn")


def assert_paired(messages) -> None:
    calls = [c.id for m in messages for c in m.tool_calls]
    results = [m.tool_call_id for m in messages if m.role == "tool"]
    assert sorted(calls) == sorted(results)


def tool_runs(events) -> list[str]:
    return [e.content for e in events if isinstance(e, ToolFinished)]


# ================================================================ API 报错
def test_API报错_历史照样回滚_进度留下_接着跑不重做前面的步():
    agent, events = make_agent([call(1), call(2), RuntimeError("429 限流"), call(3), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")

    assert agent.context.render() == [], "正式历史回滚，和以前一样"
    turn = agent.interrupted
    assert turn.steps == 2 and turn.reason == "RuntimeError: 429 限流" and turn.question == "算一下"
    assert_paired([e for e in turn.entries])

    assert agent.resume() == "答案是 42"
    assert agent.interrupted is None
    assert agent.llm.calls == 5, "前两步没有重新请求"
    assert tool_runs(events) == ["echo: 1", "echo: 2", "echo: 3"], "工具没有重跑"
    assert [e.step for e in events if isinstance(e, LLMResponded)] == [1, 2, 3, 4], "步数接着数"
    assert next(e for e in events if isinstance(e, TurnResumed)).steps == 2
    history = agent.context.render()
    assert len(turn_starts(history)) == 1 and history[0].content == "算一下"
    assert_paired(history)


def test_接着跑时模型看到的就是断之前的历史():
    agent, _ = make_agent([call(1), RuntimeError("断网"), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    agent.resume()
    before, after = agent.llm.seen[1], agent.llm.seen[2]
    assert after == before, "断的那次请求和接着跑的第一次请求，发的是同一份"


def test_步数用剩下的_用完照样收尾():
    agent, events = make_agent([call(1), call(2), RuntimeError("503"), call(3), FINAL], max_steps=3)
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    assert agent.resume() == "答案是 42"          # 第 3 步调工具，接着就是收尾
    limit = next(e for e in events if isinstance(e, StepLimitReached))
    assert limit.max_steps == 3 and limit.wrapped_up
    assert sum(isinstance(e, LLMResponded) for e in events) == 4


def test_接着跑又断了_进度累加_还能再接着跑():
    agent, events = make_agent([call(1), RuntimeError("a"), call(2), RuntimeError("b"), call(3), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    with pytest.raises(RuntimeError):
        agent.resume()
    assert agent.interrupted.steps == 2 and agent.interrupted.reason == "RuntimeError: b"
    assert agent.context.render() == []
    assert agent.resume() == "答案是 42"
    assert tool_runs(events) == ["echo: 1", "echo: 2", "echo: 3"]


# ================================================================ Ctrl-C = 暂停
class PausingTool(Tool):
    """第一次执行时用户按了 Ctrl-C。"""

    name = "slow"
    description = "跑得慢"

    class Args(BaseModel):
        pass

    def __init__(self) -> None:
        self.runs = 0

    def run(self, args: Args) -> str:
        self.runs += 1
        if self.runs == 1:
            raise KeyboardInterrupt
        return "跑完了"


def test_CtrlC打在工具执行中间_补结果未知_不重做这次调用():
    tool = PausingTool()
    slow = LLMResponse(text="", tool_calls=[ToolCall("s1", "slow", {}), ToolCall("s2", "slow", {})],
                       stop_reason="tool_use")
    agent, _ = make_agent([call(1), slow, FINAL], tools=[EchoTool(), tool])
    with pytest.raises(KeyboardInterrupt):
        agent.run("算一下")

    turn = agent.interrupted
    assert turn.steps == 2 and turn.reason == "KeyboardInterrupt"
    results = {e.tool_call_id: e for e in turn.entries if getattr(e, "role", "") == "tool"}
    assert results["s1"].content == INTERRUPTED_RESULT and results["s1"].is_error
    assert results["s2"].content == INTERRUPTED_RESULT, "没轮到的调用也补上，不能留没配对的"

    assert agent.resume() == "答案是 42"
    assert tool.runs == 1, "中断的调用不重跑：副作用可能已经发生"
    assert_paired(agent.context.render())


def test_带一句话接着跑_不算新回合():
    agent, _ = make_agent([call(1), RuntimeError("断"), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    agent.resume("别按月拆了，直接算全年")
    sent = agent.llm.seen[-1]
    assert sent[-1].content == "别按月拆了，直接算全年" and sent[-1].meta.synthetic
    assert len(turn_starts(agent.context.history)) == 1


def test_历史停在assistant上_补一句继续():
    # 模型没调工具、钩子说还没完，还没来得及补推动消息就断了
    agent, _ = make_agent([FINAL])
    agent.interrupted = InterruptedTurn((Message.user("算一下"), Message.assistant("我先想想")), steps=1)
    agent.resume()
    assert agent.llm.seen[-1][-1].content == CONTINUE_NUDGE


def test_断在收尾那次请求上_收尾提示不留两份():
    agent, events = make_agent([call(1), KeyboardInterrupt(), FINAL], max_steps=1)
    with pytest.raises(KeyboardInterrupt):
        agent.run("算一下")
    assert agent.interrupted.steps == 1
    assert agent.interrupted.entries[-1].role == "tool", "收尾提示没留在进度里"
    assert agent.resume() == "答案是 42"
    prompts = [m for m in agent.context.history if getattr(m, "content", "") == WRAP_UP.format(n=1)]
    assert len(prompts) == 1


# ================================================================ 作废
def test_问新问题或reset_进度作废():
    agent, _ = make_agent([call(1), RuntimeError("断"), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    agent.run("换个问题")
    assert agent.interrupted is None
    with pytest.raises(ValueError):
        agent.resume()

    agent, _ = make_agent([RuntimeError("断")])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    agent.reset()
    assert agent.interrupted is None


# ================================================================ 原地打转、用量锚点
def test_原地打转的计数接着算():
    agent, events = make_agent([call(1, "x"), call(2, "x"), RuntimeError("断"), call(3, "x"), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    agent.resume()
    third = [e for e in events if isinstance(e, ToolFinished)][2]
    assert third.content.endswith(REPEAT_NOTE.format(name="echo", n=3))


def test_接回来的消息保留原来的用量锚点():
    agent, _ = make_agent([call(1), call(2), RuntimeError("断"), FINAL])
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    stamps = [e.meta.measured_on for e in agent.interrupted.entries if getattr(e, "role", "") == "assistant"]
    agent.resume()
    after = [m.meta.measured_on for m in agent.context.history if m.role == "assistant"][:2]
    assert after == stamps and all(stamps)


# ================================================================ 存盘钩子
def test_每走一步交一次进度_断的时候再交一次():
    saved: list[InterruptedTurn] = []
    agent, _ = make_agent([call(1), call(2), RuntimeError("断")], checkpoint_hook=saved.append)
    with pytest.raises(RuntimeError):
        agent.run("算一下")
    assert [t.steps for t in saved] == [0, 1, 2, 2]
    assert saved[-1].reason == "RuntimeError: 断" and saved[-2].reason == ""


def test_存盘失败不影响这一轮():
    def broken(_turn):
        raise OSError("磁盘满了")

    agent, _ = make_agent([call(1), FINAL], checkpoint_hook=broken)
    assert agent.run("算一下") == "答案是 42"

    agent, _ = make_agent([RuntimeError("429")], checkpoint_hook=broken)
    with pytest.raises(RuntimeError, match="429"):       # 原来的异常没被盖掉
        agent.run("算一下")
