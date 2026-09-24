"""用量计算的测试：厂商字段归一化 + 锚点/增量估算 + 和事务回滚的配合。

不需要 key：厂商那边的 usage 直接用 SDK 自己的类型构造，和真实响应同一个类。
"""

from __future__ import annotations

import anthropic.types
import pytest
from openai.types import CompletionUsage
from openai.types.completion_usage import PromptTokensDetails

from data_agent.core.errors import OutputTruncated
from data_agent.core.events import LLMResponded
from data_agent.core.messages import LLMResponse, Message, MessageMeta, ToolCall, Usage
from data_agent.core.tokens import (
    estimate_context,
    estimate_message,
    estimate_text,
)
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.openai_provider import OpenAICompatibleProvider

from fakes import make_agent


# ====================================================== 厂商字段 → Usage
def test_anthropic_input_tokens不含缓存_完整输入要三块相加():
    """开了缓存后 input_tokens 只剩零头，只看它会以为上下文还很空。"""
    u = AnthropicProvider.convert_usage(anthropic.types.Usage(
        input_tokens=120,
        output_tokens=300,
        cache_read_input_tokens=48_000,
        cache_creation_input_tokens=2_000,
    ))
    assert u == Usage(input=120, output=300, cache_read=48_000, cache_write=2_000)
    assert u.prompt_tokens == 50_120
    assert u.context_tokens == 50_420


def test_anthropic_没开缓存时缓存字段是None():
    u = AnthropicProvider.convert_usage(anthropic.types.Usage(
        input_tokens=500, output_tokens=20,
        cache_read_input_tokens=None, cache_creation_input_tokens=None,
    ))
    assert u == Usage(input=500, output=20)


def test_openai_prompt_tokens已含缓存_要拆出来():
    u = OpenAICompatibleProvider.convert_usage(CompletionUsage(
        prompt_tokens=10_000, completion_tokens=200, total_tokens=10_200,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=8_000),
    ))
    assert u == Usage(input=2_000, output=200, cache_read=8_000)
    # 拆完再加回来，总量不能变
    assert u.prompt_tokens == 10_000


def test_deepseek的缓存命中在顶层私有字段():
    """DeepSeek 用 prompt_cache_hit_tokens，SDK 把它当 extra 字段保留下来。"""
    u = OpenAICompatibleProvider.convert_usage(CompletionUsage(
        prompt_tokens=10_000, completion_tokens=200, total_tokens=10_200,
        prompt_cache_hit_tokens=6_400, prompt_cache_miss_tokens=3_600,
    ))
    assert u.cache_read == 6_400
    assert u.input == 3_600
    assert u.prompt_tokens == 10_000


def test_没返回usage的兼容接口不崩():
    assert OpenAICompatibleProvider.convert_usage(None) == Usage()


def test_全0的usage不能当锚点():
    msg = LLMResponse(text="hi", usage=Usage()).to_message()
    assert msg.meta.usage is None


def test_usage可以累加():
    total = Usage(input=1, output=2, cache_read=3, cache_write=4) + Usage(10, 20, 30, 40)
    assert total == Usage(input=11, output=22, cache_read=33, cache_write=44)


# ============================================================ 粗估
def test_不同字符类型按不同系数估():
    assert estimate_text("a" * 400) == 100      # 字母便宜
    assert estimate_text("中" * 400) == 240     # 汉字 0.6，和 DeepSeek 官方一致
    assert estimate_text(" " * 400) == 0        # 空白并进相邻 token


def test_数字和表格符号比字母贵得多():
    """工具结果（SQL 表格）是锚点之后最主要的内容，偏偏是 chars/4 最不准的地方。
    实测一张 42 行的结果表，chars/4 低估了 57%。
    """
    word = "abcdefghij"
    number = "8100531.47"
    assert len(word) == len(number)
    assert estimate_text(number) > 2 * estimate_text(word)


def test_工具调用的参数也要算进去():
    bare = Message(role="assistant", content="查一下")
    with_call = Message(role="assistant", content="查一下", tool_calls=[
        ToolCall("c1", "run_sql", {"sql": "SELECT region, SUM(amount) FROM orders"}),
    ])
    assert estimate_message(with_call) > estimate_message(bare)


# ====================================================== 锚点 + 增量估算
def _asst(text: str, usage: Usage | None) -> Message:
    return Message(role="assistant", content=text, meta=MessageMeta(usage=usage))


def test_没有锚点时全靠估_而且要加上固定开销():
    msgs = [Message.user("各区域销售额是多少")]
    est = estimate_context(msgs, overhead=3_000)
    assert est.anchor_index is None
    assert est.usage_tokens == 0
    assert est.tokens == 3_000 + estimate_message(msgs[0])


def test_有锚点时_锚点之前用精确值_之后才估():
    tool_output = "| region | gmv |\n| 华东 | 8100531.47 |"
    msgs = [
        Message.user("问题"),
        _asst("查一下", Usage(input=5_000, output=50)),
        Message.tool_result("c1", tool_output),
    ]
    est = estimate_context(msgs, overhead=3_000)
    assert est.anchor_index == 1
    assert est.usage_tokens == 5_050
    assert est.trailing_tokens == estimate_message(msgs[2])
    # 固定开销已经包含在锚点的 usage 里了，不能再加一遍
    assert est.tokens == 5_050 + estimate_message(msgs[2])


def test_锚点取最后一条带usage的assistant():
    msgs = [
        Message.user("q1"),
        _asst("a1", Usage(input=1_000, output=10)),
        Message.user("q2"),
        _asst("a2", Usage(input=2_000, output=20)),
        Message.user("q3"),
    ]
    assert estimate_context(msgs).anchor_index == 3


def test_不带usage的assistant会被跳过():
    """步数耗尽时补的收尾消息是我们自己写的，没有 usage，要往前找。"""
    msgs = [
        Message.user("q1"),
        _asst("a1", Usage(input=1_000, output=10)),
        Message.tool_result("c1", "结果"),
        Message.assistant("已达到最大步数"),     # 没有 usage
        Message.user("q2"),
    ]
    est = estimate_context(msgs)
    assert est.anchor_index == 1
    assert est.trailing_tokens == sum(estimate_message(m) for m in msgs[2:])


# ============================================================ 接进 Agent
U1 = Usage(input=1_000, output=40)
U2 = Usage(input=1_100, output=60)


def test_模型返回的assistant消息带着usage进历史():
    agent, _ = make_agent([
        LLMResponse(text="查一下", stop_reason="tool_use", usage=U1,
                    tool_calls=[ToolCall("c1", "echo", {"text": "a"})]),
        LLMResponse(text="答完了", stop_reason="end_turn", usage=U2),
    ])
    agent.run("问题")

    history = agent.context.render()
    assert [m.meta.usage for m in history if m.role == "assistant"] == [U1, U2]
    # 刚答完，最后一条就是锚点，没有需要估的部分
    est = agent.context_usage()
    assert est.anchor_index == len(history) - 1
    assert est.tokens == U2.context_tokens
    assert est.trailing_tokens == 0


def test_事件里带着用量和窗口大小():
    agent, events = make_agent([LLMResponse(text="好", stop_reason="end_turn", usage=U1)])
    agent.llm.context_window = 128_000
    agent.run("问题")
    responded = [e for e in events if isinstance(e, LLMResponded)]
    assert responded[0].usage == U1
    assert responded[0].context_window == 128_000


def test_回滚会带走锚点_但会话累计不回滚():
    """两个数对失败的态度正好相反：

    锚点描述的是「历史长什么样」，失败的一轮从历史里抹掉了，锚点也得跟着走。
    会话累计描述的是「花了多少钱」，被截断的那次调用照样收费。
    """
    truncated = Usage(input=1_200, output=8_192)
    agent, _ = make_agent([
        LLMResponse(text="第一轮", stop_reason="end_turn", usage=U1),
        LLMResponse(text="写到一半", stop_reason="max_tokens", usage=truncated),
    ])
    agent.run("问题一")
    with pytest.raises(OutputTruncated):
        agent.run("问题二")

    assert agent.context_usage().usage_tokens == U1.context_tokens
    assert agent.session_usage == U1 + truncated


def test_reset清空历史但不清会话累计():
    agent, _ = make_agent([LLMResponse(text="好", stop_reason="end_turn", usage=U1)])
    agent.run("问题")
    agent.reset()
    assert agent.context_usage().anchor_index is None
    assert agent.session_usage == U1
