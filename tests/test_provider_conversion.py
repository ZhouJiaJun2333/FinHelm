"""两家 provider 的格式转换测试。不需要 key，纯函数测试。

重点守两件事：
1. 并行工具结果在 Anthropic 那边必须合并进同一条 user 消息
2. assistant 消息回传时必须优先用原生内容 —— 否则厂商私有字段会丢
   （reasoning_content / thinking blocks 都属于这一类）
"""

from __future__ import annotations

import json

from data_agent.core.messages import Message, ToolCall
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.openai_provider import OpenAICompatibleProvider

TOOLS = [{
    "name": "run_sql",
    "description": "执行 SQL",
    "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
}]


def history_with_parallel_calls() -> list[Message]:
    return [
        Message.user("分析一下"),
        Message(role="assistant", content="我查两个维度", tool_calls=[
            ToolCall("c1", "run_sql", {"sql": "SELECT 1"}),
            ToolCall("c2", "run_sql", {"sql": "SELECT 2"}),
        ]),
        Message.tool_result("c1", "结果1"),
        Message.tool_result("c2", "结果2"),
    ]


# ============================================================ Anthropic
def test_anthropic_并行工具结果合并进一条user消息():
    """拆成两条发，会让模型以后不敢再并行调用工具。"""
    out = AnthropicProvider.convert_messages(history_with_parallel_calls())
    tool_result_msgs = [
        m for m in out
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(tool_result_msgs) == 1
    assert len(tool_result_msgs[0]["content"]) == 2


def test_anthropic_system不进messages():
    out = AnthropicProvider.convert_messages([
        Message(role="system", content="我是系统提示"),
        Message.user("你好"),
    ])
    assert all(m["role"] != "system" for m in out)
    assert len(out) == 1


def test_anthropic_工具schema字段名是input_schema():
    assert AnthropicProvider.convert_tools(TOOLS)[0]["input_schema"] == TOOLS[0]["parameters"]


def test_anthropic_优先用原生content():
    """thinking 块等信息只存在于原生 content 里，自己拼会丢。"""
    raw = [{"type": "thinking", "thinking": "让我想想"},
           {"type": "text", "text": "好的"}]
    out = AnthropicProvider.convert_messages([
        Message(role="assistant", content="好的", raw=raw),
    ])
    assert out[0]["content"] == raw


# ============================================================ OpenAI 兼容
def test_openai_工具结果是独立的tool消息():
    out = OpenAICompatibleProvider.convert_messages(history_with_parallel_calls(), None)
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


def test_openai_system是messages第一条():
    out = OpenAICompatibleProvider.convert_messages([Message.user("你好")], "我是系统提示")
    assert out[0] == {"role": "system", "content": "我是系统提示"}


def test_openai_工具参数序列化成json字符串():
    out = OpenAICompatibleProvider.convert_messages(history_with_parallel_calls(), None)
    assistant = next(m for m in out if m["role"] == "assistant")
    args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)                 # 是字符串不是 dict
    assert json.loads(args) == {"sql": "SELECT 1"}


def test_openai_优先用原生message保住reasoning_content():
    """回归测试：思考模型要求 reasoning_content 原样回传，否则 400。

    这里曾经有个 bug —— provider 把 raw_content 设成 None，
    回传时从中立结构重新拼，reasoning_content 就丢了。
    """
    raw = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "用户想查华东的销售额，我需要调用 get_sales",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "run_sql", "arguments": '{"sql": "SELECT 1"}'},
        }],
    }
    out = OpenAICompatibleProvider.convert_messages([
        Message(role="assistant", content="", raw=raw,
                tool_calls=[ToolCall("c1", "run_sql", {"sql": "SELECT 1"})]),
    ], None)

    assert "reasoning_content" in out[0]
    assert out[0]["reasoning_content"] == raw["reasoning_content"]


def test_openai_没有原生内容时回退到手工拼装():
    """假 Provider、测试替身不会有 raw，必须还能正常工作。"""
    out = OpenAICompatibleProvider.convert_messages([
        Message(role="assistant", content="纯文本回答"),
    ], None)
    assert out[0] == {"role": "assistant", "content": "纯文本回答"}


def test_openai_原生内容被复制而不是共享引用():
    """改了转换结果不该污染历史里的 raw。"""
    raw = {"role": "assistant", "content": "x", "reasoning_content": "思考"}
    msg = Message(role="assistant", content="x", raw=raw)
    out = OpenAICompatibleProvider.convert_messages([msg], None)
    out[0]["content"] = "被改了"
    assert raw["content"] == "x"


def test_openai_工具schema包在function里():
    converted = OpenAICompatibleProvider.convert_tools(TOOLS)[0]
    assert converted["type"] == "function"
    assert converted["function"]["parameters"] == TOOLS[0]["parameters"]


# ============================================ Anthropic 的角色交替约束
def _roles(converted: list[dict]) -> list[str]:
    return [m["role"] for m in converted]


def test_anthropic_连续的user消息是非法的():
    """守住这类错误本身。

    Anthropic 要求 user / assistant 交替。OpenAI 兼容接口（DeepSeek 等）
    对此宽容，所以「连续两条 user」在本地怎么测都不出来，换厂商才炸。
    这个测试就是那个在本地也会红的哨兵。
    """
    bad = [Message.user("第一个问题"), Message.user("第二个问题")]
    roles = _roles(AnthropicProvider.convert_messages(bad))
    consecutive = [a for a, b in zip(roles, roles[1:]) if a == b]
    assert consecutive, "这个测试本身失效了：它应该能造出非法输入"


def test_agent产生的历史在anthropic格式下角色永远交替():
    """把 Agent 可能产生的几种历史形状都过一遍。"""
    shapes = {
        "纯问答": [
            Message.user("你好"),
            Message(role="assistant", content="你好"),
        ],
        "带工具调用": history_with_parallel_calls() + [
            Message(role="assistant", content="分析完了"),
        ],
        "多轮": [
            Message.user("问题一"),
            Message(role="assistant", content="答案一"),
            Message.user("问题二"),
            Message(role="assistant", content="答案二"),
        ],
        "finish_turn 推动继续": [
            Message.user("问题"),
            Message(role="assistant", content="第一版答案"),
            Message.user("还缺占比，补上"),          # nudge
            Message(role="assistant", content="补充后的答案"),
        ],
    }
    for name, history in shapes.items():
        roles = _roles(AnthropicProvider.convert_messages(history))
        pairs = [(a, b) for a, b in zip(roles, roles[1:]) if a == b]
        assert not pairs, f"{name}: 角色没交替 {roles}"


def test_anthropic_每个tool_use都有对应的tool_result():
    """另一个入口的同款死亡模式：悬空的 tool_use 让之后每轮都 400。"""
    converted = AnthropicProvider.convert_messages(history_with_parallel_calls())

    use_ids, result_ids = set(), set()
    for m in converted:
        if not isinstance(m["content"], list):
            continue
        for block in m["content"]:
            if block.get("type") == "tool_use":
                use_ids.add(block["id"])
            elif block.get("type") == "tool_result":
                result_ids.add(block["tool_use_id"])

    assert use_ids == result_ids, f"悬空的调用: {use_ids ^ result_ids}"
