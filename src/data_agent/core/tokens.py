"""上下文用量：锚点 + 增量估算（学 pi 的 estimateContextTokens）。

    总量 = 最后一条带 usage 的 assistant（锚点）的 context_tokens   ← API 报的，精确
         + 锚点之后每条消息的估算                                   ← 只有这一小段是估的
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .messages import Message


# ------------------------------------------------------------------ 粗估
# 每类字符折合多少 token。用 10 类典型内容在 deepseek-flash 上拟合，误差都在 ±13% 以内
# （chars/4 对 SQL 结果表格偏低 57%：数字和 | 远比字母费 token）。2026-09-24
# 这里只求准，安全余量放在压缩阈值上。
_TOKENS_PER_CHAR = {"letter": 0.25, "digit": 0.6, "punct": 0.9, "space": 0.0, "other": 0.6}


def estimate_text(text: str) -> int:
    """按字符类型粗估 token 数。"""
    w = _TOKENS_PER_CHAR
    total = 0.0
    for ch in text:
        if ord(ch) >= 128:
            total += w["other"]
        elif ch.isdigit():
            total += w["digit"]
        elif ch.isalpha():
            total += w["letter"]
        elif ch.isspace():
            total += w["space"]
        else:
            total += w["punct"]
    return round(total)


def estimate_message(message: Message) -> int:
    """估一条消息。不看 raw：要估的是锚点之后的消息，带 raw 的 assistant 自己就有 usage。"""
    tokens = estimate_text(message.content)
    for call in message.tool_calls:
        tokens += estimate_text(call.name)
        tokens += estimate_text(json.dumps(call.arguments, ensure_ascii=False))
    return tokens


def estimate_overhead(system: str, tools: list[dict[str, Any]]) -> int:
    """系统提示词 + 工具定义的估算。没有锚点时要加上，否则明显低估。"""
    return estimate_text(system) + estimate_text(json.dumps(tools, ensure_ascii=False))


# ------------------------------------------------------------ 锚点 + 增量
@dataclass(frozen=True, slots=True)
class ContextEstimate:
    tokens: int                  # 估算的总量 = usage_tokens + trailing_tokens
    usage_tokens: int            # 精确部分（锚点报的），没锚点时为 0
    trailing_tokens: int         # 估算部分
    anchor_index: int | None     # 锚点在消息列表里的下标，None = 全靠估

    @property
    def exact_ratio(self) -> float:
        """精确部分的占比。越接近 1，这个数越可信。"""
        return self.usage_tokens / self.tokens if self.tokens else 1.0


def estimate_context(messages: list[Message], *, overhead: int = 0) -> ContextEstimate:
    """估算把 messages（传 render() 的结果）发出去时输入有多大。overhead 只在没有锚点时加上。"""
    anchor = _last_anchor(messages)

    if anchor is None:
        trailing = overhead + sum(estimate_message(m) for m in messages)
        return ContextEstimate(trailing, 0, trailing, None)

    usage_tokens = messages[anchor].meta.usage.context_tokens
    trailing = sum(estimate_message(m) for m in messages[anchor + 1:])
    return ContextEstimate(usage_tokens + trailing, usage_tokens, trailing, anchor)


def _last_anchor(messages: list[Message]) -> int | None:
    """最后一条带 usage 的 assistant。截断、被拒的回复进不了历史，所以这里的 usage 都可信。"""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "assistant" and m.meta.usage is not None:
            return i
    return None
