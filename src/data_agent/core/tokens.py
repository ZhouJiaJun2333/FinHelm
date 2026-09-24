"""上下文用量：下一次请求大概有多大？

这是上下文管理的地基 —— 不知道现在用了多少，就没法决定什么时候该压缩。

做法抄的是 pi（packages/agent/src/harness/compaction/compaction.ts 的
estimateContextTokens）：**锚点 + 增量估算**。

    历史：  user  asst  tool  tool  asst★  tool  user
                                    ↑
                          最后一条带 usage 的 assistant（锚点）

    总量 = 锚点的 usage.context_tokens      ← API 报的，精确
         + 锚点之后每条消息的估算值          ← 只有这一小段是估的

为什么不全靠估：估算误差大（中文尤其），而且系统提示词、工具定义、消息模板
的开销很难自己算准。usage 把这些全算进去了。
为什么不全靠 usage：它是**事后**的数，说的是上一次请求。之后新追加的工具结果、
用户提问，只能估。
为什么不每次调计数接口：多一次往返，而且不是每家都有。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .messages import Message


# ------------------------------------------------------------------ 粗估
# 每类字符大约折合多少 token。
#
# pi 用的是一刀切的 chars / 4，DeepSeek 官方给的是「英文字符 0.3、汉字 0.6」。
# 两者都是按**自然语言**说的，而锚点之后要估的主要是工具结果 —— SQL 结果表格
# 满是数字和 `|`，这两类远比字母费 token：
#
#     字母   0.25   英文单词整个切成一个 token，平均每字母很便宜
#     数字   0.6    分词器把长数字切成很多段，8100531.47 要好几个 token
#     标点   0.9    | - ( ) , 之类基本一个一个切
#     空白   0      大多并进相邻的 token 里（" quick" 是一个 token）
#     非 ASCII 0.6  汉字、全角标点；和 DeepSeek 官方给的汉字比例一致
#
# 这组数是拿 10 类典型内容在 deepseek-flash 上实测拟合的（2026-09-24）：
#
#                     实报    这组系数        chars/4        DeepSeek 官方比例
#     SQL 结果表格     867    753 (-13%)     376 (-57%)     479 (-45%)
#     表结构描述       313    276 (-12%)     235 (-25%)     310  (-1%)
#     系统提示词       391    357  (-9%)     188 (-52%)     346 (-12%)
#     纯中文           232    240  (+3%)     100 (-57%)     240  (+3%)
#     纯英文           321    348  (+8%)     410 (+28%)     492 (+53%)
#     纯数字           881    933  (+6%)     407 (-54%)     489 (-44%)
#     全部 10 类误差都在 ±13% 以内；原先的「ASCII/4 + 其余 1」最差偏 72%。
#
# 注意两点：
#   · 这是**估**，不追求精确。安全余量不在这里加，而是在「什么时候压缩」
#     的阈值上留（和 pi 的 reserveTokens 一个思路）—— 估算只管尽量准。
#   · 不同厂商分词器不同。Claude 没有公开分词器，要校准得用 count_tokens 接口。
#     反正只有锚点之后那一小段是估的，系数差一点影响不大。
_TOKENS_PER_CHAR = {"letter": 0.25, "digit": 0.6, "punct": 0.9, "space": 0.0, "other": 0.6}


def estimate_text(text: str) -> int:
    """按字符类型粗估 token 数。系数的来历见上面的 _TOKENS_PER_CHAR。"""
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
    """估一条消息。只看中立结构里的字段，不看 raw。

    不看 raw 是故意的：需要估算的只是锚点**之后**的消息，那里基本都是工具结果
    和用户提问（raw 为空）。真正带 raw 的 assistant 消息自己就带着 usage。
    """
    tokens = estimate_text(message.content)
    for call in message.tool_calls:
        tokens += estimate_text(call.name)
        tokens += estimate_text(json.dumps(call.arguments, ensure_ascii=False))
    return tokens


def estimate_overhead(system: str, tools: list[dict[str, Any]]) -> int:
    """系统提示词 + 工具定义。每次请求都要发，是固定开销。

    有锚点时用不上它（锚点的 usage 已经包含了）；没锚点时不加上它会明显低估 ——
    这个项目三个工具的定义加系统提示词就有好几千 token。
    """
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
    """估算把 messages 发出去时，这次请求的输入有多大。

    Args:
        messages: 要发给模型的消息 —— 传 context.render() 的结果，不是内部历史。
                  将来压缩发生在 render() 里，估的必须是真正发出去的那份。
        overhead: 系统提示词 + 工具定义的估算值，只在找不到锚点时加上。
    """
    anchor = _last_anchor(messages)

    if anchor is None:
        trailing = overhead + sum(estimate_message(m) for m in messages)
        return ContextEstimate(trailing, 0, trailing, None)

    usage_tokens = messages[anchor].meta.usage.context_tokens
    trailing = sum(estimate_message(m) for m in messages[anchor + 1:])
    return ContextEstimate(usage_tokens + trailing, usage_tokens, trailing, anchor)


def _last_anchor(messages: list[Message]) -> int | None:
    """从后往前找最后一条带 usage 的 assistant 消息。

    pi 在这里还要跳过 stopReason 为 aborted / error 的消息，因为它会把失败的回复
    也存进历史。我们不需要：被截断、被拒绝的回复根本进不了历史（agent.py 的
    _check_stop_reason），失败的一轮整轮回滚。能在历史里看到的 usage 都是好的。
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "assistant" and m.meta.usage is not None:
            return i
    return None
