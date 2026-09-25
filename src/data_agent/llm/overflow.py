"""认出「上下文超长」的报错。每个 provider 用它把自家的报错翻译成 core.errors.ContextOverflow。"""

from __future__ import annotations

import re

# 各家「上下文超长」的写法：
#     Anthropic   prompt is too long: … / input length and `max_tokens` exceed context limit
#     OpenAI 系   context_length_exceeded / maximum context length is …
#     阿里云百炼  Range of input length should be [1, N]（code 什么参数错都用，不能拿来认）
# 只认这些，别的 400 原样抛：当成超长会白白压缩一次。接新厂商时先撑爆一次看它怎么报。
_OVERFLOW_PATTERNS = re.compile(
    r"prompt is too long"
    r"|exceed(s|ed)? (the )?context (limit|window|length)"
    r"|context_length_exceeded"
    r"|maximum context length"
    r"|range of input length should be",
    re.IGNORECASE,
)


def is_context_overflow(error_message: str) -> bool:
    """这条 API 报错是不是「上下文超长」。"""
    return bool(_OVERFLOW_PATTERNS.search(error_message))
