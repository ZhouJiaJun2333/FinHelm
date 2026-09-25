"""认出「上下文超长」的报错。每个 provider 用它把自家的报错翻译成 core.errors.ContextOverflow。"""

from __future__ import annotations

import re

# 「上下文超长」的报错，各家的写法：
#     Anthropic   prompt is too long: 210000 tokens > 200000 maximum
#                 input length and `max_tokens` exceed context limit: …
#     OpenAI      code=context_length_exceeded
#                 This model's maximum context length is 128000 tokens. However, …
#     DeepSeek 等兼容接口大多照抄 OpenAI 的那句
#     阿里云百炼  Range of input length should be [1, 983616]
#                 两个端点（OpenAI 兼容 / Anthropic 兼容）都是这句，code 分别是
#                 invalid_parameter_error / InvalidParameter —— 这两个 code 什么参数错都用，
#                 不能拿来认。2026-09-24 用 qwen3.6-flash、deepseek-v4.1-flash、glm-5.3 实测
# 只认这几种说法，别的 400 原样往外抛 —— 把别的错误当成超长，会白白压缩一次。
# ⚠️ 这张表是「见过才认得」：接新厂商时，发一个超大请求看看它怎么报，再补进来。
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
