"""LLMProvider：模型调用的抽象接口。

core/agent.py 只调用 chat()，拿到的永远是 LLMResponse。
换模型厂商 = 换一个 Provider 实例，主循环、工具、上下文管理全都不用动。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ..core.messages import LLMResponse, Message


class LLMProvider(ABC):
    """所有模型后端的统一接口。"""

    model: str

    # 这个模型的上下文窗口（token）。API 响应里不会告诉你，只能配置。
    # None = 不知道，界面上就只显示用了多少、不显示百分比。
    context_window: int | None = None

    # 模型自己会不会先思考再回答（DeepSeek Flash、开了 thinking 的 Claude…）。
    # 写摘要时用：不会思考的模型，要在提示词里让它先打个草稿。
    native_thinking: bool = False

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发一轮对话。

        Args:
            messages:   中立格式的消息历史（core/messages.py）
            tools:      中立格式的工具描述 {name, description, parameters}
            system:     系统提示词
            max_tokens: 这一次的输出上限；不传就用构造时配的。写摘要时会单独调高。

        Raises:
            ContextOverflow: 请求超出上下文窗口。各家报错写法不同，
                             每个 provider 负责翻译成这一个异常。

        Returns:
            归一化后的 LLMResponse
        """


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
