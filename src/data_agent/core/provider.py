"""LLMProvider：模型接口。定义在 core，实现在 llm/（依赖只有 llm → core 一个方向）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .messages import LLMResponse, Message

# 连不上、429、5xx 时 SDK 自己重试的次数。退避 0.5s 起翻倍、封顶 8s，5 次大约能扛过 15 秒的断网。
MAX_RETRIES = 5


class LLMProvider(ABC):
    """换厂商 = 换一个实现。"""

    model: str

    # 上下文窗口（token），只能配置。None = 不知道
    context_window: int | None = None

    # 模型会不会自己先思考。不会的话，写摘要时让它先打草稿
    native_thinking: bool = False

    # 能不能看图。不能的话不注册 view_image，历史里已有的图片发送时换成一句说明
    #（有的模型收到图片不报错，只在回答里说看不了，运行时发现不了，只能配置）
    vision: bool = False

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发一轮对话。max_tokens 不传就用构造时配的（写摘要时会单独调高）。

        请求超出上下文窗口时抛 ContextOverflow（各家报错不同，见 llm/overflow.py）。
        """

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        *,
        on_delta: OnDelta,
    ) -> LLMResponse:
        """和 chat 一样，只是边生成边把文字交给 on_delta(文字, 是不是思考)。返回值和 chat 的一模一样。

        不支持流式的实现不用管：默认整段调 chat，回答一次给完。
        """
        extra = {"max_tokens": max_tokens} if max_tokens is not None else {}
        response = self.chat(messages=messages, tools=tools, system=system, **extra)
        if response.text:
            on_delta(response.text, False)
        return response


# on_delta(文字, 是不是思考过程)
OnDelta = Callable[[str, bool], None]
