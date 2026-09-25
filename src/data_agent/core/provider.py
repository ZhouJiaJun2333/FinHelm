"""LLMProvider：模型接口。定义在 core，实现在 llm/（依赖只有 llm → core 一个方向）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .messages import LLMResponse, Message


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
