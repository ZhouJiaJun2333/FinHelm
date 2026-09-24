"""LLMProvider：模型调用的抽象接口。

core/agent.py 只调用 chat()，拿到的永远是 LLMResponse。
换模型厂商 = 换一个 Provider 实例，主循环、工具、上下文管理全都不用动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.messages import LLMResponse, Message


class LLMProvider(ABC):
    """所有模型后端的统一接口。"""

    model: str

    # 这个模型的上下文窗口（token）。API 响应里不会告诉你，只能配置。
    # None = 不知道，界面上就只显示用了多少、不显示百分比。
    context_window: int | None = None

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """发一轮对话。

        Args:
            messages: 中立格式的消息历史（core/messages.py）
            tools:    中立格式的工具描述 {name, description, parameters}
            system:   系统提示词

        Returns:
            归一化后的 LLMResponse
        """
