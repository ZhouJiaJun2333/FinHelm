"""LLMProvider：Agent 需要的「模型」长什么样。

接口定义在 core 里，实现放在 llm/ 里（依赖倒置）：core 只说「我要一个能 chat() 的东西」，
llm/anthropic_provider.py、llm/openai_provider.py 负责实现它。依赖方向只有一条：

    llm/  ──►  core/          （实现依赖接口，接口不知道有哪些实现）

以前接口放在 llm/base.py，core/agent.py 要 import 它，llm 又要 import core.messages ——
两个包互相依赖，core/__init__.py 只好用惰性导入绕开循环。接口挪进来以后，环就没了。

pi 是另一种拆法：消息类型和模型接口都在最底层的 ai 包里，agent 包依赖它。我们的 Message
上挂着 MessageMeta（锚点、摘要线索……都是 Agent 自己的事），放不进模型层，所以选了这一种。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .messages import LLMResponse, Message


class LLMProvider(ABC):
    """所有模型后端的统一接口。换厂商 = 换一个实现，主循环、工具、上下文管理都不用动。"""

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
                             每个实现负责翻译成这一个异常（见 llm/overflow.py）。

        Returns:
            归一化后的 LLMResponse
        """
