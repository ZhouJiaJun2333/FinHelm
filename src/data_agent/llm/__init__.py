"""llm —— 模型抽象层。换厂商只动这个包。"""

from .base import LLMProvider

__all__ = ["LLMProvider", "AnthropicProvider", "OpenAICompatibleProvider"]


def __getattr__(name: str):
    # 延迟导入：只装了 anthropic 没装 openai（或反之）也能正常跑
    if name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider
    if name == "OpenAICompatibleProvider":
        from .openai_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider
    raise AttributeError(name)
