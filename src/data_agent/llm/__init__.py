"""llm —— 各家模型的 LLMProvider 实现（接口在 core/provider.py）。换厂商只动这个包。"""

__all__ = ["AnthropicProvider", "OpenAICompatibleProvider"]


def __getattr__(name: str):
    # 延迟导入：只装了 anthropic 没装 openai（或反之）也能正常跑
    if name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider
    if name == "OpenAICompatibleProvider":
        from .openai_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider
    raise AttributeError(name)
