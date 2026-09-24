"""配置：从环境变量 / .env 读取。别把 key 写进代码。"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm.base import LLMProvider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------- 模型 ----------------
    provider: Literal["anthropic", "openai"] = "anthropic"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    anthropic_context_window: int = 1_000_000

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "deepseek-flash"
    # 默认值是 DeepSeek 官方给的 1M。换别家要按官方文档改 ——
    # 不确定就往小了填：填小了只是早点压缩，填大了会撞上限直接报错。
    openai_context_window: int = 1_000_000

    # ---------------- 数据库 ----------------
    # 默认连 docker/docker-compose.yml 起的那个，用只读账号
    database_url: str = (
        "postgresql://agent_ro:agent_ro_pwd@localhost:5433/analytics"
    )
    db_schema: str = "shop"
    db_statement_timeout_ms: int = 30_000

    # ---------------- Agent ----------------
    max_steps: int = 12
    max_tokens: int = 8192

    # ---------------- 上下文管理 ----------------
    # 请求估算超过这个数，就把较早的工具结果换成占位。
    #
    # 为什么窗口有 1M 还要在 10 万就动手：压缩不只是为了不撞窗口。
    # 每次请求都把整段历史发一遍，历史越长每一步越贵；上下文里旧数据越多，
    # 模型越容易被干扰。Anthropic 同款功能的默认值也是 10 万。
    # 想亲眼看到清理，可以在 .env 里临时调成几千。
    context_clear_trigger_tokens: int = 100_000
    context_keep_tool_results: int = 3
    context_clear_at_least: int = 10_000
    # 给模型输出留的余量。窗口小的模型，触发线会被压到「窗口 - 这个数」以下。
    # 安全余量放在这里，而不是加在估算系数上 —— 估算只管尽量准。
    context_reserve_tokens: int = 16_384


def build_provider(settings: Settings) -> LLMProvider:
    """按配置造出对应的 provider。加新厂商就在这里加一个分支。"""
    if settings.provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=max(settings.max_tokens, 16000),
            context_window=settings.anthropic_context_window,
        )

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("PROVIDER=openai 但没有配置 OPENAI_API_KEY")
        from .llm.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_tokens,
            context_window=settings.openai_context_window,
        )

    raise ValueError(f"未知的 provider：{settings.provider}")
