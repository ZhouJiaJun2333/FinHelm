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

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "deepseek-flash"

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


def build_provider(settings: Settings) -> LLMProvider:
    """按配置造出对应的 provider。加新厂商就在这里加一个分支。"""
    if settings.provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=max(settings.max_tokens, 16000),
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
        )

    raise ValueError(f"未知的 provider：{settings.provider}")
