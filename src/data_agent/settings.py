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
    # 请求估算超过这个数，就把较早的工具结果换成占位（留一句线索）。
    #
    # ⚠️ 这只是「清理工具结果」这一层的阈值，不是摘要压缩的阈值。两层该分开：
    #
    #   清理工具结果  几乎无损：数据能重查，占位里还留着线索 → 可以早点做
    #                 Anthropic context editing 的默认值就是 10 万
    #   摘要压缩      有损：细节概括掉就找不回来，压一次还要把全部历史读一遍
    #                 → 能晚就晚。Anthropic API 默认 15 万；Claude Code 在 1M
    #                 窗口上约 96.7 万才压，pi 是「窗口 - 16384」
    #
    # 窗口有 1M 还在 10 万就清，主要是为了**质量**：token 越多，模型的准确率和
    # 回忆能力越差（Anthropic 文档管这叫 context rot）。钱是次要的 —— 开了缓存，
    # 重发的历史按缓存价算（Anthropic 是一折），但一轮十几步、每步都付，照样会攒起来。
    #
    # 想亲眼看到清理，可以在 .env 里临时调成几千。
    context_clear_trigger_tokens: int = 100_000
    context_keep_tool_results: int = 3
    context_clear_at_least: int = 10_000
    # 摘要压缩：清理之后请求估算还超过这个数，就把较早的回合换成模型写的摘要。
    # 15 万是 Anthropic API 服务端压缩的默认值。窗口 1M 不代表要用满 ——
    # 我们的旧回合（查完的 SQL 结果）价值低，DeepSeek Flash 长上下文容易走神，
    # 早点压更划算。换成更强的模型可以调高。
    context_compact_trigger_tokens: int = 150_000
    # 压缩时保留多少最近的原文（按回合取整，至少保留当前这一轮）。照抄 pi 的默认值。
    context_compact_keep_recent_tokens: int = 20_000
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
