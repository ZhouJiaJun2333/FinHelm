"""配置：从环境变量 / .env 读取。别把 key 写进代码。"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.provider import LLMProvider


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
    # 不确定就往小了填：填小了只是早点压缩，填大了会撞上限
    openai_context_window: int = 1_000_000
    # 模型会不会自己思考（deepseek-flash 会）。不会的要改成 false，写摘要时让它先打草稿
    openai_native_thinking: bool = True
    # 模型能不能看图（deepseek-flash 能，deepseek-v4-pro 不能）。不能的要改成 false：
    # v4-pro 收到图片不报错，只在回答里说 Unsupported Image。关掉就不注册 view_image
    openai_vision: bool = True

    # ---------------- 数据库 ----------------
    # docker/docker-compose.yml 起的库，只读账号
    database_url: str = (
        "postgresql://agent_ro:agent_ro_pwd@localhost:5433/analytics"
    )
    # 场景包（domains/）：shop = 自己造的电商库，financial = BIRD 的银行库，research = 医学科研（只用上传的文件）
    domain: str = "shop"
    db_statement_timeout_ms: int = 30_000
    # 每次对话一个子目录：日志、查询结果、导出的 CSV
    sessions_dir: str = "sessions"
    # 没有会话目录时（评测、测试）export_csv 写到这里
    export_dir: str = "outputs"

    # ---------------- Python 沙箱 ----------------
    # run_python 跑在 Docker 里（先 docker build -t finhelm-sandbox docker/sandbox）。没有 Docker 就关掉
    python_sandbox: bool = True
    sandbox_image: str = "finhelm-sandbox"
    # run_r（先 docker build -t finhelm-sandbox-r docker/sandbox-r）。场景包要 r 才注册，比如 research
    r_sandbox: bool = True
    sandbox_r_image: str = "finhelm-sandbox-r"
    sandbox_timeout_s: int = 60
    sandbox_memory: str = "2g"
    sandbox_cpus: float = 2
    # 没有会话目录时（评测、测试）沙箱的工作目录，图表存在它下面的 figures/
    work_dir: str = "outputs/work"

    # ---------------- Agent ----------------
    max_steps: int = 12
    # 步数用完时怎么收尾：report = 确定的照实说、没做完的说清楚做到哪一步；
    # best_guess = 没定下来的按最合理的假设给出答案（评测：交一个答案比弃权划算）
    wrap_up: Literal["report", "best_guess"] = "report"
    # 输出上限，思考 token 也算在内。8192 时列长清单会被截断；只是封顶，调高不多花钱
    max_tokens: int = 32768

    # ---------------- 上下文管理 ----------------
    # 清理工具结果（几乎无损，数据能重查）：超过它就把较早的结果换成占位。Anthropic 默认 10 万
    context_clear_trigger_tokens: int = 100_000
    context_keep_tool_results: int = 3
    context_clear_at_least: int = 10_000
    # 摘要压缩（有损）：清理之后还超过它才压。Anthropic 默认 15 万。窗口 1M 不代表要用满：
    # 上下文越长模型越容易走神（context rot），多带的历史每一步都要付钱
    context_compact_trigger_tokens: int = 150_000
    # 压缩时保留的最近原文（按回合取整，至少当前这一轮）。pi 的默认值
    context_compact_keep_recent_tokens: int = 20_000
    # 写摘要那一次的输出上限（deepseek-flash 实测 5000~7000）。Anthropic 不开流式超过约 21333 会报错
    context_compact_max_tokens: int = 16_000
    # 给输出留的余量：窗口小的模型，触发线压到「窗口 - 它」以下
    context_reserve_tokens: int = 16_384


def build_provider(settings: Settings) -> LLMProvider:
    """按配置造 provider。加新厂商就在这里加一个分支。"""
    if settings.provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            # Anthropic SDK 不开流式时 max_tokens 超过约 21333 直接报错
            max_tokens=min(max(settings.max_tokens, 16000), 21_000),
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
            native_thinking=settings.openai_native_thinking,
            vision=settings.openai_vision,
        )

    raise ValueError(f"未知的 provider：{settings.provider}")
