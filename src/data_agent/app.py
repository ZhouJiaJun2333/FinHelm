"""组装层（composition root）—— 唯一一个知道「所有零件怎么拼起来」的地方。

为什么要单独一层？
    因为每个模块都只依赖抽象：agent 不知道有 Postgres，工具不知道有 DeepSeek。
    总得有个地方把具体实现塞进去 —— 就是这里，而且只有这里。

    好处：cli 要的是 Agent，测试要的也是 Agent，两边都从这里拿，
    差别只是传进来的 settings 不同。以后加 Web API，同样调 build_agent()。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .core.agent import Agent, ApprovalHook, FinishTurnHook
from .core.context import ClearOldToolResults, CompactHistory, Context, llm_summarizer
from .core.events import Event, noop_sink
from .db.connection import Database
from .db.introspection import SchemaInspector
from .llm.base import LLMProvider
from .prompts import SYSTEM_PROMPT
from .settings import Settings, build_provider
from .tools.registry import ToolRegistry
from .tools.sql.describe_table import DescribeTableTool
from .tools.sql.export_csv import ExportCsvTool
from .tools.sql.list_tables import ListTablesTool
from .tools.sql.run_sql import RunSqlTool


@dataclass(slots=True)
class Application:
    """装配好的一整套东西。CLI / 测试 / 以后的 Web 层都拿这个。"""

    agent: Agent
    db: Database
    inspector: SchemaInspector
    tools: ToolRegistry
    llm: LLMProvider
    settings: Settings


def build_application(
    settings: Settings | None = None,
    *,
    on_event: Callable[[Event], None] = noop_sink,
    approval_hook: ApprovalHook | None = None,
    finish_turn_hook: FinishTurnHook | None = None,
    llm: LLMProvider | None = None,
) -> Application:
    """把所有零件拼成一个能跑的 Agent。

    Args:
        settings:         不传就从 .env / 环境变量读
        on_event:         事件消费者（CLI 打终端、测试收集起来断言）
        approval_hook:    工具执行前的审批钩子
        finish_turn_hook: 每轮结束时决定「收工还是继续」
        llm:              显式指定 provider。测试时可以塞个假的，不打真实 API。
    """
    settings = settings or Settings()

    # --- 数据层 ---
    db = Database(
        settings.database_url,
        statement_timeout_ms=settings.db_statement_timeout_ms,
    )
    inspector = SchemaInspector(db, schemas=(settings.db_schema,))

    # --- 工具层：依赖在这里注入，工具内部不碰全局变量 ---
    run_sql = RunSqlTool(db)
    tools = ToolRegistry([
        ListTablesTool(inspector),
        DescribeTableTool(db, inspector, default_schema=settings.db_schema),
        run_sql,
        # 导出按编号重跑 run_sql 记下的 SQL —— 两个工具共用那张「编号 → SQL」表
        ExportCsvTool(db, run_sql.queries, Path(settings.export_dir)),
    ])

    # --- 模型层 ---
    llm = llm or build_provider(settings)

    # --- 上下文 ---
    # 触发线取「配置值」和「窗口 - 余量」里小的那个：配置写 10 万，
    # 但换成一个 32k 窗口的模型时，不能等到 10 万才动手。
    def cap(trigger: int) -> int:
        if llm.context_window:
            return min(trigger, llm.context_window - settings.context_reserve_tokens)
        return trigger

    # 编辑工序按顺序套用：先清理（几乎无损），清理完还超标再压缩（有损）。
    # 以后加去重之类，往这个列表里加就行。
    context = Context([
        ClearOldToolResults(
            trigger_tokens=cap(settings.context_clear_trigger_tokens),
            keep_recent=settings.context_keep_tool_results,
            clear_at_least=settings.context_clear_at_least,
        ),
        CompactHistory(
            summarize=llm_summarizer(llm, max_tokens=settings.context_compact_max_tokens),
            trigger_tokens=cap(settings.context_compact_trigger_tokens),
            keep_recent_tokens=settings.context_compact_keep_recent_tokens,
        ),
    ])

    # --- Agent ---
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context=context,
        max_steps=settings.max_steps,
        approval_hook=approval_hook,
        finish_turn_hook=finish_turn_hook,
        on_event=on_event,
        # 每轮把库概览拼到系统提示词末尾。
        # 以后接记忆 / RAG，也从这个钩子注入。
        dynamic_context=inspector.overview,
    )

    return Application(
        agent=agent, db=db, inspector=inspector,
        tools=tools, llm=llm, settings=settings,
    )
