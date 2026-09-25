"""组装层：唯一知道所有零件怎么拼起来的地方。CLI、测试、评测都从 build_application() 拿。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .core.agent import Agent, ApprovalHook, FinishTurnHook
from .core.context import ClearOldToolResults, CompactHistory, Context, llm_summarizer
from .core.events import Event, noop_sink
from .core.provider import LLMProvider
from .core.tools import ToolRegistry
from .db.connection import Database
from .db.introspection import SchemaInspector
from .domains import get_domain
from .prompts import build_system_prompt
from .settings import Settings, build_provider
from .tools.sql.describe_table import DescribeTableTool
from .tools.sql.export_csv import ExportCsvTool
from .tools.sql.list_tables import ListTablesTool
from .tools.sql.results import ResultStore
from .tools.sql.run_sql import RunSqlTool


@dataclass(slots=True)
class Application:
    """装配好的一整套东西。"""

    agent: Agent
    db: Database
    inspector: SchemaInspector
    tools: ToolRegistry
    llm: LLMProvider
    settings: Settings
    # 这次会话查出过的结果（r1、r2…）：界面展开 {{r3}}、/save 都从这里拿
    results: ResultStore
    export_dir: Path                  # CSV 写到哪（/save 和 export_csv 共用）


def build_application(
    settings: Settings | None = None,
    *,
    on_event: Callable[[Event], None] = noop_sink,
    approval_hook: ApprovalHook | None = None,
    finish_turn_hook: FinishTurnHook | None = None,
    llm: LLMProvider | None = None,
    results: ResultStore | None = None,
    export_dir: Path | None = None,
) -> Application:
    """把所有零件拼成一个能跑的 Agent。

    settings 不传就从 .env 读；llm 可以塞假的（测试）。results 由界面先建好传进来
    （打印事件的 sink 要用它展开 {{r3}}）；export_dir 不传用 settings.export_dir。
    """
    settings = settings or Settings()
    # 场景包：数据在哪个 schema、业务约定是什么。内核的其余部分不知道行业
    domain = get_domain(settings.domain)

    # --- 数据层 ---
    db = Database(
        settings.database_url,
        statement_timeout_ms=settings.db_statement_timeout_ms,
        search_path=domain.schema,
    )
    inspector = SchemaInspector(db, schemas=(domain.schema,))

    # --- 工具层 ---
    # run_sql 往里存、export_csv 按编号取、界面和评测读：只有这一份
    results = results if results is not None else ResultStore()
    export_dir = export_dir or Path(settings.export_dir)
    tools = ToolRegistry([
        ListTablesTool(inspector),
        DescribeTableTool(db, inspector, default_schema=domain.schema),
        RunSqlTool(db, results),
        ExportCsvTool(db, results, export_dir),
    ])

    # --- 模型层 ---
    llm = llm or build_provider(settings)

    # --- 上下文 ---
    # 触发线不超过「窗口 - 余量」：换成小窗口的模型时不能等到 10 万才动手
    def cap(trigger: int) -> int:
        if llm.context_window:
            return min(trigger, llm.context_window - settings.context_reserve_tokens)
        return trigger

    # 先清理（几乎无损），清理完还超标再压缩（有损）
    context = Context([
        ClearOldToolResults(
            trigger_tokens=cap(settings.context_clear_trigger_tokens),
            keep_recent=settings.context_keep_tool_results,
            clear_at_least=settings.context_clear_at_least,
            tools=[t.name for t in tools if t.rerunnable],
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
        system_prompt=build_system_prompt(domain),
        context=context,
        max_steps=settings.max_steps,
        approval_hook=approval_hook,
        finish_turn_hook=finish_turn_hook,
        on_event=on_event,
        session_context=inspector.overview,
    )

    return Application(
        agent=agent, db=db, inspector=inspector,
        tools=tools, llm=llm, settings=settings, results=results, export_dir=export_dir,
    )
