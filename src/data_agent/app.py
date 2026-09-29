"""组装层：唯一知道所有零件怎么拼起来的地方。CLI、测试、评测都从 build_application() 拿。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .core.agent import WRAP_UP, Agent, ApprovalHook, FinishTurnHook, InterruptedTurn
from .core.context import ClearOldToolResults, CompactHistory, Context, llm_summarizer
from .core.events import Event, noop_sink
from .core.messages import Message
from .core.provider import LLMProvider
from .core.tools import ToolRegistry
from .db.connection import Database
from .db.introspection import SchemaInspector
from .domains import get_domain
from .prompts import WRAP_UP_BEST_GUESS, build_system_prompt
from .settings import Settings, build_provider
from .tools.python import PYTHON_KERNEL, RunPythonTool
from .tools.r import R_KERNEL, RunRTool
from .tools.sandbox import Sandbox
from .tools.sql.describe_table import DescribeTableTool
from .tools.sql.export_csv import ExportCsvTool
from .tools.sql.list_tables import ListTablesTool
from .tools.sql.results import ResultStore, result_resolver, result_saver
from .tools.sql.run_sql import RunSqlTool
from .tools.paths import SandboxPaths
from .tools.read_file import ReadFileTool
from .tools.view_image import ViewImageTool


@dataclass(slots=True)
class Application:
    """装配好的一整套东西。"""

    agent: Agent
    db: Database | None               # 场景包不连数据库时是 None
    inspector: SchemaInspector | None
    tools: ToolRegistry
    llm: LLMProvider
    settings: Settings
    # 这次会话查出过的结果（r1、r2…）：界面展开 {{r3}}、/save 都从这里拿
    results: ResultStore
    export_dir: Path                  # CSV 写到哪（/save 和 export_csv 共用）
    work_dir: Path                    # 沙箱的工作目录：inputs/ 放上传的文件，figures/ 放图
    sandboxes: dict[str, Sandbox] = field(default_factory=dict)   # "python" / "r"
    pending_uploads: list[Path] = field(default_factory=list)     # 上传了、还没告诉模型的文件

    def attach(self, paths: list[Path]) -> list[Path]:
        """把用户的文件复制进 work_dir/inputs/（原件不给沙箱碰），返回复制后的路径。"""
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"找不到文件：{'、'.join(map(str, missing))}")
        inputs = self.work_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        copied = [Path(shutil.copy2(p, inputs / p.name)) for p in paths]
        self.pending_uploads += copied
        return copied

    def with_uploads(self, text: str) -> str:
        """下一条用户消息前面带上刚上传的文件：记进历史，--resume 之后模型也知道有这些文件。"""
        if not self.pending_uploads:
            return text
        files = "、".join(f"{p.name}（{_size(p)}）" for p in self.pending_uploads)
        self.pending_uploads = []
        return f"[用户上传了文件，在 inputs/ 下：{files}]\n\n{text}"

    def fresh_kernel_note(self, turn: InterruptedTurn) -> str:
        """程序重启后接着跑：这一轮在沙箱里建的变量都没了，要告诉模型，否则它会直接用。"""
        kernels = {tool.name for tool in self.tools if isinstance(tool, (RunPythonTool, RunRTool))}
        used = sorted(kernels & {c.name for e in turn.entries if isinstance(e, Message) for c in e.tool_calls})
        if not used:
            return ""
        return f"[程序重启过，{'、'.join(used)} 的内核是新的：之前定义的变量都没了，要用就重新读取或计算。]"

    def reset(self) -> None:
        """清空对话，内核也换个空的：新对话不该看到上一段留下的变量。"""
        self.agent.reset()
        self.close()

    def close(self) -> None:
        for sandbox in self.sandboxes.values():
            sandbox.close()


def _size(path: Path) -> str:
    n = path.stat().st_size
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{max(1, round(n / 1024))} KB"


def build_application(
    settings: Settings | None = None,
    *,
    on_event: Callable[[Event], None] = noop_sink,
    approval_hook: ApprovalHook | None = None,
    finish_turn_hook: FinishTurnHook | None = None,
    llm: LLMProvider | None = None,
    results: ResultStore | None = None,
    export_dir: Path | None = None,
    work_dir: Path | None = None,
) -> Application:
    """把所有零件拼成一个能跑的 Agent。

    settings 不传就从 .env 读；llm 可以塞假的（测试）。results 由界面先建好传进来
    （打印事件的 sink 要用它展开 {{r3}}）；export_dir、work_dir 不传用 settings 里的。
    用完要 close()：沙箱是个容器。
    """
    settings = settings or Settings()
    # 场景包：数据在哪个 schema、业务约定是什么。内核的其余部分不知道行业
    domain = get_domain(settings.domain)

    # --- 数据层 ---
    db = inspector = None
    if domain.schema is not None and "sql" in domain.tools:
        db = Database(
            settings.database_url,
            statement_timeout_ms=settings.db_statement_timeout_ms,
            search_path=domain.schema,
        )
        inspector = SchemaInspector(db, schemas=(domain.schema,))

    # --- 模型层 --- 放在工具前面：注册哪些工具要看模型能力（能不能看图）
    llm = llm or build_provider(settings)

    # --- 工具层 ---
    # run_sql 往里存、export_csv 按编号取、界面、评测和沙箱读：只有这一份
    results = results if results is not None else ResultStore()
    export_dir = export_dir or Path(settings.export_dir)
    work_dir = work_dir or Path(settings.work_dir)
    tools = ToolRegistry()
    if db is not None:
        for tool in (ListTablesTool(inspector), DescribeTableTool(db, inspector, default_schema=domain.schema),
                     RunSqlTool(db, results), ExportCsvTool(db, results, export_dir)):
            tools.register(tool)
    # 沙箱：场景包要、.env 里也开着才有。第一次调用才启动容器；两个容器挂同一个工作目录
    sandboxes: dict[str, Sandbox] = {}
    data_dir = Path(domain.data_dir) if domain.data_dir else None
    if data_dir is not None and not data_dir.is_dir():
        raise FileNotFoundError(f"场景包 {domain.name} 的数据目录 {data_dir} 不存在（在项目根目录下运行？数据下载了吗？）")
    for kind, enabled, image, kernel, tool_class in (
        ("python", settings.python_sandbox, settings.sandbox_image, PYTHON_KERNEL, RunPythonTool),
        ("r", settings.r_sandbox, settings.sandbox_r_image, R_KERNEL, RunRTool),
    ):
        if kind in domain.tools and enabled:
            sandboxes[kind] = Sandbox.docker(
                image, kernel, work_dir, result_resolver(results), timeout_s=settings.sandbox_timeout_s,
                memory=settings.sandbox_memory, cpus=settings.sandbox_cpus, save=result_saver(results, kind),
                data_dir=data_dir,
            )
            tools.register(tool_class(sandboxes[kind]))
    # 读文档、看图：有沙箱才有文件可读。看图要模型能看：不注册的话，提示词里「交付前看一眼」那句也就没了
    if sandboxes:
        paths = SandboxPaths(work_dir, data_dir)
        tools.register(ReadFileTool(paths))
        if llm.vision:
            tools.register(ViewImageTool(paths))

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
        system_prompt=build_system_prompt([t.name for t in tools], rules=f"- 数据是{domain.subject}。\n{domain.rules}",
                                          data_dir=data_dir is not None),
        context=context,
        max_steps=settings.max_steps,
        approval_hook=approval_hook,
        finish_turn_hook=finish_turn_hook,
        on_event=on_event,
        session_context=inspector.overview if inspector else None,
        wrap_up_prompt=WRAP_UP_BEST_GUESS if settings.wrap_up == "best_guess" else WRAP_UP,
    )

    return Application(
        agent=agent, db=db, inspector=inspector,
        tools=tools, llm=llm, settings=settings, results=results, export_dir=export_dir,
        work_dir=work_dir, sandboxes=sandboxes,
    )
