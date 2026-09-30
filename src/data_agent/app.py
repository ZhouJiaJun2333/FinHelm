"""组装层：唯一知道所有零件怎么拼起来的地方。CLI、测试、评测都从 build_application() 拿。"""

from __future__ import annotations

import shutil
from datetime import date
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .core.agent import WRAP_UP, Agent, ApprovalHook, FinishTurnHook, InterruptedTurn
from .core.messages import ToolCall
from .core.context import ClearOldToolResults, CompactHistory, Context, Entry, llm_summarizer
from .core.events import Event, noop_sink
from .core.messages import Message
from .core.provider import LLMProvider
from .core.tools import ToolRegistry
from .db.connection import Database
from .db.introspection import SchemaInspector
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
from .tools.ask_user import AskUserTool
from .tools.docs import ListDocsTool, ReadDocTool, SearchDocsTool
from .mcp import Decision, McpApproval, McpClient, McpTool, connect, load_config
from .tools.load_skill import LoadSkillTool
from .tools.memory import ReadMemoryTool, RememberTool
from .memory import Memory
from .rag import Collection, IndexSpec, SearchSpec
from .skills import BUILTIN, Skill, load_skills, usable
from .subagents import BUILTIN as BUILTIN_AGENTS, Child, Definition, DelegateTool, load_definitions
from .subagents import usable as usable_agents


@dataclass(slots=True)
class Application:
    """装配好的一整套东西。"""

    agent: Agent
    db: Database | None               # 没配 DATABASE_URL 时是 None
    inspector: SchemaInspector | None
    tools: ToolRegistry
    llm: LLMProvider
    settings: Settings
    # 这次会话查出过的结果（r1、r2…）：界面展开 {{r3}}、/save 都从这里拿
    results: ResultStore
    export_dir: Path                  # CSV 写到哪（/save 和 export_csv 共用）
    work_dir: Path                    # 沙箱的工作目录：inputs/ 放上传的文件，figures/ 放图
    sandboxes: dict[str, Sandbox] = field(default_factory=dict)   # "python" / "r"
    skills: list[Skill] = field(default_factory=list)             # 能用的技能（要的工具都在）
    skill_problems: list[str] = field(default_factory=list)       # 写坏了、被跳过的 SKILL.md
    subagents: list[Definition] = field(default_factory=list)     # 能分派的子 Agent 类型（没开是空的）
    subagent_problems: list[str] = field(default_factory=list)    # 写坏了、被跳过的子 Agent 定义
    memory: Memory | None = None                                  # 关了长期记忆是 None
    pending_uploads: list[Path] = field(default_factory=list)     # 上传了、还没告诉模型的文件
    mcp_clients: dict[str, McpClient] = field(default_factory=dict)  # 这次起的 MCP 服务器（共用的不在这里）
    mcp_tools: list[McpTool] = field(default_factory=list)
    mcp_problems: list[str] = field(default_factory=list)            # 连不上的 MCP 服务器

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

    def restore(self, history: list[Entry], turn: InterruptedTurn | None) -> None:
        """接着以前的会话：历史接回去，上次没跑完的那一轮放回 agent.interrupted。
        程序重启过，沙箱内核是新的，要的话补一句提醒接在进度后面。"""
        self.agent.context.restore(history)
        if turn is not None and (note := self.fresh_kernel_note(turn)):
            turn = replace(turn, entries=(*turn.entries, Message.user(note).with_meta(synthetic=True)))
        self.agent.interrupted = turn

    def reset(self) -> None:
        """清空对话，内核也换个空的：新对话不该看到上一段留下的变量。MCP 服务器不用重起。"""
        self.agent.reset()
        for sandbox in self.sandboxes.values():
            sandbox.close()

    def close(self) -> None:
        for sandbox in self.sandboxes.values():
            sandbox.close()
        for client in self.mcp_clients.values():
            client.close()


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
    mcp_clients: dict[str, McpClient] | None = None,
    ask_mcp: Callable[[ToolCall, McpTool], Decision] | None = None,
) -> Application:
    """把所有零件拼成一个能跑的 Agent。

    settings 不传就从 .env 读；llm 可以塞假的（测试）。results 由界面先建好传进来
    （打印事件的 sink 要用它展开 {{r3}}）；export_dir、work_dir 不传用 settings 里的。
    mcp_clients：已经连好的 MCP 服务器（评测里几个 Agent 共用一个），按名字替代配置里的；
    ask_mcp：外部工具第一次调用时怎么问用户，不给就只放行 autoApprove 里的。
    用完要 close()：沙箱是个容器，MCP 服务器是子进程。
    """
    settings = settings or Settings()
    # 只有一个 Agent：项目的约定来自 AGENTS.md，有哪些工具只看环境里配了什么
    project = Path(settings.project_dir)
    if not project.is_dir():
        raise FileNotFoundError(f"项目目录 {project} 不存在")
    agents_md = project / "AGENTS.md"
    rules = agents_md.read_text(encoding="utf-8") if agents_md.is_file() else ""

    # --- 数据层 --- 配了库才连。指定了 schema 就只看它（SQL 不带前缀也能找到表）；
    # 没指定就看库里所有有表的 schema，表名带前缀写
    db = inspector = None
    if settings.database_url:
        schemas = (settings.db_schema,) if settings.db_schema else None
        db = Database(settings.database_url, statement_timeout_ms=settings.db_statement_timeout_ms,
                      search_path=schemas or ())
        inspector = SchemaInspector(db, schemas=schemas)

    # --- 模型层 --- 放在工具前面：注册哪些工具要看模型能力（能不能看图）
    llm = llm or build_provider(settings)

    # --- 工具层 ---
    # run_sql 往里存、export_csv 按编号取、界面、评测和沙箱读：只有这一份
    results = results if results is not None else ResultStore()
    export_dir = export_dir or Path(settings.export_dir)
    work_dir = work_dir or Path(settings.work_dir)
    tools = ToolRegistry()
    if db is not None:
        for tool in (ListTablesTool(inspector), DescribeTableTool(db, inspector),
                     RunSqlTool(db, results), ExportCsvTool(db, results, export_dir)):
            tools.register(tool)
    # 沙箱：.env 里开着才有。第一次调用才启动容器；两个容器挂同一个工作目录
    sandboxes: dict[str, Sandbox] = {}
    data_dir = Path(settings.data_dir) if settings.data_dir else None
    if data_dir is not None and not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在（在项目根目录下运行？数据下载了吗？）")
    kernels = {"python": (settings.sandbox_image, PYTHON_KERNEL), "r": (settings.sandbox_r_image, R_KERNEL)}

    def make_sandbox(kind: str, figures_dir: Path | None = None) -> Sandbox:
        image, kernel = kernels[kind]
        return Sandbox.docker(
            image, kernel, work_dir, result_resolver(results), timeout_s=settings.sandbox_timeout_s,
            memory=settings.sandbox_memory, cpus=settings.sandbox_cpus, save=result_saver(results, kind),
            data_dir=data_dir, figures_dir=figures_dir,
        )

    for kind, enabled, tool_class in (("python", settings.python_sandbox, RunPythonTool),
                                      ("r", settings.r_sandbox, RunRTool)):
        if enabled:
            sandboxes[kind] = make_sandbox(kind)
            tools.register(tool_class(sandboxes[kind]))
    # 读文档、看图：有沙箱才有文件可读。看图要模型能看：不注册的话，提示词里「交付前看一眼」那句也就没了
    if sandboxes:
        paths = SandboxPaths(work_dir, data_dir)
        tools.register(ReadFileTool(paths))
        if llm.vision:
            tools.register(ViewImageTool(paths))
    # 知识库：配了文档目录才有。第一次搜的时候才加载索引（有新文档才解析、编码）
    collections = _collections(settings)
    if collections:
        retrievers = ("bm25", "dense") if settings.rag_embedder else ("bm25",)
        tools.register(ListDocsTool(collections))
        tools.register(SearchDocsTool(collections, SearchSpec(retrievers, reranker=settings.rag_reranker)))
        tools.register(ReadDocTool(collections))
    # MCP：外部服务器的工具。连不上的记下来给界面提示，不让整个程序起不来
    mcp_owned: dict[str, McpClient] = {}
    mcp_tools: list[McpTool] = []
    mcp_problems: list[str] = []
    if settings.mcp_enabled:
        config = Path(settings.mcp_config) if settings.mcp_config else project / ".mcp.json"
        mcp_owned, mcp_tools, mcp_problems = connect(load_config(config), settings.mcp_timeout_s, mcp_clients)
        for tool in mcp_tools:
            tools.register(tool)
    if mcp_tools:
        approval_hook = _chain(McpApproval({t.name: t for t in mcp_tools}, ask_mcp), approval_hook)
    mcp_servers = {t.server: t.client for t in mcp_tools}
    # 技能：项目的（.agents/skills/）盖过内置的；要的工具不在就不列，一个都没有就不注册 load_skill
    found, skill_problems = load_skills([project / ".agents" / "skills", BUILTIN])
    skills = usable(found, [t.name for t in tools])
    if skills:
        tools.register(LoadSkillTool(skills))
    # 长期记忆：目录（每条一行摘要）会话开始时拼进系统提示词，正文 read_memory 按需读
    memory = Memory.open(Path(settings.memory_dir).expanduser(), project) if settings.memory_enabled else None
    if memory:
        tools.register(RememberTool(memory))
        tools.register(ReadMemoryTool(memory))
    # 中途问用户：界面拿到 AwaitingUser 去问，回答用 agent.resume() 接回来。没人可问的场合关掉
    if settings.ask_user:
        tools.register(AskUserTool())

    # 系统提示词：主 Agent 和子 Agent 按各自的工具拼，其余一样
    def system_prompt(names: list[str], role: str = "") -> str:
        return build_system_prompt(names, rules=rules, data_dir=data_dir is not None,
                                   skills=usable(skills, names), memory="read_memory" in names,
                                   collections=[(c.name, len(c.files())) for c in collections],
                                   mcp=[(n, c.instructions) for n, c in mcp_servers.items()
                                        if any(t.startswith(f"mcp__{n}__") for t in names)],
                                   role=role)

    session_context = _session_context(inspector, memory)
    wrap_up = WRAP_UP_BEST_GUESS if settings.wrap_up == "best_guess" else WRAP_UP

    # 子 Agent：全新的上下文，工具按类型从主 Agent 的里面挑。数据库、知识库、结果编号共用同一批实例；
    # 沙箱各起各的（几个子 Agent 同时跑，变量不能串），往 figures/ 存的落在 figures/<任务号>/
    definitions: list[Definition] = []
    agent_problems: list[str] = []
    if settings.subagents:
        found_agents, agent_problems = load_definitions([project / ".agents" / "agents", BUILTIN_AGENTS])
        definitions = usable_agents(found_agents, [t.name for t in tools])

    def spawn(definition: Definition, task: str, emit: Callable[[Event], None]) -> Child:
        allowed = definition.allowed([t.name for t in tools])
        figures = work_dir.resolve() / "figures" / task
        own: dict[str, Sandbox] = {}
        child_tools = ToolRegistry()
        for tool in tools:
            if tool.name not in allowed:
                continue
            if isinstance(tool, (RunPythonTool, RunRTool)):
                kind = "python" if isinstance(tool, RunPythonTool) else "r"
                own[kind] = make_sandbox(kind, figures)
                tool = type(tool)(own[kind])
            elif isinstance(tool, (ReadFileTool, ViewImageTool)):
                tool = type(tool)(SandboxPaths(work_dir, data_dir, figures))
            child_tools.register(tool)
        names = [t.name for t in child_tools]
        agent = Agent(llm=llm, tools=child_tools, system_prompt=system_prompt(names, definition.role()),
                      context=_context(settings, llm, child_tools), max_steps=definition.max_steps,
                      approval_hook=approval_hook, on_event=emit, session_context=session_context,
                      wrap_up_prompt=wrap_up)

        def close() -> None:
            for sandbox in own.values():
                sandbox.close()

        def files() -> list[str]:
            root = work_dir.resolve()
            return sorted(p.relative_to(root).as_posix() for p in figures.rglob("*") if p.is_file())

        return Child(agent, close, files)

    delegate = None
    if definitions:
        delegate = DelegateTool(definitions, spawn, results)
        tools.register(delegate)

    # --- Agent ---
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt([t.name for t in tools]),
        context=_context(settings, llm, tools),
        max_steps=settings.max_steps,
        approval_hook=approval_hook,
        finish_turn_hook=finish_turn_hook,
        on_event=on_event,
        session_context=session_context,
        wrap_up_prompt=wrap_up,
        stream=settings.stream,
    )
    if delegate is not None:
        delegate.bind(agent)

    return Application(
        agent=agent, db=db, inspector=inspector,
        tools=tools, llm=llm, settings=settings, results=results, export_dir=export_dir,
        work_dir=work_dir, sandboxes=sandboxes, skills=skills, skill_problems=skill_problems,
        subagents=definitions, subagent_problems=agent_problems,
        memory=memory, mcp_clients=mcp_owned, mcp_tools=mcp_tools, mcp_problems=mcp_problems,
    )


def _context(settings: Settings, llm: LLMProvider, tools: ToolRegistry) -> Context:
    """先清理（几乎无损），清理完还超标再压缩（有损）。"""
    # 触发线不超过「窗口 - 余量」：换成小窗口的模型时不能等到 10 万才动手
    def cap(trigger: int) -> int:
        if llm.context_window:
            return min(trigger, llm.context_window - settings.context_reserve_tokens)
        return trigger

    return Context([
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


def _chain(first: ApprovalHook, then: ApprovalHook | None) -> ApprovalHook:
    """两道审批都过了才执行。"""
    if then is None:
        return first

    def hook(call: ToolCall) -> tuple[bool, str]:
        allowed, reason = first(call)
        return then(call) if allowed else (allowed, reason)
    return hook


def _collections(settings: Settings) -> list[Collection]:
    """DOCS_DIRS（; 隔开）→ 知识库，名字是目录名。"""
    dirs = [Path(d.strip()) for d in settings.docs_dirs.split(";") if d.strip()]
    if missing := [str(d) for d in dirs if not d.is_dir()]:
        raise FileNotFoundError(f"文档目录不存在：{'、'.join(missing)}（在项目根目录下运行？）")
    names = [d.resolve().name for d in dirs]
    if len(set(names)) < len(names):
        raise ValueError(f"DOCS_DIRS 里有同名的目录：{names}（知识库名 = 目录名，要不一样）")
    spec = IndexSpec(embedder=settings.rag_embedder)
    root = Path(settings.rag_dir).expanduser()
    return [Collection(name, d.resolve(), root, spec) for name, d in zip(names, dirs)]


def _session_context(inspector: SchemaInspector | None, memory: Memory | None) -> Callable[[], str] | None:
    """会话开始时现算一次、附在系统提示词后面的：库概览、长期记忆目录。/reset 之后重算（记忆可能变了）。"""
    parts = [p for p in (inspector.overview if inspector else None,
                         (lambda: memory.index(date.today())) if memory else None) if p]
    if not parts:
        return None
    return lambda: "\n\n".join(p() for p in parts)
