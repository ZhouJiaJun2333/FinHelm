"""命令行界面：订阅 Agent 的事件打到终端，处理斜杠命令。"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .app import Application, build_application
from .core.agent import AwaitingUser, InterruptedTurn, PendingQuestion
from .core.context import Entry, turn_starts
from .core.errors import AgentError
from .core.events import (
    AutoCompactionPaused,
    ContextEdited,
    ContextEditFailed,
    ContextOverflowed,
    Event,
    LLMResponded,
    StepLimitReached,
    StepStarted,
    TextDelta,
    ToolCallRepeated,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
    TurnEnded,
    TurnResumed,
)
from .core.messages import Message, ToolCall, Usage
from .mcp import Decision, McpTool
from .memory import SCOPES, age
from .session import Session
from .settings import Settings
from .tools.sandbox import Execution
from .tools.sql.export_csv import export_result
from .tools.sql.results import REF, ResultStore, StoredResult, markdown_table

REF_NAME = re.compile(r"r\d+")

# 终端里一张结果表最多显示几行，要完整的就 /save
TERMINAL_ROWS = 20

BANNER = """
┌─────────────────────────────────────────────────────┐
│  FinHelm · 通用数据分析 Agent                       │
│                                                     │
│  /tables  看库里有哪些表      /tools   看有哪些工具 │
│  /attach  上传文件（/attach 路径1 "带 空格的路径2"）│
│  /continue 接着跑暂停、出错的那一轮（可带一句话）   │
│  /context 看上下文用量        /compact 压缩上下文   │
│  /save    把最近的结果存成 CSV（/save r3 指定编号） │
│  /skills  看有哪些技能        /skill:名字 按技能做  │
│  /memory  看长期记忆（改、删直接动文件，或跟我说）  │
│  /mcp     看外部 MCP 服务器和它们的工具             │
│  /reset   清空对话            /exit    退出         │
└─────────────────────────────────────────────────────┘"""


# ---------------------------------------------------------- 工具结果怎么显示
def _show_table(table: StoredResult) -> str:
    r = table.result
    title = f"（{table.title}）" if table.title else ""
    text = f"结果 {table.ref}{title}：{r.row_count} 行 × {len(r.columns)} 列\n"
    text += markdown_table(r.columns, r.rows[:TERMINAL_ROWS])
    if r.row_count > TERMINAL_ROWS:
        text += f"\n…终端只显示前 {TERMINAL_ROWS} 行。/save {table.ref} 可存成 CSV"
    if r.truncated:
        text += f"\n⚠️ 结果超过 {r.row_count} 行，只取了前 {r.row_count} 行"
    return text


def _show_execution(ex: Execution) -> str:
    """输出可能很长会被折叠，图的路径单独列出来，保证用户看得到。"""
    text = _preview("\n\n".join(p for p in (ex.output.rstrip(), ex.value or "") if p), 800)
    return "\n".join([text, *(f"🖼  {path.resolve()}" for path in ex.figures)]).strip()


# 按工具名找渲染函数，参数是 details（学 pi 的 renderResult）。没登记的显示模型看到的那份
RESULT_RENDERERS: dict[str, Callable[[Any], str]] = {
    "run_sql": _show_table,
    "run_python": _show_execution,
    "run_r": _show_execution,
}


DIM, RESET = "\x1b[2m", "\x1b[0m"


def make_console_sink(verbose: bool, results: ResultStore):
    """results 用来展开回答里的 {{r3}}。

    流式时文字边来边打，{{r3}} 先原样出来，这一步说完再把表格补在后面。
    最终回答流式打过了就不再打；没流式（STREAM=false、步数用完的兜底话）在 TurnEnded 时打。
    """
    live = {"text": False, "thinking": False}      # 这一步已经开始打正文 / 思考了
    streamed = ""                                  # 最近一段流式打出来的回复

    def end_thinking() -> None:
        # --verbose 时思考用暗色原样打，正文开始或这一步结束时关掉暗色
        if verbose and live["thinking"] and not live["text"]:
            print(RESET)
            live["thinking"] = False

    def sink(event: Event) -> None:
        nonlocal streamed
        match event:
            case StepStarted():
                live.update(text=False, thinking=False)
                streamed = ""

            case TextDelta(text=text, thinking=True):
                if verbose:
                    print(("" if live["thinking"] else f"\n💭 {DIM}") + text, end="", flush=True)
                elif not live["thinking"]:
                    print("\n💭 思考中…", flush=True)
                live["thinking"] = True

            case TextDelta(text=text):
                if not live["text"]:
                    text = text.lstrip()             # 思考完常常先来几个换行
                    if not text:
                        return
                    end_thinking()
                    print("\n💬 ", end="")
                    live["text"] = True
                print(text, end="", flush=True)

            case LLMResponded(text=text, tool_calls=calls, usage=usage, context_window=window):
                end_thinking()
                if live["text"]:
                    print()
                    streamed = text
                    for ref in REF.findall(text):
                        print(_show_table(t) if (t := results.get(ref)) else f"（找不到结果 {ref}）")
                elif calls and text:
                    # 不流式时只打印动手之前说的话，最终回答等 TurnEnded
                    print(f"\n🤖 {results.expand(text, _show_table)}")
                if calls:
                    print(f"   ↳ 调用：{', '.join(calls)}")
                print(f"   📊 {_usage_line(usage, window)}")

            case TurnEnded(answer=answer) if answer and answer != streamed:
                print(f"\n💬 {results.expand(answer, _show_table)}")

            case TurnEnded():
                end_thinking()                       # 断在思考中间

            case ToolStarted(name="ask_user"):
                pass                                  # 问题由 _ask 打印

            case ToolStarted(name=name, arguments=args):
                shown = _preview(args, 400 if verbose else 200)
                print(f"\n🔧 {name}  {shown}")

            case ToolFinished(name=name, is_error=False, details=details, elapsed_ms=ms) \
                    if details is not None and name in RESULT_RENDERERS:
                print(f"✅ ({ms}ms)\n{RESULT_RENDERERS[name](details)}")

            case ToolFinished(is_error=is_error, content=content, elapsed_ms=ms):
                mark = "❌" if is_error else "✅"
                body = content if verbose else _preview(content, 800)
                print(f"{mark} ({ms}ms)\n{body}")

            case ToolDenied(name=name, reason=reason):
                print(f"\n⛔ 已拒绝 {name}：{reason}")

            case ToolCallRepeated(name=name, count=n):
                print(f"\n🔁 同样的参数第 {n} 次调用 {name}，已提醒模型换思路")

            case TurnContinued(nudge=nudge):
                print(f"\n🔁 判定未完成，继续：{nudge}")

            case TurnResumed(steps=n, message=message):
                print(f"\n▶️ 从第 {n + 1} 步接着跑" + (f"：{message}" if message else ""))

            case StepLimitReached(max_steps=n, wrapped_up=wrapped, failure=failure):
                print(f"\n⚠️ 触发步数上限 {n}" + ("，已根据现有结果收尾" if wrapped else f"，收尾没成（{failure}）"))

            case ContextOverflowed():
                print("\n⚠️ 请求超出了模型的上下文窗口，强制整理后重试")

            case ContextEdited(description=what, tokens_before=before, tokens_after=after,
                               usage=usage):
                cost = f"，写摘要花了 {_k(usage.prompt_tokens + usage.output)} token" if usage.output else ""
                print(f"\n🧹 {what}：上下文 {_k(before)} → {_k(after)}（估算）{cost}")

            case ContextEditFailed(reason=reason):
                print(f"\n⚠️ 压缩没做成：{reason} 这一步带着没压的上下文接着跑")

            case AutoCompactionPaused(failures=n):
                print(f"\n⚠️ 自动压缩连续失败 {n} 次，本会话不再自动压缩（清理照常）。"
                      "可以 /compact 手动再试，成功后恢复")

    return sink


# ---------------------------------------------------------------- /save
def parse_save(arg: str, refs: list[str]) -> tuple[str | None, str]:
    """/save 的参数 → (编号, 文件名)，编号不存在返回 None。

        /save、/save r3、/save r3 华东订单、/save 华东订单（第一个词不像编号就当文件名）
    """
    first, _, rest = arg.strip().partition(" ")
    if REF_NAME.fullmatch(first):
        return (first if first in refs else None), rest.strip()
    return (refs[-1] if refs else None), arg.strip()


def _save(arg: str, app: Application) -> None:
    """和 export_csv 工具走同一个函数（SQL 结果按当时的 SQL 重跑，沙箱存的表直接写）。"""
    ref, filename = parse_save(arg, app.results.refs())
    table = app.results.get(ref) if ref else None
    if table is None:
        print(f"用法：/save [r3] [文件名]，不写编号就存最近一个结果。"
              f"本次对话里的编号：{'、'.join(app.results.refs()) or '还没有'}")
        return
    try:
        done = export_result(app.db, table, app.export_dir, filename or f"{ref}.csv")
        print(done.describe(ref))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 导出失败：{type(exc).__name__}: {exc}")


def parse_paths(arg: str) -> list[Path]:
    """/attach 的参数：空格分开，带空格的路径加引号。不用 POSIX 规则：Windows 路径里的反斜杠要留着。"""
    return [Path(p.strip('"').strip("'")) for p in shlex.split(arg, posix=False)]


def _attach(arg: str, app: Application) -> None:
    if not app.sandboxes:
        print("没有开沙箱（PYTHON_SANDBOX / R_SANDBOX），上传了模型也读不了。")
        return
    paths = parse_paths(arg)
    if not paths:
        print('用法：/attach 文件1 "带 空格的文件2"。文件会复制进会话的 work/inputs/，下一条消息会告诉模型')
        return
    try:
        for path in app.attach(paths):
            print(f"📎 {path.name} → {path.resolve()}")
        print("下一条消息会告诉模型有这些文件。")
    except FileNotFoundError as exc:
        print(f"❌ {exc}")


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _of_window(tokens: int, window: int | None) -> str:
    if not window:
        return _k(tokens)
    return f"{_k(tokens)} / {_k(window)}（{tokens / window:.1%}）"


def _usage_line(usage: Usage, window: int | None) -> str:
    """一次调用之后：对话占了多少上下文，这次输入里多少走了缓存。"""
    if usage.context_tokens == 0:
        return "用量：厂商没有返回"
    return (
        f"上下文 {_of_window(usage.context_tokens, window)}"
        f" · 本次输入 {_k(usage.prompt_tokens)}（缓存命中 {_k(usage.cache_read)}）"
        f" · 输出 {_k(usage.output)}"
    )


def _print_context(app: Application) -> None:
    est = app.agent.context_usage()
    total = app.agent.session_usage
    print(f"下一次请求的输入（估算）：{_of_window(est.tokens, app.llm.context_window)}")
    if est.anchor_index is None:
        print("  全部是估算（还没有模型返回过用量），含系统提示词和工具定义")
    else:
        print(f"  ├ 精确 {_k(est.usage_tokens):>7}  ← 第 {est.anchor_index + 1} 条消息的 usage")
        print(f"  └ 估算 {_k(est.trailing_tokens):>7}  ← 之后新加的消息")
    for line in app.agent.context.status():
        print(line)
    print(
        f"本次会话累计：输入 {_k(total.prompt_tokens)}"
        f"（缓存命中 {_k(total.cache_read)}，写入 {_k(total.cache_write)}）"
        f" · 输出 {_k(total.output)}"
    )


def _preview(obj: object, limit: int) -> str:
    text = obj if isinstance(obj, str) else str(obj)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n   …（已折叠，共 {len(text)} 字符，加 --verbose 看全部）"


# ---------------------------------------------------------------- 斜杠命令
def handle_command(cmd: str, app: Application) -> bool:
    """处理 /开头的命令。返回 True 表示已处理。"""
    name, _, arg = cmd.partition(" ")
    match name:
        case "/exit" | "/quit":
            print("再见。")
            raise SystemExit(0)

        case "/reset":
            app.reset()
            print("对话已清空。")

        case "/tools":
            for tool in app.tools:
                print(f"  · {tool.name}\n      {tool.description}")

        case "/tables":
            print(app.inspector.overview() if app.inspector else "没有连数据库（DATABASE_URL 为空）。")

        case "/attach":
            _attach(arg, app)

        case "/continue":
            if app.agent.interrupted is None:
                print("没有暂停或出错的回合可以接着跑。")
            else:
                answer = arg.strip()
                if (q := app.agent.interrupted.pending) is not None:
                    answer = pick_option(answer, q.options)
                _run(app, lambda: app.agent.resume(answer))

        case "/context":
            _print_context(app)

        case "/save":
            if app.db is None:
                print("没有连数据库，也没有沙箱存下的结果可存。")
            else:
                _save(arg, app)

        case "/skills":
            if not app.skills:
                print("没有能用的技能。")
            for skill in app.skills:
                print(f"  · {skill.name}（{skill.path.parent}）\n      {skill.description}")

        case "/memory":
            _print_memory(app)

        case "/mcp":
            _print_mcp(app)

        case _ if name.startswith("/skill:"):
            _run_skill(name.removeprefix("/skill:"), arg, app)

        case "/compact":
            # 整理的过程由事件打印，这里只管「什么都没做」和失败
            try:
                if not app.agent.compact():
                    print("没有可整理的内容（压缩至少要保留当前这一轮，对话还太短）。")
            except AgentError as exc:
                print(f"⚠️ {exc}（上下文没有改动）")

        case _:
            return False
    return True


def _print_mcp(app: Application) -> None:
    if not app.mcp_tools and not app.mcp_problems:
        print("没有配 MCP 服务器（项目目录的 .mcp.json，或者 MCP_CONFIG 指定的文件）。")
    servers: dict[str, list] = {}
    for tool in app.mcp_tools:
        servers.setdefault(tool.server, []).append(tool)
    for name, tools in servers.items():
        auto = tools[0].client.config.auto_approve
        print(f"  · {name}：{len(tools)} 个工具")
        for t in tools:
            print(f"      {t.remote_name}{'（自动放行）' if t.remote_name in auto else ''}")
    for problem in app.mcp_problems:
        print(f"  ⚠️ {problem}")


def ask_mcp(call: ToolCall, tool: McpTool) -> Decision:
    """外部工具第一次调用：给用户看参数，问一下。"""
    args = json.dumps(call.arguments, ensure_ascii=False)
    print(f"\n🔌 要调用外部 MCP 服务器 {tool.server} 的工具 {tool.remote_name}，参数：{_preview(args, 300)}")
    while True:
        try:
            choice = input("   1. 这次允许　2. 本会话都允许　3. 拒绝 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if choice in ("1", "2", "3"):
            return {"1": "once", "2": "session", "3": "deny"}[choice]


def _print_memory(app: Application) -> None:
    if app.memory is None:
        print("长期记忆关着（MEMORY_ENABLED=false）。")
        return
    today = date.today()
    for scope, store in app.memory.stores.items():
        entries = store.entries()
        print(f"[{SCOPES[scope]}] {store.root}（{len(entries)} 条）")
        for e in entries:
            print(f"  · {e.name}（{age(e.updated, today)}）：{e.description}")
    print("改、删：直接改对应的 .md 文件，或者跟我说「忘掉……」「……改成……」。新会话才会读到手改的内容。")


def _run_skill(skill_name: str, request: str, app: Application) -> None:
    """/skill:名字 要做的事：模型有时会漏加载技能，用户可以直接指定。技能全文拼在这条消息前面（学 pi）。"""
    skill = next((s for s in app.skills if s.name == skill_name), None)
    if skill is None:
        print(f"没有叫 {skill_name} 的技能。/skills 看有哪些。")
    elif not request.strip():
        print(f"用法：/skill:{skill.name} 要做的事")
    else:
        message = f"[用户指定按技能 {skill.name} 来做，下面是技能全文]\n{skill.body()}\n\n{request.strip()}"
        _run(app, lambda: app.agent.run(app.with_uploads(message)))


# -------------------------------------------------------------------- main
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python run.py", description="FinHelm：金融垂直领域的通用 Agent")
    parser.add_argument("--verbose", action="store_true", help="工具输出不折叠")
    parser.add_argument("--resume", nargs="?", const="", metavar="会话ID",
                        help="接着上次的对话聊；不写 ID 就是最近的一次")
    return parser.parse_args(argv)


def open_session(base: Path, resume: str | None) -> tuple[Session, list[Entry]]:
    """resume：None = 新开，"" = 最近的，其他 = 那个 ID。"""
    if resume is None:
        return Session.create(base), []
    session = Session.open(base, resume or None)
    return session, session.load()


def _handle(user_input: str, app: Application) -> None:
    """处理一次输入：斜杠命令，回答它刚问的问题，或者让 Agent 跑一轮。"""
    if user_input.startswith("/"):
        if not handle_command(user_input, app):
            print(f"未知命令：{user_input}")
        return
    turn = app.agent.interrupted
    if turn is not None and turn.pending is not None:
        answer = pick_option(user_input, turn.pending.options)
        _run(app, lambda: app.agent.resume(answer))
        return
    _run(app, lambda: app.agent.run(app.with_uploads(user_input)))


def pick_option(text: str, options: tuple[str, ...]) -> str:
    """回答提问：输入编号就是选那个选项，别的原样当回答。"""
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    return text


def _ask(q: PendingQuestion) -> None:
    print(f"\n❓ {q.question}")
    for n, option in enumerate(q.options, 1):
        print(f"   {n}. {option}")
    how = "输入编号选一个，或者直接写你的回答" if q.options else "直接写你的回答"
    print(f"   （{how}；/continue 不回答、让它自己判断；/reset 放弃这一轮）")


def _run(app: Application, turn: Callable[[], str]) -> None:
    """跑一轮（新问题或接着跑）。断了的话进度留在 agent.interrupted，提示可以 /continue。"""
    try:
        turn()                  # 回答由事件打印（流式时边生成边打）
    except AwaitingUser as asked:
        _ask(asked.pending)
    except KeyboardInterrupt:
        print("\n⏸ 已暂停本轮。" + _resume_hint(app.agent.interrupted))
    except AgentError as exc:
        print(f"\n⚠️ {exc}" + _resume_hint(app.agent.interrupted))
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 出错：{type(exc).__name__}: {exc}" + _resume_hint(app.agent.interrupted))


def _resume_hint(turn: InterruptedTurn | None) -> str:
    if turn is None:
        return ""
    done = f"做完了 {turn.steps} 步，" if turn.steps else ""
    return (f"\n   进度已保留（{done}/continue 接着跑，后面可以加一句话调整方向）；"
            "直接问新问题就放弃这一轮。")


def main() -> None:
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(sys.argv[1:])
    settings = Settings()
    try:
        session, history = open_session(Path(settings.sessions_dir), args.resume)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ 没法恢复会话：{exc}")
        return

    # 结果仓库先建好：打印事件的 sink 要用它展开 {{r3}}
    results = ResultStore(session.results_path)
    try:
        app = build_application(settings, on_event=make_console_sink(args.verbose, results),
                                results=results, export_dir=session.exports_dir,
                                work_dir=session.work_dir, ask_mcp=ask_mcp)
    except Exception as exc:
        print(f"❌ 初始化失败：{type(exc).__name__}: {exc}")
        print("检查 .env 配置（参考 .env.example）")
        return
    app.agent.context.restore(history)
    app.agent.checkpoint_hook = session.save_checkpoint     # 每走一步存一次：进程被杀了也能接着跑
    if args.resume is not None:
        app.agent.interrupted = _pending_turn(session, app)
    try:
        _repl(app, session, results, settings, history)
    finally:
        app.close()


def _pending_turn(session: Session, app: Application) -> InterruptedTurn | None:
    """上次没跑完的那一轮。程序重启过，沙箱内核是新的，要的话补一句提醒接在进度后面。"""
    turn = session.load_checkpoint()
    if turn is None:
        return None
    note = app.fresh_kernel_note(turn)
    if note:
        turn = replace(turn, entries=(*turn.entries, Message.user(note).with_meta(synthetic=True)))
    return turn


def _repl(app: Application, session: Session, results: ResultStore, settings: Settings,
          history: list[Entry]) -> None:
    version = "不连数据库（只分析上传的文件）"
    if app.db is not None:
        try:
            version = app.db.ping().split(",")[0]
        except Exception as exc:
            print(f"❌ 连不上数据库：{exc}")
            print("   数据库起来了吗？  cd docker && docker compose up -d")
            return

    print(BANNER)
    print(f"模型：{settings.provider} / {app.llm.model}")
    agents_md = Path(settings.project_dir) / "AGENTS.md"
    print(f"项目：{Path(settings.project_dir).resolve()}（{'有' if agents_md.is_file() else '没有'} AGENTS.md）")
    if app.inspector:
        print(f"数据库：{version}　schema：{', '.join(app.inspector.schemas)}")
    print(f"工具：{len(app.tools)} 个 —— {', '.join(t.name for t in app.tools)}")
    if app.skills:
        print(f"技能：{', '.join(s.name for s in app.skills)}（/skills 查看）")
    for problem in app.skill_problems:
        print(f"⚠️ 跳过了一个技能 {problem}")
    if app.mcp_tools:
        print(f"MCP：{', '.join(sorted({t.server for t in app.mcp_tools}))}（/mcp 查看）")
    for problem in app.mcp_problems:
        print(f"⚠️ MCP 服务器没连上：{_preview(problem, 200)}")
    print(f"会话：{session.root}（下次 python run.py --resume {session.id} 接着聊）")
    if history:
        print(f"已恢复 {len(turn_starts(history))} 轮对话、{len(results.refs())} 个查询结果")
    if (turn := app.agent.interrupted) is not None and turn.pending is not None:
        print(f"⏸ 上次那一轮停在一个问题上：{_preview(turn.question, 60)}")
        _ask(turn.pending)
    elif turn is not None:
        why = f"（{turn.reason}）" if turn.reason else "（程序被关掉了）"
        print(f"⏸ 上次有一轮没跑完{why}：{_preview(turn.question, 60)}" + _resume_hint(turn))

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user_input:
            continue
        try:
            _handle(user_input, app)
        finally:
            # 每处理完一次输入就落盘。失败的一轮已经回滚，历史没变，什么都不写；进度另外存。
            # 先写历史再删检查点：中间崩了，检查点的 base 对不上，读的时候会丢掉
            session.sync(app.agent.context.history)
            session.save_checkpoint(app.agent.interrupted)


if __name__ == "__main__":
    main()
