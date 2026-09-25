"""命令行界面：订阅 Agent 的事件打到终端，处理斜杠命令。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .app import Application, build_application
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
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
)
from .core.messages import Usage
from .session import Session
from .settings import Settings
from .tools.python.sandbox import Execution
from .tools.sql.export_csv import export_query
from .tools.sql.results import ResultStore, SqlResult, markdown_table

REF_NAME = re.compile(r"r\d+")

# 终端里一张结果表最多显示几行，要完整的就 /save
TERMINAL_ROWS = 20

BANNER = """
┌─────────────────────────────────────────────────────┐
│  FinHelm · 金融数据分析 Agent                       │
│                                                     │
│  /tables  看库里有哪些表      /tools   看有哪些工具 │
│  /context 看上下文用量        /compact 压缩上下文   │
│  /save    把最近的结果存成 CSV（/save r3 指定编号） │
│  /reset   清空对话            /exit    退出         │
└─────────────────────────────────────────────────────┘"""


# ---------------------------------------------------------- 工具结果怎么显示
def _show_table(table: SqlResult) -> str:
    r = table.result
    text = f"结果 {table.ref}：{r.row_count} 行 × {len(r.columns)} 列\n"
    text += markdown_table(r.columns, r.rows[:TERMINAL_ROWS])
    if r.row_count > TERMINAL_ROWS:
        text += f"\n…终端只显示前 {TERMINAL_ROWS} 行。/save {table.ref} 可存成 CSV"
    if r.truncated:
        text += f"\n⚠️ 结果超过 {r.row_count} 行，只取了前 {r.row_count} 行"
    return text


def _show_python(ex: Execution) -> str:
    """输出可能很长会被折叠，图的路径单独列出来，保证用户看得到。"""
    text = _preview("\n\n".join(p for p in (ex.output.rstrip(), ex.value or "") if p), 800)
    return "\n".join([text, *(f"🖼  {path.resolve()}" for path in ex.figures)]).strip()


# 按工具名找渲染函数，参数是 details（学 pi 的 renderResult）。没登记的显示模型看到的那份
RESULT_RENDERERS: dict[str, Callable[[Any], str]] = {
    "run_sql": _show_table,
    "run_python": _show_python,
}


def make_console_sink(verbose: bool, results: ResultStore):
    """results 用来展开过渡的话里的 {{r3}}。"""

    def sink(event: Event) -> None:
        match event:
            case LLMResponded(text=text, tool_calls=calls, usage=usage, context_window=window):
                # 只打印动手之前说的话，最终回答由主循环打印
                if calls:
                    if text:
                        print(f"\n🤖 {results.expand(text, _show_table)}")
                    print(f"   ↳ 调用：{', '.join(calls)}")
                print(f"   📊 {_usage_line(usage, window)}")

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

            case TurnContinued(nudge=nudge):
                print(f"\n🔁 判定未完成，继续：{nudge}")

            case StepLimitReached(max_steps=n):
                print(f"\n⚠️ 触发步数上限 {n}")

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
    """和 export_csv 工具走同一个函数（按当时的 SQL 重跑再写）。"""
    ref, filename = parse_save(arg, app.results.refs())
    table = app.results.get(ref) if ref else None
    if table is None:
        print(f"用法：/save [r3] [文件名]，不写编号就存最近一个结果。"
              f"本次对话里的编号：{'、'.join(app.results.refs()) or '还没有'}")
        return
    try:
        done = export_query(app.db, table.sql, app.export_dir, filename or f"{ref}.csv")
        print(done.describe(ref))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 导出失败：{type(exc).__name__}: {exc}")


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
            print(app.inspector.overview())

        case "/context":
            _print_context(app)

        case "/save":
            _save(arg, app)

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
    """处理一次输入：斜杠命令，或者让 Agent 跑一轮。"""
    if user_input.startswith("/"):
        if not handle_command(user_input, app):
            print(f"未知命令：{user_input}")
        return
    try:
        answer = app.agent.run(user_input)
        print(f"\n💬 {app.results.expand(answer, _show_table)}")
    except KeyboardInterrupt:
        print("\n已中断本轮。")
    except AgentError as exc:
        print(f"\n⚠️ {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 出错：{type(exc).__name__}: {exc}")


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
                                work_dir=session.work_dir)
    except Exception as exc:
        print(f"❌ 初始化失败：{type(exc).__name__}: {exc}")
        print("检查 .env 配置（参考 .env.example）")
        return
    app.agent.context.restore(history)
    try:
        _repl(app, session, results, settings, history)
    finally:
        app.close()


def _repl(app: Application, session: Session, results: ResultStore, settings: Settings,
          history: list[Entry]) -> None:
    try:
        version = app.db.ping().split(",")[0]
    except Exception as exc:
        print(f"❌ 连不上数据库：{exc}")
        print("   数据库起来了吗？  cd docker && docker compose up -d")
        return

    print(BANNER)
    print(f"模型：{settings.provider} / {app.llm.model}")
    print(f"数据库：{version}")
    print(f"工具：{len(app.tools)} 个 —— {', '.join(t.name for t in app.tools)}")
    print(f"会话：{session.root}（下次 python run.py --resume {session.id} 接着聊）")
    if history:
        print(f"已恢复 {len(turn_starts(history))} 轮对话、{len(results.refs())} 个查询结果")

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
            # 每处理完一次输入就落盘。失败的一轮已经回滚，历史没变，什么都不写
            session.sync(app.agent.context.history)


if __name__ == "__main__":
    main()
