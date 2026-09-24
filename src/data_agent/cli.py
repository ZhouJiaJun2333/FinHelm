"""命令行界面。

这一层只负责「怎么显示」——订阅 Agent 抛出的事件，打到终端。
换成 Web 界面时，只要换一个 EventSink，core/ 里一行都不用动。
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from .app import Application, build_application
from .core.errors import AgentError
from .core.events import (
    ContextEdited,
    Event,
    LLMResponded,
    StepLimitReached,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    TurnContinued,
)
from .core.messages import Usage
from .settings import Settings

BANNER = """
┌──────────────────────────────────────────────────┐
│  SQL 数据分析 Agent                               │
│                                                  │
│  /tables  看库里有哪些表      /tools  看有哪些工具 │
│  /context 看上下文用量        /reset  清空对话   │
│  /exit    退出                                   │
└──────────────────────────────────────────────────┘"""


def make_console_sink(verbose: bool):
    """把 Agent 事件打印到终端。"""

    def sink(event: Event) -> None:
        match event:
            case LLMResponded(text=text, tool_calls=calls, usage=usage, context_window=window):
                # 只打印「动手之前说的话」。最终回答由主循环统一打印，
                # 否则同一段话会出现两遍。
                if calls:
                    if text:
                        print(f"\n🤖 {text}")
                    print(f"   ↳ 调用：{', '.join(calls)}")
                print(f"   📊 {_usage_line(usage, window)}")

            case ToolStarted(name=name, arguments=args):
                shown = _preview(args, 400 if verbose else 200)
                print(f"\n🔧 {name}  {shown}")

            case ToolFinished(ok=ok, content=content, elapsed_ms=ms):
                mark = "✅" if ok else "❌"
                body = content if verbose else _preview(content, 800)
                print(f"{mark} ({ms}ms)\n{body}")

            case ToolDenied(name=name, reason=reason):
                print(f"\n⛔ 已拒绝 {name}：{reason}")

            case TurnContinued(nudge=nudge):
                print(f"\n🔁 判定未完成，继续：{nudge}")

            case StepLimitReached(max_steps=n):
                print(f"\n⚠️ 触发步数上限 {n}")

            case ContextEdited(description=what, tokens_before=before, tokens_after=after):
                print(f"\n🧹 {what}：上下文 {_k(before)} → {_k(after)}（估算）")

    return sink


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
    match cmd:
        case "/exit" | "/quit":
            print("再见。")
            raise SystemExit(0)

        case "/reset":
            app.agent.reset()
            print("对话已清空。")

        case "/tools":
            for tool in app.tools:
                print(f"  · {tool.name}\n      {tool.description}")

        case "/tables":
            print(app.inspector.overview())

        case "/context":
            _print_context(app)

        case _:
            return False
    return True


# -------------------------------------------------------------------- main
def main() -> None:
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    verbose = "--verbose" in sys.argv

    settings = Settings()
    try:
        app = build_application(settings, on_event=make_console_sink(verbose))
    except Exception as exc:
        print(f"❌ 初始化失败：{type(exc).__name__}: {exc}")
        print("检查 .env 配置（参考 .env.example）")
        return

    # 启动时就把数据库连通性验掉，别等跑到一半才报错
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

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not user_input:
            continue
        if user_input.startswith("/"):
            if not handle_command(user_input, app):
                print(f"未知命令：{user_input}")
            continue

        try:
            answer = app.agent.run(user_input)
            print(f"\n💬 {answer}")
        except KeyboardInterrupt:
            print("\n已中断本轮。")
        except AgentError as exc:
            # 我们自己抛的，消息里已经写清楚该怎么办了
            print(f"\n⚠️ {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ 出错：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
