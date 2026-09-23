"""抓出真实发给模型服务器的 HTTP 请求体，看 tools 到底待在哪。

    python docs/raw_http_body.py

用 httpx 的 event hook 拦截 SDK 发出的请求，打印原始 JSON body。
这是字面意义上「线上传的那串字节」，没有任何加工。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:                       # 新版 SDK 依赖 httpx 2.x，包名叫 httpx2
    import httpx2 as httpx  # noqa: E402
except ImportError:        # 老环境仍是 httpx
    import httpx  # type: ignore[no-redef]  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from data_agent.core.messages import Message  # noqa: E402
from data_agent.llm.openai_provider import OpenAICompatibleProvider  # noqa: E402
from data_agent.settings import Settings  # noqa: E402

LINE = "=" * 76

TOOLS = [{
    "name": "run_sql",
    "description": "执行只读 SQL",
    "parameters": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "要执行的 SQL"}},
        "required": ["sql"],
    },
}]


def main() -> None:
    load_dotenv()
    settings = Settings()

    captured: list[dict] = []

    def capture(request: httpx.Request) -> None:
        captured.append(json.loads(request.read()))

    # 把带 hook 的 http client 塞给 SDK
    provider = OpenAICompatibleProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
    provider.client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        http_client=httpx.Client(event_hooks={"request": [capture]}),
    )

    history = [
        Message.user("2025年华东卖了多少？"),
        Message(role="assistant", content="我查一下",
                tool_calls=[__import__("data_agent.core.messages", fromlist=["ToolCall"])
                            .ToolCall("c1", "run_sql", {"sql": "SELECT 1"})]),
        Message.tool_result("c1", "| gmv |\n| 810万 |"),
    ]

    # 我们只关心「发出去的是什么」，服务端接不接受不影响这个实验。
    # （deepseek-flash 这类思考模型会要求把 reasoning_content 回传，
    #   我们的 provider 目前没存它，所以这里大概率会 400 —— 不影响抓包。）
    try:
        provider.chat(history, tools=TOOLS, system="你是数据分析助手。")
        print("（服务端正常返回）\n")
    except Exception as exc:  # noqa: BLE001
        print(f"（服务端返回了错误，不影响本实验：{type(exc).__name__}）\n")

    if not captured:
        print("没抓到请求，实验失败。")
        return
    body = captured[0]

    # ------------------------------------------------------------------
    print(LINE)
    print("真实 HTTP 请求体的顶层字段")
    print(LINE)
    for key, value in body.items():
        kind = type(value).__name__
        size = f"{len(value)} 项" if isinstance(value, (list, dict)) else repr(value)
        print(f"  {key:<12} {kind:<6} {size}")
    print()
    print("  ↑ tools 和 messages 是**平级的两个顶层字段**。")
    print("    tools 不在 messages 里面，messages 里也没有 tools 的定义。")

    # ------------------------------------------------------------------
    print()
    print(LINE)
    print("body['tools'] —— 每轮重发的工具定义")
    print(LINE)
    print(json.dumps(body["tools"], ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    print()
    print(LINE)
    print("body['messages'] —— 历史记录")
    print(LINE)
    print(json.dumps(body["messages"], ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    print()
    print(LINE)
    print("关键对照")
    print(LINE)
    msg_text = json.dumps(body["messages"], ensure_ascii=False)
    print(f"messages 里有 run_sql 这个**名字**吗？        {'run_sql' in msg_text}")
    print(f"messages 里有 run_sql 的**参数定义**吗？      "
          f"{'要执行的 SQL' in msg_text}")
    print()
    print("历史里有「我调用过 run_sql，参数是 xxx」这条记录，")
    print("但没有「run_sql 接受一个叫 sql 的字符串参数」这份定义。")
    print("定义只存在于 tools 字段里 —— 所以它必须每轮都发。")


if __name__ == "__main__":
    main()
