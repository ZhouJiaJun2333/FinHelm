"""看清楚 context 里到底存了什么、每一轮请求发出去的是什么。

    python docs/what_is_in_context.py

不需要数据库，也不需要 API key —— 用假模型和假工具，结构和真的完全一样。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field  # noqa: E402

from data_agent.core.agent import Agent  # noqa: E402
from data_agent.core.messages import LLMResponse, Message, ToolCall  # noqa: E402
from data_agent.llm.base import LLMProvider  # noqa: E402
from data_agent.tools.base import Tool  # noqa: E402
from data_agent.tools.registry import ToolRegistry  # noqa: E402

LINE = "=" * 76


class FakeSqlTool(Tool):
    name = "run_sql"
    description = "执行只读 SQL"

    class Args(BaseModel):
        sql: str = Field(description="要执行的 SQL")

    def run(self, args: Args) -> str:
        return "| region | gmv |\n| 华东 | 8100531.47 |"


class FakeListTool(Tool):
    name = "list_tables"
    description = "列出所有表"

    class Args(BaseModel):
        pass

    def run(self, args: Args) -> str:
        return "shop.customers, shop.orders, shop.order_items, shop.products"


class SpyProvider(LLMProvider):
    """照剧本回答，同时把每一轮收到的 messages 拍照存下来。"""

    model = "spy"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script
        self.snapshots: list[list[Message]] = []

    def chat(self, messages, tools=None, system=None) -> LLMResponse:
        self.snapshots.append(list(messages))
        self.tools_seen = [t["name"] for t in (tools or [])]
        return self.script[min(len(self.snapshots) - 1, len(self.script) - 1)]


def dump(messages: list[Message], indent: str = "  ") -> None:
    for i, m in enumerate(messages):
        if m.role == "user":
            print(f"{indent}[{i}] user      「{m.content}」")
        elif m.role == "assistant":
            said = f"「{m.content}」" if m.content else "（没说话）"
            print(f"{indent}[{i}] assistant {said}")
            for c in m.tool_calls:
                print(f"{indent}                 └─ 发起调用 {c.name}({c.arguments})")
        elif m.role == "tool":
            body = m.content.replace("\n", " ⏎ ")
            print(f"{indent}[{i}] tool      ← {c_trunc(body)}   (对应调用 {m.tool_call_id})")


def c_trunc(s: str, n: int = 44) -> str:
    return s if len(s) <= n else s[:n] + "…"


def main() -> None:
    script = [
        LLMResponse(text="先看看有哪些表。", stop_reason="tool_use",
                    tool_calls=[ToolCall("c1", "list_tables", {})]),
        LLMResponse(text="现在按大区统计。", stop_reason="tool_use",
                    tool_calls=[ToolCall("c2", "run_sql", {"sql": "SELECT region, SUM(...)"})]),
        LLMResponse(text="华东最高，810 万。", stop_reason="end_turn"),
    ]

    llm = SpyProvider(script)
    agent = Agent(
        llm=llm,
        tools=ToolRegistry([FakeListTool(), FakeSqlTool()]),
        system_prompt="你是数据分析助手。",
    )

    answer = agent.run("2025年哪个大区销售额最高？")

    # ------------------------------------------------------------------
    print(LINE)
    print("一次 agent.run() 里，模型被请求了几次？每次看到什么？")
    print(LINE)
    for n, snapshot in enumerate(llm.snapshots, start=1):
        print(f"\n─── 第 {n} 轮请求 · messages 共 {len(snapshot)} 条 ───")
        dump(snapshot)
        print(f"  tools 参数：{llm.tools_seen}   ← 每轮都发，不在 messages 里")

    # ------------------------------------------------------------------
    print()
    print(LINE)
    print("注意看：用户只问了一次")
    print(LINE)
    final = agent.context.render()
    user_msgs = [m for m in final if m.role == "user"]
    print(f"三轮请求，但 role='user' 的消息始终只有 {len(user_msgs)} 条：")
    for m in user_msgs:
        print(f"    「{m.content}」")
    print()
    print("第 2、3 轮没有「新问题」。推动模型继续的是 role='tool' 的工具结果。")
    print("下一个用户问题，要等你再调一次 agent.run() 才会出现。")

    # ------------------------------------------------------------------
    print()
    print(LINE)
    print("context 里最终存了什么（按 role 分类）")
    print(LINE)
    for role, desc in [
        ("user", "你问的问题"),
        ("assistant", "模型说的话 + 它发起的工具调用"),
        ("tool", "工具执行结果 ← 最容易被忽略，而且通常最占地方"),
    ]:
        items = [m for m in final if m.role == role]
        chars = sum(len(m.content) for m in items)
        print(f"  {role:<10} {len(items)} 条，{chars:>5} 字符   {desc}")

    total = sum(len(m.content) for m in final)
    tool_chars = sum(len(m.content) for m in final if m.role == "tool")
    print(f"\n  工具结果占了 {tool_chars / total:.0%} 的篇幅。")
    print("  真实场景里这个比例只会更高 —— 一条 SELECT 返回 100 行就几千字符。")
    print("  **这就是为什么上下文压缩迟早要做，也是为什么工具结果要截断。**")

    print()
    print(LINE)
    print(f"最终回答：{answer}")
    print(LINE)


if __name__ == "__main__":
    main()
