"""实验：为什么工具必须每轮重发？历史里的工具调用痕迹算不算数？

    python docs/why_tools_resent.py

不需要数据库，只打 LLM 接口。

结论剧透：
    历史里有工具调用的**痕迹**（调过什么、参数是什么、结果是什么），
    但没有工具的**定义**（有哪些工具可用、参数怎么填）。
    不发 tools 参数，模型就无法发起新的工具调用 —— 哪怕它刚刚调过。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

from data_agent.core.messages import Message  # noqa: E402
from data_agent.settings import Settings, build_provider  # noqa: E402

LINE = "=" * 72

FAKE_TOOL = {
    "name": "list_tables",
    "description": "列出数据库里所有的表",
    "parameters": {"type": "object", "properties": {}},
}


def main() -> None:
    load_dotenv()
    provider = build_provider(Settings())
    system = "你是数据分析助手。需要了解库结构时调用工具。"

    # ---------------------------------------------------------------
    print(LINE)
    print("① 第一轮：带上 tools 参数")
    print(LINE)
    history = [Message.user("库里有哪些表？")]
    first = provider.chat(history, tools=[FAKE_TOOL], system=system)

    print(f"tool_calls : {[(c.name, c.arguments) for c in first.tool_calls]}")
    print(f"text       : {first.text!r}")
    if not first.tool_calls:
        print("\n（这次模型没调工具，实验不成立，再跑一次）")
        return

    # 把这一轮的调用和结果记进历史 —— 历史里现在**有工具调用的痕迹**了
    call = first.tool_calls[0]
    history.append(first.to_message())
    history.append(Message.tool_result(call.id, "shop.customers, shop.orders"))
    history.append(Message.user("再查一次，确认一下表的数量。"))

    print()
    print("历史里现在有这些东西：")
    for m in history:
        mark = f"tool_calls={[c.name for c in m.tool_calls]}" if m.tool_calls else ""
        print(f"  {m.role:<10} {m.content[:40]!r:<45} {mark}")

    # ---------------------------------------------------------------
    print()
    print(LINE)
    print("② 第二轮 A：照常带 tools —— 模型能继续调用")
    print(LINE)
    with_tools = provider.chat(history, tools=[FAKE_TOOL], system=system)
    print(f"tool_calls : {[c.name for c in with_tools.tool_calls]}")
    print(f"text       : {with_tools.text[:120]!r}")

    # ---------------------------------------------------------------
    print()
    print(LINE)
    print("③ 第二轮 B：同样的历史，但**不带** tools 参数")
    print(LINE)
    print("历史完全没变，里面依然有它上一轮调用 list_tables 的完整记录。")
    print("如果『模型记得给过什么工具』，它应该还能调。看看会怎样：")
    print()
    try:
        without_tools = provider.chat(history, tools=None, system=system)
        print(f"tool_calls : {[c.name for c in without_tools.tool_calls]}")
        print(f"text       : {without_tools.text[:200]!r}")
        verdict = "调不了" if not without_tools.tool_calls else "居然还能调（意外）"
    except Exception as exc:  # noqa: BLE001
        print(f"直接报错了：{type(exc).__name__}: {str(exc)[:200]}")
        verdict = "调不了（API 直接拒绝）"

    # ---------------------------------------------------------------
    print()
    print(LINE)
    print("结论")
    print(LINE)
    print(f"带 tools    → 能调用工具")
    print(f"不带 tools  → {verdict}")
    print()
    print("历史里有的是「我调过 list_tables，结果是 XXX」这条**记录**，")
    print("没有的是「list_tables 这个工具存在、它长这样、参数这么填」这份**定义**。")
    print()
    print("记录 ≠ 授权。所以 tools 必须每一轮都发。")
    print()
    print(LINE)
    print("⚠️ 顺带撞见的一个坑：工具调用会泄漏成文本")
    print(LINE)
    print("注意第 ③ 步的 text —— 模型其实从历史里认出了 list_tables，")
    print("**并且真的试图调用它**。但请求里没有 tools 参数，API 没有 schema")
    print("可以把它解析成结构化的 tool_calls，于是模型的内部调用格式")
    print("直接泄漏成了普通文本。")
    print()
    print("后果：")
    print("  · tool_calls 是空的  → Agent 循环判定「答完了」")
    print("  · 把那坨乱码当最终答案返回给用户")
    print("  · 全程不报错，stop_reason 也是正常的")
    print()
    print("这是一种**静默失败**。Anthropic 文档里也提到过同类现象：")
    print("模型可能把工具调用写进可见文本而不是 tool_use 块，")
    print("这一轮会成功返回，但调用永远不会执行，也不会有任何报错。")
    print()
    print("防御办法：工具表在一个会话里**始终保持一致**。")
    print("不要中途把工具拿掉 —— 历史里留着调用痕迹、当前又没有定义，")
    print("正是触发这个问题的配方。")


if __name__ == "__main__":
    main()
