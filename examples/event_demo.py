"""事件 / 回调机制的最小可运行演示 —— 对照 src/data_agent 的 core/events.py 与 core/agent.py。

回答一个问题:cli.py 从来不去"查询"任何东西,为什么每次都能拿到事件?
答案:启动时它把一个函数(电话号码)交给了 Agent;
      Agent 每逢大事就【调用】这个函数。方向永远是 Agent → sink,不存在 CLI 主动获取。

零依赖,Python 3.10+ 直接运行:
    python examples/event_demo.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 事件定义:和 core/events.py 一样,就是普通 dataclass(≈ Java 的 record)
# ---------------------------------------------------------------------------


@dataclass
class LLMResponded:
    step: int
    text: str
    tool_calls: list[str]


@dataclass
class ToolStarted:
    name: str
    arguments: dict


@dataclass
class ToolFinished:
    name: str
    ok: bool
    content: str


Event = LLMResponded | ToolStarted | ToolFinished


# ---------------------------------------------------------------------------
# 【一】函数是对象(Java 里要包一层接口,Python 不用)
# ---------------------------------------------------------------------------

def part1_函数是对象() -> None:
    print("【一】函数本身就是对象,可以像值一样传来传去")
    print("-" * 56)

    def hello(name: str) -> str:
        return f"你好,{name}"

    print(f"  type(hello) = {type(hello).__name__}   ← 不是字符串,是个对象")

    a = hello        # 注意:没有括号!传的是函数对象本身,不是调用结果
    b = hello
    print(f"  a = hello; b = hello  →  a is b = {a is b}(同一个对象)")
    print(f"  现在才调用:a('世界') → {a('世界')}")
    print("  Java 对照:hello ≈ 一个只实现了单方法接口的实例,a/b 是两个引用")
    print()


# ---------------------------------------------------------------------------
# 【二】最小回调:干活的人,在关键时刻调用你给的函数
# ---------------------------------------------------------------------------

def part2_最小回调() -> None:
    print("【二】最小回调:cook() 干活,关键时刻拨打你给的函数")
    print("-" * 56)

    def make_sink(prefix: str, sleep_s: float = 0.0):
        """工厂:返回一个闭包。对应 cli.py 的 make_console_sink。"""
        def sink(message: str) -> None:
            if sleep_s:
                time.sleep(sleep_s)      # 故意睡一秒,观察主流程会不会等
            print(f"        [{prefix}] 收到电话:{message}")
        return sink

    class Cooker:
        """对应 Agent:构造时收下电话号码存成字段,之后反复拨打同一个号码。"""

        def __init__(self, on_event: Callable[[str], None]) -> None:
            self.on_event = on_event
            self.phone_id = id(on_event)
            print(f"  [组装] 电话号码交出去一次,sink 的 id = {self.phone_id}")

        def cook(self, dish: str) -> str:
            print(f"  [厨师] 开始做 {dish}")
            print(f"  [厨师] 我拨的还是构造时那个号码(id={self.phone_id})")

            t0 = time.perf_counter()
            self.on_event("开始")        # ← "发射事件"的全部真相:调用一次你的函数
            print(f"  [厨师] 通话耗时 {time.perf_counter() - t0:.2f}s"
                  " —— sink 睡了一秒,厨师也只能干等(同一个线程!)")

            self.on_event("完成")
            return f"{dish} 好了"

    result = Cooker(make_sink("Sink", sleep_s=1.0)).cook("红烧肉")
    print(f"  [厨师] 返回:{result}")
    print()


# ---------------------------------------------------------------------------
# 【三】微型 Agent:模型 → 工具 → 结果入史 → 再调模型,事件沿途拨打
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class FakeLLM:
    """照剧本走的假模型,对应 tests/test_agent_loop.py 的 ScriptedProvider。不联网。"""

    model = "fake-model"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script
        self.calls = 0

    def chat(self, messages: list, tools=None, system=None) -> LLMResponse:
        self.calls += 1
        print(f"    [模型] 第 {self.calls} 次被调用,收到历史 {len(messages)} 条"
              "(它什么都不记得,全靠你每次重发)")
        return self.script[min(self.calls, len(self.script)) - 1]


def echo(text: str) -> str:
    """对应 tools/sql/run_sql.py —— 模型没有手,真正干活的是这行 Python。"""
    return f"echo: {text}(共 {len(text)} 个字符)"


class MiniAgent:
    """对应 core/agent.py 的 run() + _execute(),去掉了审批/上下文等插槽。"""

    def __init__(self, llm: FakeLLM, tools: dict[str, Callable], on_event) -> None:
        self.llm = llm
        self.tools = tools
        self.on_event = on_event          # ← 电话号码,启动时交一次,终身使用
        self.history: list[dict] = []

    def run(self, user_input: str, max_steps: int = 4) -> str:
        self.history.append({"role": "user", "content": user_input})
        for step in range(1, max_steps + 1):
            resp = self.llm.chat(self.history)
            self.history.append({"role": "assistant", "content": resp.text,
                                 "tool_calls": resp.tool_calls})
            self._emit(LLMResponded(step, resp.text, [c.name for c in resp.tool_calls]))

            if not resp.tool_calls:                    # 纯文本 = 模型交卷
                print("    [Agent] 模型没再要工具 → run() 到此返回")
                return resp.text

            for call in resp.tool_calls:
                self._emit(ToolStarted(call.name, call.arguments))
                result = self.tools[call.name](**call.arguments)   # 真正干活
                self.history.append({"role": "tool", "content": result})
                self._emit(ToolFinished(call.name, True, result))

        return "步数用完还没交卷"

    def _emit(self, event: Event) -> None:
        print(f"    [Agent] → 拨号:调用你给的 sink({type(event).__name__})")
        self.on_event(event)               # ← 就是普通函数调用,没有总线/注册中心
        print("    [Agent] ← sink 返回,继续往下执行")


def console_sink(event: Event) -> None:
    """对应 cli.py 的 make_console_sink —— match-case 按类型分派并打印。"""
    match event:
        case LLMResponded(step=step, tool_calls=calls):
            print(f"        [Sink] LLM 第 {step} 步说话了,"
                  f"想要工具: {calls or '(无,要交卷)'}")
        case ToolStarted(name=name, arguments=args):
            print(f"        [Sink] 工具 {name} 开始,参数 {args}")
        case ToolFinished(name=name, ok=ok, content=content):
            print(f"        [Sink] 工具 {name} 结束,ok={ok},返回 {content!r}")
        case _:
            print(f"        [Sink] 未知事件 {event!r}")


def part3_微型Agent循环() -> None:
    print("【三】微型 Agent:模型→工具→结果入史→再调模型,事件沿途拨打")
    print("-" * 56)

    script = [
        LLMResponse("我先查一下第一句话的长度",
                    [ToolCall("c1", "echo", {"text": "体测数据"})]),
        LLMResponse("再核对一次",
                    [ToolCall("c2", "echo", {"text": "赣南师范大学"})]),
        LLMResponse("两轮工具结果都对上了。结论:机制通了。"),
    ]

    print("  你 > 演示一下事件机制\n")
    answer = MiniAgent(FakeLLM(script), {"echo": echo}, console_sink).run(
        "演示一下事件机制")
    print(f"  最终答案: {answer}\n")

    print("  --- 换一个消费者:sink 换成 bucket.append,一个字都不打印 ---")
    bucket: list[Event] = []
    MiniAgent(FakeLLM(script), {"echo": echo}, bucket.append).run("同款循环,静默收集")
    print(f"  收到 {len(bucket)} 个事件(测试就是这个原理:收集起来做断言):")
    for e in bucket:
        print(f"    · {e}")
    print()


# ---------------------------------------------------------------------------
# 【四】你输入一句话,让假模型对它调一次工具再回答
# ---------------------------------------------------------------------------

def part4_你来当用户() -> None:
    print("【四】你输入一句话,假模型会对它调一次 echo 工具,然后给结论")
    print("-" * 56)
    try:
        text = input("  你 > ").strip()
    except EOFError:
        text = "默认文本"
    if not text:
        text = "默认文本"

    script = [
        LLMResponse("我先查查这句话", [ToolCall("c1", "echo", {"text": text})]),
        LLMResponse(f"结论:你说的是「{text}」,共 {len(text)} 个字符,"
                    "echo 工具如实转告了我。"),
    ]
    answer = MiniAgent(FakeLLM(script), {"echo": echo}, console_sink).run(text)
    print(f"  最终答案: {answer}")


# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台防 GBK 编码崩溃
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 58)
    print("  事件 / 回调机制演示 —— Agent 怎么把发生的事告诉外界")
    print("=" * 58)
    part1_函数是对象()
    part2_最小回调()
    part3_微型Agent循环()
    part4_你来当用户()

    print()
    print("对照回真实项目:")
    print("  make_sink / console_sink  ↔  cli.py:33    make_console_sink")
    print("  on_event=sink 传进去      ↔  app.py:45   build_application(on_event=...)")
    print("  self.on_event(...) 拨号   ↔  agent.py:77/99/114  三处发射点")
    print("  bucket.append 当 sink     ↔  tests 的 collect_sink")


if __name__ == "__main__":
    main()
