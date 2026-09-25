"""工具框架：一个工具要提供什么（Tool、ToolOutput），以及 Agent 怎么找到它们（ToolRegistry）。

具体的工具不在这里，在 tools/ 下面（tools/sql/…）。框架属于 core：Agent 只认这里的接口，
不知道有哪些具体工具 —— 和 pi 一样，AgentTool 定义在 agent 核心包里，具体工具在 coding-agent 里。

几个设计要点：

1. 参数用 pydantic 模型声明，JSON Schema 自动生成。不用手写 schema，
   而且模型传回来的参数会先过一遍校验。

2. 一个工具结果有两个读者：模型和界面。content 发给模型，details 只给界面（完整的结果表、
   文件路径…），不发给模型 —— 学 pi 的 content / details。模型要的是「够推理的
   最少信息」，用户要的是完整数据，混在一份里，要么模型撑爆上下文，要么用户拿不到全貌。

3. 工具失败就抛异常，execute() 把它变成 is_error=True 的结果（pi 的约定也是
   「失败就 throw，别把错误编进 content」）。**工具出错不应该中断 Agent** ——
   把错误告诉模型，让它自己决定重试还是换路子。这是 Agent 能「自愈」的关键。
   is_error 会一路带进历史、发给模型（Anthropic 的 tool_result 有 is_error 字段）。

4. 工具描述自己，通用层不替它做假设。比如「结果能不能靠再调一次拿回来」只有工具知道
   （rerunnable），上下文清理只清这种工具的结果。

5. 工具的依赖（数据库连接等）在 __init__ 里注入，不用全局变量。
   这样写测试时塞个假的 Database 进去就行。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Iterable, Iterator

from pydantic import BaseModel, ValidationError

MAX_OUTPUT_CHARS = 6000  # 单个工具结果的上限，防止一条结果吃掉半个上下文


@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str
    # 一句话摘要。结果以后被上下文清理掉时，它留在占位里当线索（见 MessageMeta.summary）。
    # 工具不给也行，上下文会退回到「约 N 字符」。
    summary: str = ""
    # 只给界面、不发给模型的数据（见模块说明第 2 点）。截断只截 content，不动它。
    details: Any = None
    is_error: bool = False

    @staticmethod
    def error(text: str) -> "ToolOutput":
        return ToolOutput(text, is_error=True)

    # TODO 大结果落盘（见 README「下一步扩展」）：这里现在是**截掉**，后面的内容就丢了。
    #   能重拿的结果（run_sql 重查）无所谓；以后接网页、实时 API、Python 这类结果不能重拿的工具，
    #   要换成：全文存进 会话目录/tool-results/<调用id>.txt，content 给开头一段 + 路径，
    #   再配一个按位置读的工具（Claude Code、pi 都是这么做的）。需要先有会话目录。
    def capped(self, limit: int = MAX_OUTPUT_CHARS) -> "ToolOutput":
        """兜底：工具自己没控制好大小时，按整行截到 limit 以内。

        按整行截（学 pi 的 truncate.ts）：表格从一行中间截断，模型会读到一个残缺的数，
        还当真。只有第一行就超长时才截在行中间。
        """
        if len(self.content) <= limit:
            return self
        kept = self.content[:limit]
        newline = kept.rfind("\n")
        if newline > 0:
            kept = kept[:newline]
            shown, total = kept.count("\n") + 1, self.content.count("\n") + 1
            note = f"只显示了前 {shown} 行，共 {total} 行"
        else:
            note = f"只显示了前 {limit} 个字符，共 {len(self.content)} 个"
        return replace(self, content=f"{kept}\n…（输出过长，{note}。需要其余部分，请缩小范围后重新调用）")


class Tool(ABC):
    """所有工具的基类。

    子类必须提供：
        name        工具名，模型用它来指定调用哪个
        description 说明书 —— 模型唯一的判断依据，**本质上是提示词**
        Args        pydantic 模型，声明参数
        run()       真正干活的代码

    可选：
        rerunnable  结果能不能靠再调用一次拿回来：只读、没有副作用、同样的参数给同样的结果。
                    上下文清理只清这种工具的旧结果（占位里会叫模型「重新调用一次」）。
                    默认 False：有副作用的工具（导出文件、发消息）重跑一次就是再做一遍。
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    rerunnable: ClassVar[bool] = False

    def schema(self) -> dict[str, Any]:
        """生成中立格式的工具描述，由各 provider 再翻译成自家格式。"""
        params = self.Args.model_json_schema()
        params.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }

    @abstractmethod
    def run(self, args: Any) -> "str | ToolOutput":
        """真正干活。args 是已经校验过的 Args 实例。失败就抛异常。

        一般返回字符串就行。想顺带给一句摘要（结果被清理后留作线索），
        就返回 ToolOutput(内容, summary=摘要)；有给界面的完整数据放 details。
        """

    def execute(self, raw_args: dict[str, Any]) -> ToolOutput:
        """统一入口：校验参数 -> 执行 -> 兜住异常。"""
        try:
            args = self.Args(**raw_args)
        except ValidationError as exc:
            # 把校验错误原样告诉模型，它通常下一轮就能改对
            return ToolOutput.error(f"参数不合法：{exc}").capped()
        try:
            out = self.run(args)
            if not isinstance(out, ToolOutput):
                out = ToolOutput(str(out))
            return out.capped()
        except Exception as exc:  # noqa: BLE001 —— 故意兜住所有异常喂回模型
            return ToolOutput.error(f"{type(exc).__name__}: {exc}").capped()


# ================================================================ 注册表
# Agent 本身不持有任何具体工具，只持有一个 registry。加工具 = 注册一个对象，主循环一行不用动。
# 以后可以在这里加：按场景裁剪工具列表（工具一多，全塞给模型反而会降智）、调用统计、并发执行。
class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"工具名重复：{tool.name}")
        self._tools[tool.name] = tool
        return self                      # 支持链式调用

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """这一轮要告诉模型的工具表。

        注意：模型是**无状态**的，每一轮请求都要把完整工具表重新发一遍，
        它不会「记住」上次告诉过它有哪些工具。
        """
        return [t.schema() for t in self._tools.values()]

    def invoke(self, name: str, raw_args: dict[str, Any]) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            # 模型偶尔会幻觉出不存在的工具名。告诉它有哪些，别抛异常。
            return ToolOutput.error(f"不存在名为 '{name}' 的工具。可用工具：{', '.join(self._tools)}")
        return tool.execute(raw_args)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
