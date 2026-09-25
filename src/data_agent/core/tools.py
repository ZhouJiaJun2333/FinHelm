"""工具框架：一个工具要提供什么（Tool、ToolOutput），以及 Agent 怎么找到它们（ToolRegistry）。

具体的工具不在这里，在 tools/ 下面（tools/sql/…）。框架属于 core：Agent 只认这里的接口，
不知道有哪些具体工具 —— 和 pi 一样，AgentTool 定义在 agent 核心包里，具体工具在 coding-agent 里。

三个设计要点：

1. 参数用 pydantic 模型声明，JSON Schema 自动生成。不用手写 schema，
   而且模型传回来的参数会先过一遍校验。

2. 一个工具结果有两个读者：模型和界面。content 发给模型，details 只给界面（完整的结果表、
   文件路径…），不进历史、不发给模型 —— 学 pi 的 content / details。模型要的是「够推理的
   最少信息」，用户要的是完整数据，混在一份里，要么模型撑爆上下文，要么用户拿不到全貌。

3. run() 里抛的任何异常都会被 execute() 捕获，变成 ToolOutput(ok=False)。
   **工具出错不应该中断 Agent** —— 把错误告诉模型，让它自己决定重试还是换路子。
   这是 Agent 能「自愈」的关键。

4. 工具的依赖（数据库连接等）在 __init__ 里注入，不用全局变量。
   这样写测试时塞个假的 Database 进去就行。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Iterator

from pydantic import BaseModel, ValidationError

MAX_OUTPUT_CHARS = 6000  # 单个工具结果的上限，防止一条结果吃掉半个上下文


@dataclass(frozen=True, slots=True)
class ToolOutput:
    ok: bool
    content: str
    # 一句话摘要。结果以后被上下文清理掉时，它留在占位里当线索（见 Message.summary）。
    # 工具不给也行，上下文会退回到「约 N 字符」。
    summary: str = ""
    # 只给界面、不发给模型的数据（见模块说明第 2 点）。截断只截 content，不动它。
    details: Any = None

    # TODO 大结果落盘（见 README「下一步扩展」）：这里现在是**截掉**，后面的内容就丢了。
    #   能重拿的结果（run_sql 重查）无所谓；以后接网页、实时 API、Python 这类结果不能重拿的工具，
    #   要换成：全文存进 会话目录/tool-results/<调用id>.txt，content 给开头一段 + 路径，
    #   再配一个按位置读的工具（Claude Code、pi 都是这么做的）。需要先有会话目录。
    def capped(self, limit: int = MAX_OUTPUT_CHARS) -> "ToolOutput":
        if len(self.content) <= limit:
            return self
        omitted = len(self.content) - limit
        return ToolOutput(
            self.ok,
            self.content[:limit] + f"\n…（输出过长，已截断 {omitted} 字符，请缩小查询范围）",
            self.summary,
            self.details,
        )


class Tool(ABC):
    """所有工具的基类。

    子类必须提供：
        name        工具名，模型用它来指定调用哪个
        description 说明书 —— 模型唯一的判断依据，**本质上是提示词**
        Args        pydantic 模型，声明参数
        run()       真正干活的代码
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

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
        """真正干活。args 是已经校验过的 Args 实例。

        一般返回字符串就行。想顺带给一句摘要（结果被清理后留作线索），
        就返回 ToolOutput(True, 内容, summary=摘要)；有给界面的完整数据放 details。
        """

    def execute(self, raw_args: dict[str, Any]) -> ToolOutput:
        """统一入口：校验参数 -> 执行 -> 兜住异常。"""
        try:
            args = self.Args(**raw_args)
        except ValidationError as exc:
            # 把校验错误原样告诉模型，它通常下一轮就能改对
            return ToolOutput(False, f"参数不合法：{exc}").capped()
        try:
            out = self.run(args)
            if not isinstance(out, ToolOutput):
                out = ToolOutput(True, str(out))
            return out.capped()
        except Exception as exc:  # noqa: BLE001 —— 故意兜住所有异常喂回模型
            return ToolOutput(False, f"{type(exc).__name__}: {exc}").capped()


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
            return ToolOutput(
                False,
                f"不存在名为 '{name}' 的工具。可用工具：{', '.join(self._tools)}",
            )
        return tool.execute(raw_args)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
