"""工具基类 —— 定义「一个工具需要提供什么」。

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
from typing import Any, ClassVar

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
