"""工具框架：Tool、ToolOutput、ToolRegistry。具体工具在 tools/ 下。

    · 参数用 pydantic 声明，JSON Schema 自动生成，模型传回的参数先过校验
    · 结果有两个读者：content 发给模型，details 只给界面（学 pi）
    · 工具失败就抛异常，execute() 变成 is_error 的结果喂回模型，不中断 Agent
    · 依赖在 __init__ 注入，不用全局变量
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Iterable, Iterator

from pydantic import BaseModel, ValidationError

from .messages import Image

MAX_OUTPUT_CHARS = 6000  # 单个工具结果的上限，防止一条结果吃掉半个上下文


@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str
    # 一句话摘要，结果被清理后留在占位里当线索
    summary: str = ""
    # 只给界面、不发给模型。截断只截 content
    details: Any = None
    is_error: bool = False
    # 发给模型的图片（view_image）。不算进 MAX_OUTPUT_CHARS
    images: tuple[Image, ...] = ()

    @staticmethod
    def error(text: str) -> "ToolOutput":
        return ToolOutput(text, is_error=True)

    # TODO 大结果落盘（README「下一步扩展」）：结果不能重拿的工具来了，全文存进
    #   会话目录/tool-results/，content 给开头一段 + 路径，再配一个按位置读的工具
    def capped(self, limit: int = MAX_OUTPUT_CHARS) -> "ToolOutput":
        """兜底：按整行截到 limit 以内（从一行中间截断，模型会把残缺的数当真）。"""
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
    """工具基类：name、description（本质上是提示词）、Args、run()。

    rerunnable：只读、没副作用、同样参数给同样结果。只有这种工具的旧结果会被上下文清理。
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    rerunnable: ClassVar[bool] = False

    def schema(self) -> dict[str, Any]:
        """中立格式的工具描述，provider 再翻译成自家格式。"""
        params = self.Args.model_json_schema()
        params.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }

    @abstractmethod
    def run(self, args: Any) -> "str | ToolOutput":
        """args 是校验过的 Args 实例。失败就抛异常。返回字符串，或带 summary / details 的 ToolOutput。"""

    def execute(self, raw_args: dict[str, Any]) -> ToolOutput:
        """校验参数 → 执行 → 兜住异常。"""
        try:
            args = self.Args(**raw_args)
        except ValidationError as exc:
            # 校验错误原样告诉模型，它通常下一轮就能改对
            return ToolOutput.error(f"参数不合法：{exc}").capped()
        try:
            out = self.run(args)
            if not isinstance(out, ToolOutput):
                out = ToolOutput(str(out))
            return out.capped()
        except Exception as exc:  # noqa: BLE001 —— 故意兜住所有异常喂回模型
            return ToolOutput.error(f"{type(exc).__name__}: {exc}").capped()


# ================================================================ 注册表
class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"工具名重复：{tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """模型是无状态的，每次请求都要发完整工具表。"""
        return [t.schema() for t in self._tools.values()]

    def invoke(self, name: str, raw_args: dict[str, Any]) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            # 模型偶尔会编出不存在的工具名，告诉它有哪些
            return ToolOutput.error(f"不存在名为 '{name}' 的工具。可用工具：{', '.join(self._tools)}")
        return tool.execute(raw_args)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
