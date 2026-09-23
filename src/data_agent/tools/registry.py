"""ToolRegistry：工具的注册表 / 调度中心。

Agent 本身不持有任何具体工具，只持有一个 registry。加工具 = 注册一个对象，
主循环一行不用动 —— 这就是所谓的可扩展性。

以后可以在这里加：
    · 按场景动态裁剪工具列表（工具一多，全塞给模型反而会降智）
    · 调用次数 / 耗时统计
    · 并发执行多个 tool_call
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from .base import Tool, ToolOutput


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
