"""remember / read_memory：写和读长期记忆。

不用 read_file 读记忆文件：read_file 要开沙箱才有，记忆在没有沙箱的环境里也得能读写。
remember 有副作用，不是 rerunnable（不能被上下文清理掉当作没发生过）；read_memory 可以，清掉了再读。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ..core.tools import Tool, ToolOutput
from ..memory import Memory

Scope = Literal["project", "user"]
_SCOPE = "project：这个项目的口径、定义、事实；user：跨项目通用的个人偏好（回答风格、单位）"
_NAME = "英文小写字母、数字、连字符，比如 active-customer。改一条记忆就用它原来的 name"


class RememberTool(Tool):
    name = "remember"
    description = (
        "写长期记忆（以后的会话还在）：记下用户定的口径和定义、个人偏好、对你做法的纠正，"
        "或者按用户的要求改掉、删掉一条。同一件事用同一个 name 覆盖。"
        "只对这一轮有效的条件、查出来算出来的数字、密码密钥不要记。"
    )

    class Args(BaseModel):
        action: Literal["save", "delete"]
        scope: Scope = Field(description=_SCOPE)
        name: str = Field(description=_NAME)
        description: str = Field(default="", description=(
            "save 时必填。一行摘要，以后每次会话都会出现在记忆目录里，要写出关键内容，"
            "比如「活跃客户 = 近 90 天内下过单的客户」"))
        content: str = Field(default="", description="正文：细节和来由（用户哪天怎么说的）。不写就只有摘要")

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Args) -> ToolOutput:
        store = self.memory.store(args.scope)
        key = f"{args.scope}/{args.name}"
        if args.action == "delete":
            store.delete(args.name)
            return ToolOutput(f"已删除记忆 {key}", summary=f"删除了记忆 {key}")
        if not args.description.strip():
            raise ValueError("save 要写 description：一行摘要，以后的会话靠它知道记了什么")
        created = store.save(args.name, args.description.strip(), args.content or args.description, date.today())
        return ToolOutput(f"已{'新建' if created else '更新'}记忆 {key}：{args.description.strip()}",
                          summary=f"{'新建' if created else '更新'}了记忆 {key}")


class ReadMemoryTool(Tool):
    name = "read_memory"
    description = "读一条长期记忆的全文。记忆目录里的一行摘要不够用时再读。"
    rerunnable = True

    class Args(BaseModel):
        scope: Scope = Field(description=_SCOPE)
        name: str = Field(description="记忆目录里列出的名字")

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Args) -> ToolOutput:
        text = self.memory.store(args.scope).read(args.name)
        return ToolOutput(text, summary=f"读了记忆 {args.scope}/{args.name}")
