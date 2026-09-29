"""load_skill：读一个技能的全文。系统提示词里只有技能的名字和一句描述，做对应的任务前用它读正文。

不用 read_file 读 SKILL.md（pi 的做法）：read_file 要开沙箱才有，技能在没有沙箱的环境里也该能读；
单独一个工具，评测也好统计「这一轮加载了哪些技能」。
rerunnable：旧结果被上下文清理后，线索里留着技能名，要用再加载一次。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ..core.tools import Tool, ToolOutput
from ..skills import Skill


class LoadSkillTool(Tool):
    name = "load_skill"
    description = (
        "读一个技能的全文：某类任务的做法、默认口径和注意事项。系统提示词「技能」一节列出了有哪些技能，"
        "任务对得上时先读它，再照着做。"
    )
    rerunnable = True

    class Args(BaseModel):
        name: str = Field(description="技能名，比如 meta-analysis")

    def __init__(self, skills: Sequence[Skill]) -> None:
        self.skills = {s.name: s for s in skills}

    def run(self, args: Args) -> ToolOutput:
        skill = self.skills.get(args.name.strip())
        if skill is None:
            raise LookupError(f"没有叫 {args.name} 的技能。有这些：{', '.join(self.skills)}")
        return ToolOutput(f"[技能 {skill.name}]\n{skill.body()}", summary=f"加载了技能 {skill.name}")
