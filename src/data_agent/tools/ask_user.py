"""ask_user：一轮做到一半问用户一个问题，拿到回答接着做（学 Claude Code 的 AskUserQuestion）。

不等回答、不碰界面：抛 NeedsUserInput，Agent 把这一轮暂停、存检查点，界面拿到问题去问，
回答用 agent.resume(回答) 接回来，成为这次调用的结果。命令行、评测、以后的 Web 各自决定怎么问。
不是 rerunnable：回答没法重拿，清理掉就丢了。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.tools import NeedsUserInput, Tool

MAX_OPTIONS = 4


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "问用户一个问题，这一轮停下来等回答；回答作为这次调用的结果返回，你接着做。\n"
        "该问：缺的信息会明显改变结果，又没法从数据、项目约定、长期记忆里查到——"
        "比如一个说法有几种合理的理解、算出来差很多；用户这次说的和记忆或项目约定矛盾，不知道以哪个为准。\n"
        "不该问：能自己查到的先查；不影响结论的小事自己定，在回答里说明假设；用户已经说清楚的不要再确认。\n"
        "一次只问一个问题，说清楚为什么要问；有几个明确的选项就放进 options，用户也可以不选、自己写。"
    )

    class Args(BaseModel):
        question: str = Field(min_length=1, description="要问的问题，带上为什么要问（几种理解差在哪）")
        options: list[str] = Field(default_factory=list, max_length=MAX_OPTIONS, description=(
            f"可选：最多 {MAX_OPTIONS} 个选项，每个一句话写清楚区别，比如「按实付金额（扣折扣）」"))

    def run(self, args: Args) -> str:
        options = tuple(o.strip() for o in args.options if o.strip())
        raise NeedsUserInput(args.question.strip(), options)
