"""delegate：把任务分派给子 Agent（学 Claude Code 的 Agent 工具、pi 的 subagent 扩展）。

子 Agent 是一个上下文全新的 Agent：看不到主对话，只拿到任务说明；中间过程不进主上下文，只交回最后的话。
和主 Agent 共用结果编号、work 目录、数据库、知识库、审批；工具按类型给，没有 delegate、ask_user、remember。
主 Agent 等所有任务做完再继续。事件包成 SubagentEvent 从主 Agent 发出去，停止照样生效；
子 Agent 花的 token 记进主 Agent 的 session_usage（钱是这个会话花的，评测、界面只看这一个数）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.events import Event, SubagentEvent
from ..core.messages import Usage
from ..core.tools import Tool, ToolOutput
from ..tools.sql.results import ResultStore
from .catalog import Definition

MAX_TASKS = 4
ANSWER_CHARS = 4000            # 每个任务交回的话最多这么长，多个任务加起来也不会撑爆主上下文

# 按类型造一个子 Agent，事件交给 on_event
Spawn = Callable[[Definition, Callable[[Event], None]], Agent]

DESCRIPTION = """\
把任务分派给子 Agent。子 Agent 从全新的上下文开始：看不到这段对话，只看得到你写的任务说明；
它的中间过程不进你的上下文，只交回最后的结论。

适合：要翻很多表、文档才能摸清的查探（你只需要结论）；彼此独立、能分开做的子问题。
不适合：一两步就能做完的事（分派本身有开销）；要和用户确认的事（子 Agent 不能问用户）。

写任务说明（prompt）：子 Agent 什么都不知道，写清楚目标、已知的表和字段、口径、要交回什么，
不要写「按上面说的做」。
子 Agent 查出来的结果编号（r7 这种）你可以直接引用、load_result。几个任务的结论要自己核对：
口径不一致、数字对不上时说明原因或再查，不要随便挑一个用。

可用的子 Agent：
{agents}"""


class Task(BaseModel):
    agent: str = Field(description="子 Agent 类型")
    title: str = Field(description="一句话标题，给用户看")
    prompt: str = Field(description="完整的任务说明：目标、已知信息、口径、要交回什么")


@dataclass(slots=True)
class TaskReport:
    """一个任务的结果。也是 ToolFinished.details 给界面的。"""

    task: str
    agent: str
    title: str
    ok: bool
    answer: str = ""
    error: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    refs: list[str] = field(default_factory=list)      # 这个任务新产生的结果编号

    def render(self) -> str:
        cost = f"{self.steps} 步，输入 {self.usage.prompt_tokens:,} / 输出 {self.usage.output:,} token"
        head = f"## {self.task} · {self.agent} · {self.title}：" + (f"完成（{cost}）" if self.ok else f"失败（{cost}）")
        answer = self.answer.strip()
        if len(answer) > ANSWER_CHARS:
            answer = answer[:ANSWER_CHARS] + f"\n…（子 Agent 交回的话太长，只保留前 {ANSWER_CHARS} 字）"
        parts = [head, answer if self.ok else f"出错了：{self.error}"]
        if self.refs:
            parts.append("新产生的结果：" + "、".join(self.refs))
        return "\n".join(p for p in parts if p)


class DelegateTool(Tool):
    name = "delegate"
    description = ""
    max_output_chars = MAX_TASKS * (ANSWER_CHARS + 500)

    class Args(BaseModel):
        tasks: list[Task] = Field(min_length=1, max_length=MAX_TASKS, description=f"要分派的任务，1 到 {MAX_TASKS} 个")

    def __init__(self, definitions: list[Definition], spawn: Spawn, results: ResultStore) -> None:
        self.definitions = {d.name: d for d in definitions}
        self.spawn = spawn
        self.results = results
        # 组装时绑上主 Agent（bind）：子 Agent 的事件经它发出去，花的 token 记在它账上
        self.parent: Agent | None = None
        self.description = DESCRIPTION.format(
            agents="\n".join(f"- {d.name}：{d.description}" for d in definitions))
        self._running: Agent | None = None
        self._count = 0            # 任务号在会话里一直往下编，界面不会把两次分派的任务混在一起

    def bind(self, parent: Agent) -> None:
        self.parent = parent

    def _emit(self, event: Event) -> None:
        if self.parent is not None:
            self.parent.emit(event)

    def cancel(self) -> None:
        agent = self._running
        if agent is not None:
            agent.cancel_tool()

    def run(self, args: Args) -> ToolOutput:
        unknown = sorted({t.agent for t in args.tasks} - set(self.definitions))
        if unknown:
            raise ValueError(f"没有子 Agent 类型 {'、'.join(unknown)}。可用的：{'、'.join(self.definitions)}")
        reports = [self._run_one(task) for task in args.tasks]
        done = sum(r.ok for r in reports)
        return ToolOutput("\n\n".join(r.render() for r in reports),
                          summary=f"分派了 {len(reports)} 个任务，{done} 个完成",
                          details=reports, is_error=done == 0)

    def _run_one(self, task: Task) -> TaskReport:
        self._count += 1
        tid = f"t{self._count}"
        definition = self.definitions[task.agent]
        agent = self.spawn(definition, lambda e: self._emit(SubagentEvent(tid, task.agent, task.title, e)))
        before = set(self.results.refs())
        self._running = agent
        try:
            answer, error = agent.run(task.prompt), ""
        except Exception as exc:  # noqa: BLE001 —— 一个任务失败不连累别的；停止（BaseException）照常往上抛
            answer, error = "", f"{type(exc).__name__}: {exc}"
        finally:
            self._running = None
            if self.parent is not None:        # 被停下来也要记：停之前的请求已经花了
                self.parent.session_usage += agent.session_usage
        return TaskReport(tid, task.agent, task.title, ok=not error, answer=answer, error=error,
                          steps=agent.state.step, usage=agent.session_usage,
                          refs=[r for r in self.results.refs() if r not in before])
