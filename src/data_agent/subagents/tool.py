"""delegate：把任务分派给子 Agent（学 Claude Code 的 Agent 工具、pi 的 subagent 扩展）。

子 Agent 是一个上下文全新的 Agent：看不到主对话，只拿到任务说明；中间过程不进主上下文，只交回最后的话。
和主 Agent 共用结果编号、数据库、知识库、审批；工具按类型给，没有 delegate、ask_user、remember。
一次分派的几个任务同时跑，各在自己的线程里；主 Agent 等全部做完再继续。
沙箱每个子 Agent 各起一个（变量互不干扰），往 figures/ 存的文件落在 figures/<任务号>/，同名文件不会互相覆盖。

事件包成 SubagentEvent 经主 Agent 发出去，一次只发一个（加锁）。停止：订阅者在某个事件上抛了异常
（界面的 Stopped、Ctrl-C），其余子 Agent 在它们的下一个事件上也停，正在跑的工具取消，等都停下再把原来的异常抛给主 Agent。
子 Agent 花的 token 记进主 Agent 的 session_usage（钱是这个会话花的，评测、界面只看这一个数）。
"""

from __future__ import annotations

import contextlib
import re
import threading
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.events import Event, SubagentEvent, TurnEnded
from ..core.messages import Usage
from ..core.tools import Tool, ToolOutput
from ..session.store import save_transcript
from ..tools.sql.results import ResultStore
from .catalog import Definition

MAX_TASKS = 4
ANSWER_CHARS = 4000            # 每个任务交回的话最多这么长，多个任务加起来也不会撑爆主上下文
POLL_S = 0.2                   # 主线程等子任务时隔多久醒一次：Windows 上一直阻塞着收不到 Ctrl-C
TASK_FILE = re.compile(r"t(\d+)\.jsonl")
# 交回给主 Agent 的每一段的开头。界面重新打开会话时靠它找到每个任务的过程（web/serialize.py）
REPORT_HEAD = re.compile(r"^## (t\d+) · ([\w-]+) · (.*)：(完成|失败)", re.M)

DESCRIPTION = """\
把任务分派给子 Agent。子 Agent 从全新的上下文开始：看不到这段对话，只看得到你写的任务说明；
它的中间过程不进你的上下文，只交回最后的结论。一次给几个任务会同时跑。

适合：要翻很多表、文档才能摸清的查探（你只需要结论）；彼此独立、能分开做的子问题。
不适合：一两步就能做完的事（分派本身有开销）；要和用户确认的事（子 Agent 不能问用户）；
后一步要用前一步结果的（分两次分派，或者自己做）。
几个任务之间不要重叠：划清各自负责哪一块，同一件事不要派两次（花两遍钱，还会交回两份对不上的结果）。

写任务说明（prompt）：子 Agent 什么都不知道，写清楚目标、已知的表和字段、口径、要交回什么，
不要写「按上面说的做」。
子 Agent 查出来的结果编号（r7 这种）你可以直接引用、load_result；它存的文件以交回的清单为准
（在 figures/<任务号>/ 下）。几个任务的结论要自己核对：口径不一致、数字对不上时说明原因或再查，
不要随便挑一个用。

可用的子 Agent：
{agents}"""


@dataclass(slots=True)
class Child:
    """spawn 造出来的一个子 Agent，和它自己的东西（沙箱）。"""

    agent: Agent
    close: Callable[[], None] = lambda: None
    files: Callable[[], list[str]] = list        # 它产出的文件（相对工作目录）


# 按类型造一个子 Agent：(类型, 任务号, 事件交给谁)
Spawn = Callable[[Definition, str, Callable[[Event], None]], Child]


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
    files: list[str] = field(default_factory=list)     # 这个任务存的文件（相对工作目录）

    def render(self) -> str:
        cost = f"{self.steps} 步，输入 {self.usage.prompt_tokens:,} / 输出 {self.usage.output:,} token"
        head = f"## {self.task} · {self.agent} · {self.title}：" + (f"完成（{cost}）" if self.ok else f"失败（{cost}）")
        answer = self.answer.strip()
        if len(answer) > ANSWER_CHARS:
            answer = answer[:ANSWER_CHARS] + f"\n…（子 Agent 交回的话太长，只保留前 {ANSWER_CHARS} 字）"
        parts = [head, answer if self.ok else f"出错了：{self.error}"]
        if self.refs:
            parts.append("新产生的结果：" + "、".join(self.refs))
        if self.files:
            parts.append("存的文件：" + "、".join(self.files))
        return "\n".join(p for p in parts if p)


class _Aborted(BaseException):
    """兄弟任务被停下了，这个任务也在下一个事件上停。只在 delegate 里面用，主 Agent 收到的是原来那个异常。"""


class DelegateTool(Tool):
    name = "delegate"
    description = ""
    max_output_chars = MAX_TASKS * (ANSWER_CHARS + 800)

    class Args(BaseModel):
        tasks: list[Task] = Field(min_length=1, max_length=MAX_TASKS, description=f"要分派的任务，1 到 {MAX_TASKS} 个")

    def __init__(self, definitions: list[Definition], spawn: Spawn, results: ResultStore,
                 transcripts: Path | None = None) -> None:
        self.definitions = {d.name: d for d in definitions}
        self.spawn = spawn
        self.results = results
        self.description = DESCRIPTION.format(
            agents="\n".join(f"- {d.name}：{d.description}" for d in definitions))
        # 组装时绑上主 Agent（bind）：子 Agent 的事件经它发出去，花的 token 记在它账上
        self.parent: Agent | None = None
        self._lock = threading.Lock()                # 事件一次发一个；改下面几个字段
        self._running: dict[str, Child] = {}
        self._abort: BaseException | None = None     # 这次分派被停下的原因（第一个异常）
        # 每个子任务的过程存在这里（会话目录/subagents/，没有就不存：评测）
        self.transcripts = transcripts
        # 任务号在会话里一直往下编：两次分派的任务不会混，重新打开会话也不会盖掉以前的过程和 figures/t1/
        self._count = max((int(m.group(1)) for p in transcripts.glob("t*.jsonl")
                           if (m := TASK_FILE.fullmatch(p.name))), default=0) if transcripts else 0

    def bind(self, parent: Agent) -> None:
        self.parent = parent

    def cancel(self) -> None:
        with self._lock:
            children = list(self._running.values())
        for child in children:
            child.agent.cancel_tool()

    def run(self, args: Args) -> ToolOutput:
        unknown = sorted({t.agent for t in args.tasks} - set(self.definitions))
        if unknown:
            raise ValueError(f"没有子 Agent 类型 {'、'.join(unknown)}。可用的：{'、'.join(self.definitions)}")
        # 一模一样的任务派两次只会花两遍钱（同一个模型错也错得一样，当核对也没用）。部分重叠认不出来，靠说明约束
        seen: dict[str, int] = {}
        for i, task in enumerate(args.tasks, 1):
            if (first := seen.setdefault(" ".join(task.prompt.split()), i)) != i:
                raise ValueError(f"第 {first} 个和第 {i} 个任务的说明一模一样。同一件事只派一次；"
                                 "想让它们分别做不同的部分，就把各自负责哪一块写清楚。")
        with self._lock:
            self._abort = None
            ids = [f"t{self._count + i}" for i in range(1, len(args.tasks) + 1)]
            self._count += len(args.tasks)
        with ThreadPoolExecutor(max_workers=len(args.tasks), thread_name_prefix="subagent") as pool:
            futures = [pool.submit(self._run_one, tid, task) for tid, task in zip(ids, args.tasks)]
            self._wait(futures)
        reports = [f.result() for f in futures]
        done = sum(r.ok for r in reports)
        return ToolOutput("\n\n".join(r.render() for r in reports),
                          summary=f"分派了 {len(reports)} 个任务，{done} 个完成",
                          details=reports, is_error=done == 0)

    def _wait(self, futures: list[Future]) -> None:
        """等全部做完。有一个被停下（或者主线程自己被 Ctrl-C 打断）就让其余的也停，等它们停稳了再抛。"""
        try:
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=POLL_S, return_when=FIRST_EXCEPTION)
                for f in finished:
                    if (exc := f.exception()) is not None:
                        raise exc
        except BaseException as exc:
            self._stop_all(exc)
            while not all(f.done() for f in futures):
                wait(futures, timeout=POLL_S)
            raise self._abort or exc from None

    def _stop_all(self, exc: BaseException) -> None:
        with self._lock:
            if self._abort is None and not isinstance(exc, _Aborted):
                self._abort = exc
        self.cancel()

    def _save(self, tid: str, header: dict, entries: list | None = None) -> None:
        if self.transcripts is not None:
            with contextlib.suppress(OSError):        # 存盘是给界面回看的，失败了不影响交回结果
                save_transcript(self.transcripts / f"{tid}.jsonl", header, entries or [])

    def _emit(self, event: SubagentEvent) -> None:
        with self._lock:
            # 一轮结束的事件照发：界面要知道这个子任务停了
            if self._abort is not None and not isinstance(event.event, TurnEnded):
                raise _Aborted()
            if self.parent is not None:
                self.parent.emit(event)

    def _run_one(self, tid: str, task: Task) -> TaskReport:
        try:
            child = self.spawn(self.definitions[task.agent], tid,
                               lambda e: self._emit(SubagentEvent(tid, task.agent, task.title, e)))
        except Exception as exc:  # noqa: BLE001
            return TaskReport(tid, task.agent, task.title, ok=False, error=f"子 Agent 没能创建：{type(exc).__name__}: {exc}")
        with self._lock:
            self._running[tid] = child
        header = {"task": tid, "agent": task.agent, "title": task.title, "prompt": task.prompt}
        self._save(tid, header)                       # 先占住任务号：进程中途被杀，下次也不会重用
        try:
            with self.results.origin(tid):
                answer, error = child.agent.run(task.prompt), ""
        except Exception as exc:  # noqa: BLE001 —— 一个任务失败不连累别的；停止（BaseException）照常往上抛
            answer, error = "", f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            self._stop_all(exc)
            raise
        finally:
            with self._lock:
                self._running.pop(tid, None)
                if self.parent is not None:          # 被停下来也要记：停之前的请求已经花了
                    self.parent.session_usage += child.agent.session_usage
            child.close()
            agent = child.agent
            self._save(tid, header, [*agent.context.history, *(agent.interrupted.entries if agent.interrupted else ())])
        return TaskReport(tid, task.agent, task.title, ok=not error, answer=answer, error=error,
                          steps=child.agent.state.step, usage=child.agent.session_usage,
                          refs=self.results.made_by(tid), files=child.files())
