"""运行器：一道题跑一次 = 一个全新的 Agent，不开界面，收集全部事件，然后判分。

为什么每次都新建：上一题的对话留在上下文里，下一题可能直接抄答案。
为什么能不开界面：Agent 只往外抛事件（core/events.py），CLI 拿去打印，
                  这里拿去存档 —— Agent 的代码一行不用改。

多轮会话（run_session）反过来：一段会话**共用**一个 Agent，每一轮单独记一个 Trial、单独判分。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent.app import build_application
from data_agent.core.events import (
    ContextEdited,
    Event,
    LLMResponded,
    StepLimitReached,
    ToolFinished,
    ToolStarted,
    collect_sink,
)
from data_agent.core.messages import Message, Usage
from data_agent.db.connection import Database
from data_agent.settings import Settings

from .cases import Case, Session
from .graders import AnswerCheck, ResultMatch, check_answer, compare_results

# 判分时重跑 SQL 最多取多少行。标准答案不会有这么多行；Agent 的查询超过这个数，肯定不对。
GRADE_MAX_ROWS = 5000


@dataclass(frozen=True, slots=True)
class Gold:
    """一道题的标准答案，跑一次、所有 trial 共用。"""

    alternatives: list[list[tuple]]     # 每条 gold_sql 的结果
    answer: list[tuple] | None = None   # answer_sql 的结果：回答里必须说到的数


@dataclass(slots=True)
class SqlCall:
    sql: str
    ok: bool
    purpose: str = ""


@dataclass(slots=True)
class Trial:
    """一道题跑一次的全部结果。存进 trials.jsonl，失败分析就看它。"""

    case_id: str
    trial: int
    answer: str = ""
    error: str = ""                       # 运行时抛了异常（API 错、截断……）
    sql_calls: list[SqlCall] = field(default_factory=list)
    final_sql: str = ""                   # 最后一条执行成功的 run_sql（报告里展示用）
    matched_sql: str = ""                 # 查出了标准答案的那条；空 = 没有一条对上
    steps: int = 0                        # 调了几次模型
    usage: Usage = field(default_factory=Usage)
    calls: list[Usage] = field(default_factory=list)   # 每次调模型的用量，按顺序（不含写摘要）
    after_edit: list[int] = field(default_factory=list)  # calls 里哪几次紧跟在清理/压缩之后
    after_compact: list[int] = field(default_factory=list)  # 其中哪几次之前做过压缩（after_edit 的子集）
    elapsed_s: float = 0.0
    step_limit: bool = False
    context_edits: int = 0
    result: ResultMatch | None = None     # SQL 结果比对；None = 没有可比的 SQL
    answer_check: AnswerCheck | None = None
    grade_error: str = ""                 # 重跑 Agent 的 SQL 时出错
    transcript: list[dict[str, Any]] = field(default_factory=list)
    graded: bool = True                   # 多轮会话里的填充轮不判分

    # ------------------------------------------------------------ 结论
    @property
    def result_ok(self) -> bool:
        """结果对了：SQL 查出来的和标准答案一致（宽松：允许多几列）。"""
        return bool(self.result and self.result.lenient)

    @property
    def answer_ok(self) -> bool:
        """回答也对了：结果对，而且回答里的数字对得上（没有数字可核对的题只看结果）。"""
        return self.result_ok and (self.answer_check is None or self.answer_check.ok is not False)

    @property
    def failure(self) -> str:
        """失败归类。报告里按它统计，比一个总分更能告诉你下一步该改哪里。"""
        if not self.graded:
            return ""
        if self.error:
            return "运行出错"
        if self.step_limit:
            return "步数耗尽"
        if self.result is None:
            return "没有执行成功的 SQL"
        if not self.result.lenient:
            return "结果不对"
        if not self.answer_ok:
            return "SQL 对了但回答里的数字不对"
        return ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(result_ok=self.result_ok, answer_ok=self.answer_ok, failure=self.failure)
        return d


# ================================================================ 跑一次
def run_trial(case: Case, trial: int, settings: Settings, db: Database,
              gold: Gold) -> Trial:
    """让 Agent 回答一道题，然后判分。任何异常都记进结果，不往外抛 —— 一题出错不能拖垮整批。"""
    events: list[Event] = []
    t = Trial(case.id, trial)
    started = time.perf_counter()
    try:
        app = build_application(settings, on_event=collect_sink(events))
        try:
            t.answer = app.agent.run(case.question)
        finally:
            t.usage = app.agent.session_usage
            t.transcript = _transcript(app.agent.context.history)
    except Exception as exc:  # noqa: BLE001
        t.error = f"{type(exc).__name__}: {exc}"
        t.transcript.append({"error": traceback.format_exc(limit=5)})
    t.elapsed_s = round(time.perf_counter() - started, 1)
    digest(t, events)
    grade(t, case, db, gold)
    return t


def digest(t: Trial, events: list[Event]) -> None:
    """从一轮的事件流里取出判分和统计要用的东西，写回 t。"""
    t.sql_calls = extract_sql_calls(events)
    edited = compacted = False
    for e in events:
        if isinstance(e, ContextEdited):
            t.context_edits += 1
            edited = True
            compacted |= e.kind == "HistoryCompacted"
        elif isinstance(e, LLMResponded):
            if edited:
                t.after_edit.append(len(t.calls))
            if compacted:
                t.after_compact.append(len(t.calls))
            edited = compacted = False
            t.calls.append(e.usage)
    t.steps = len(t.calls)
    t.step_limit = any(isinstance(e, StepLimitReached) for e in events)
    succeeded = [c for c in t.sql_calls if c.ok]
    t.final_sql = succeeded[-1].sql if succeeded else ""


# ================================================================ 多轮会话
@dataclass(slots=True)
class SessionTrial:
    """一段会话跑一次。每一轮是一个 Trial（判分和单题一样），会话级的东西记在这里。"""

    session_id: str
    trial: int
    turns: list[Trial] = field(default_factory=list)
    edits: list[dict[str, Any]] = field(default_factory=list)   # 每次清理/压缩：哪一轮、哪种、前后多大
    error: str = ""                       # 连 Agent 都没建起来
    elapsed_s: float = 0.0
    transcript: list[dict[str, Any]] = field(default_factory=list)

    # 给 report.cache_stats 用：整段会话连起来看，「首次调用」是整段会话的第一次
    @property
    def usage(self) -> Usage:
        return sum((t.usage for t in self.turns), Usage())

    @property
    def calls(self) -> list[Usage]:
        return [u for t in self.turns for u in t.calls]

    @property
    def after_edit(self) -> list[int]:
        return self._offsets("after_edit")

    @property
    def after_compact(self) -> list[int]:
        return self._offsets("after_compact")

    def _offsets(self, attr: str) -> list[int]:
        """每一轮里的下标换算成整段会话里的下标。"""
        out, offset = [], 0
        for t in self.turns:
            out += [offset + i for i in getattr(t, attr)]
            offset += len(t.calls)
        return out

    def edit_count(self, kind: str) -> int:
        return sum(e["kind"] == kind for e in self.edits)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        return d


def run_session(session: Session, trial: int, settings: Settings, db: Database,
                golds: dict[str, Gold]) -> SessionTrial:
    """同一个 Agent 按顺序回答每一轮。某一轮出错（Agent.run 是事务，历史会回滚）就记下来接着问。"""
    st = SessionTrial(session.id, trial)
    events: list[Event] = []
    started = time.perf_counter()
    try:
        app = build_application(settings.model_copy(update=session.settings),
                                on_event=collect_sink(events))
    except Exception as exc:  # noqa: BLE001
        st.error = f"{type(exc).__name__}: {exc}"
        return st

    for n, case in enumerate(session.turns, 1):
        t = Trial(case.id, trial, graded=case.graded)
        start, before, t0 = len(events), app.agent.session_usage, time.perf_counter()
        try:
            t.answer = app.agent.run(case.question)
        except Exception as exc:  # noqa: BLE001
            t.error = f"{type(exc).__name__}: {exc}"
        t.elapsed_s = round(time.perf_counter() - t0, 1)
        t.usage = _minus(app.agent.session_usage, before)
        turn_events = events[start:]
        digest(t, turn_events)
        st.edits += [{"turn": n, "kind": e.kind, "before": e.tokens_before, "after": e.tokens_after}
                     for e in turn_events if isinstance(e, ContextEdited)]
        if case.graded:
            grade(t, case, db, golds[case.id])
        st.turns.append(t)

    st.elapsed_s = round(time.perf_counter() - started, 1)
    st.transcript = _transcript(app.agent.context.history)
    return st


def _minus(a: Usage, b: Usage) -> Usage:
    return Usage(a.input - b.input, a.output - b.output,
                 a.cache_read - b.cache_read, a.cache_write - b.cache_write)


def extract_sql_calls(events: list[Event]) -> list[SqlCall]:
    """从事件流里取出每次 run_sql：参数里的 SQL + 执行成没成功。

    工具是一个接一个执行的，所以 ToolStarted 后面紧跟的那个 ToolFinished 就是它的结果。
    """
    calls: list[SqlCall] = []
    pending: ToolStarted | None = None
    for e in events:
        if isinstance(e, ToolStarted):
            pending = e
        elif isinstance(e, ToolFinished) and pending is not None:
            if pending.name == "run_sql":
                args = pending.arguments
                calls.append(SqlCall(str(args.get("sql", "")), e.ok, str(args.get("purpose", ""))))
            pending = None
    return calls


def grade(t: Trial, case: Case, db, gold: Gold) -> None:
    """给一次 trial 判分，结果写回 t。db 只需要有 query(sql, max_rows=) 方法。"""
    if case.match == "answer":
        t.answer_check = check_answer(gold.answer or [], t.answer)
        ok = t.answer_check.ok is True
        t.result = ResultMatch(ok, ok, "" if ok else "回答里没说到标准答案的数")
        return
    if case.match == "empty":
        # 该查不到东西的题按回答判：说了「没有」就算对。Agent 常常先查数据覆盖哪几年
        # 来证明没有（冒烟测试里就是这样），这比硬跑一条返回空的 SQL 更好。
        says_none = any(w in t.answer for w in NO_DATA_WORDS)
        t.result = ResultMatch(says_none, says_none, "" if says_none else "回答里没说查不到数据")
        return

    # Agent 跑过的每一条成功的 SQL 都拿来比，有一条查出了标准答案就算对（去重，先比后面的）
    best: tuple[ResultMatch, int] | None = None
    for sql in dict.fromkeys(c.sql for c in reversed(t.sql_calls) if c.ok):
        rows = _rerun(t, db, sql)
        if rows is None:
            continue
        for i, g in enumerate(gold.alternatives):
            m = compare_results(g, rows, case.match)
            if best is None or _better(m, best[0]):
                best = (m, i)
                if m.lenient:
                    t.matched_sql = sql
        if best and best[0].strict:
            break
    if best is None:
        return
    t.result, alt = best
    t.answer_check = check_answer(gold.answer if gold.answer is not None else gold.alternatives[alt],
                                  t.answer)


NO_DATA_WORDS = ("没有", "无数据", "不存在", "为空", "暂无", "查不到")


def _better(a: ResultMatch, b: ResultMatch) -> bool:
    return (a.lenient, a.strict) > (b.lenient, b.strict)


def _rerun(t: Trial, db, sql: str) -> list[tuple] | None:
    """重跑 Agent 的 SQL 拿结构化结果。工具返回给模型的是截断过的文本表格，没法精确比。"""
    try:
        return db.query(sql, max_rows=GRADE_MAX_ROWS).rows
    except Exception as exc:  # noqa: BLE001
        t.grade_error = f"{type(exc).__name__}: {exc}"
        return None


def run_gold(case: Case, db: Database) -> Gold:
    """跑标准答案。每道题只跑一次，所有 trial 共用。"""
    return Gold(
        [db.query(sql, max_rows=GRADE_MAX_ROWS).rows for sql in case.gold_sql],
        db.query(case.answer_sql, max_rows=GRADE_MAX_ROWS).rows if case.answer_sql else None,
    )


def _transcript(history: list) -> list[dict[str, Any]]:
    """历史存成 JSON：消息 + 模型的思考（失败分析时最有用的就是它当时怎么想的）。"""
    out = []
    for e in history:
        if not isinstance(e, Message):
            out.append({"marker": type(e).__name__})
            continue
        item: dict[str, Any] = {"role": e.role, "content": e.content}
        if e.tool_calls:
            item["tool_calls"] = [{"name": c.name, "arguments": c.arguments} for c in e.tool_calls]
        if isinstance(e.raw, dict) and e.raw.get("reasoning_content"):
            item["reasoning"] = e.raw["reasoning_content"]
        out.append(item)
    return out
