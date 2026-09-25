"""运行器：一道题跑一次 = 一个全新的 Agent，不开界面，收集全部事件，然后判分。

为什么每次都新建：上一题的对话留在上下文里，下一题可能直接抄答案。
为什么能不开界面：Agent 只往外抛事件（core/events.py），CLI 拿去打印，
                  这里拿去存档 —— Agent 的代码一行不用改。

多轮会话（run_session）反过来：一段会话**共用**一个 Agent，每一轮单独记一个 Trial、单独判分。

提交轮（题库级 submit，BIRD 用）：答完之后评测再追问一句，让 Agent 交一条只含所问列的 SQL。
BIRD 官方一题只收一条 SQL、结果要完全一样；Agent 回答真人时会多给几列上下文、把数 ROUND 好看 ——
对人是更好的回答，对官方判分是错。适配评测格式的活放在评测里，Agent 本身不改。
提交轮不影响主分数：主分数只看回答那一轮的 SQL，和以前的运行照样能比。

上传文件的题（research）：每个 trial 一个自己的工作目录（留在运行目录的 work/ 下，失败时能看图），
先 /attach 再提问；按回答里的数、沙箱代码、有没有出图判分（grade_files）。
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data_agent.app import build_application
from data_agent.core.events import (
    ContextEdited,
    ContextEditFailed,
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
from data_agent.tools.sandbox import Execution
from data_agent.tools.sql.results import REF, ResultStore, markdown_table

from .cases import CASES_DIR, Case, Session
from .dabstep.scorer import question_scorer
from .graders import AnswerCheck, ResultMatch, check_answer, check_values, compare_results, said_scalar

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
class Submission:
    """提交轮交上来的那条 SQL，按 BIRD 官方的规则判：只和 BIRD 原版标准答案比、列数也要一样。"""

    sql: str = ""                         # 提交轮里最后一条执行成功的 SQL；空 = 这一轮没跑成 SQL
    strict: bool = False
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str = ""                       # 提交轮出错只影响提交分，不算这题错


@dataclass(slots=True)
class Trial:
    """一道题跑一次的全部结果。存进 trials.jsonl，失败分析就看它。"""

    case_id: str
    trial: int
    answer: str = ""                      # 模型写的原文，{{r3}} 这类引用没展开
    shown: str = ""                       # 用户看到的（引用展开成整张表）；没有引用时为空
    refs: int = 0                         # 回答里引用了几次结果
    error: str = ""                       # 运行时抛了异常（API 错、截断……）
    sql_calls: list[SqlCall] = field(default_factory=list)
    final_sql: str = ""                   # 最后一条执行成功的 run_sql（报告里展示用）
    matched_sql: str = ""                 # 查出了标准答案的那条；空 = 没有一条对上
    # 只看最后一条 SQL、列数也一样 —— BIRD 官方的判法。我们的主分数看「任何一条」，
    # 两个都报：和公开榜单比用这个，和自己的旧版本比用主分数
    final_strict: bool = False
    submission: Submission | None = None  # 题库要求提交轮时才有
    # SQL 没对上，但标准答案是单个数、回答里算对了（graders.said_scalar）。算回答对，报告里单独列
    text_ok: bool = False
    steps: int = 0                        # 调了几次模型
    usage: Usage = field(default_factory=Usage)
    calls: list[Usage] = field(default_factory=list)   # 每次调模型的用量，按顺序（不含写摘要）
    after_edit: list[int] = field(default_factory=list)  # calls 里哪几次紧跟在清理/压缩之后
    after_compact: list[int] = field(default_factory=list)  # 其中哪几次之前做过压缩（after_edit 的子集）
    summary_calls: list[Usage] = field(default_factory=list)  # 写摘要那几次调用（不在 calls 里）
    elapsed_s: float = 0.0
    step_limit: bool = False
    context_edits: int = 0
    compaction_failures: int = 0          # 自动压缩没做成（这一步照常跑）
    result: ResultMatch | None = None     # SQL 结果比对；None = 没有可比的 SQL
    answer_check: AnswerCheck | None = None
    grade_error: str = ""                 # 重跑 Agent 的 SQL 时出错
    transcript: list[dict[str, Any]] = field(default_factory=list)
    graded: bool = True                   # 多轮会话里的填充轮不判分
    # 上传文件的题（research）
    no_sql: bool = False
    code: list[str] = field(default_factory=list)        # run_python / run_r 执行过的代码
    figures: list[str] = field(default_factory=list)     # 出过的图（相对工作目录）
    # 自己写代码画的图（不是 fh_ 模板画的）有几次，其中几次之后调了 view_image
    custom_plots: int = 0
    viewed_after: int = 0
    final_answer: str = ""                # DABstep：回答最后「最终答案：」那一行，提交文件用它
    official: bool = False                # DABstep 的题：只按「最终答案」判

    # ------------------------------------------------------------ 结论
    @property
    def result_ok(self) -> bool:
        """结果对了：SQL 查出来的和标准答案一致（宽松：允许多几列）。"""
        return bool(self.result and self.result.lenient)

    @property
    def answer_ok(self) -> bool:
        """回答也对了：结果对，而且回答里的数字对得上（没有数字可核对的题只看结果）；
        或者 SQL 没对上、但回答里把标准答案那个数算对了。
        步数耗尽、出错的不算：只查「有没有出图」的题，图画出来了、回答却是兜底的那句话。"""
        if self.error or self.step_limit:
            return False
        return (self.result_ok and (self.answer_check is None or self.answer_check.ok is not False)
                or self.text_ok)

    @property
    def official_sql(self) -> str:
        """交给 BIRD 官方判分的那条：有提交轮用提交的，提交轮没跑出 SQL 或者没有提交轮就用最后一条。"""
        return (self.submission and self.submission.sql) or self.final_sql

    @property
    def failure(self) -> str:
        """失败归类。报告里按它统计，比一个总分更能告诉你下一步该改哪里。"""
        if not self.graded:
            return ""
        if self.error:
            return "运行出错"
        if self.step_limit:
            return "步数耗尽"
        if self.text_ok:
            return ""
        if self.official:
            return "" if self.answer_ok else "最终答案不对"
        if self.no_sql:
            if not self.result_ok:
                return "要求的步骤没做"
            return "" if self.answer_ok else "回答里的数字不对"
        if self.result is None:
            return "没有执行成功的 SQL"
        if not self.result.lenient:
            return "结果不对"
        if not self.answer_ok:
            return "SQL 对了但回答里的数字不对"
        return ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(result_ok=self.result_ok, answer_ok=self.answer_ok, failure=self.failure,
                 official_sql=self.official_sql)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        """to_dict 的反操作，给 --rebuild 用。to_dict 多写的几个结论字段（result_ok 等）丢掉，现算。"""
        if "uses_files" in d:                    # 2026-09-26 改名前的运行记录
            d["no_sql"] = d.pop("uses_files")
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["sql_calls"] = [SqlCall(**c) for c in d.get("sql_calls", [])]
        d["usage"] = Usage(**d["usage"])
        d["calls"] = [Usage(**u) for u in d.get("calls", [])]
        d["summary_calls"] = [Usage(**u) for u in d.get("summary_calls", [])]
        if d.get("result"):
            d["result"] = ResultMatch(**d["result"])
        if d.get("answer_check"):
            d["answer_check"] = AnswerCheck(**d["answer_check"])
        if d.get("submission"):
            d["submission"] = Submission(**{**d["submission"], "usage": Usage(**d["submission"]["usage"])})
        return cls(**d)


# ================================================================ 跑一次
def run_trial(case: Case, trial: int, settings: Settings, db: Database | None,
              gold: Gold, submit: str = "", work_root: Path | None = None) -> Trial:
    """让 Agent 回答一道题（题库要求的话再加一轮提交），然后判分。

    任何异常都记进结果，不往外抛 —— 一题出错不能拖垮整批。
    work_root：没有 SQL 的题每个 trial 在它下面建自己的工作目录（并发的 trial 不能共用 figures/）。
    """
    events: list[Event] = []
    t = Trial(case.id, trial, no_sql=case.no_sql, graded=case.graded,
              official=case.official_answer is not None or case.answer_hidden)
    results = ResultStore()
    started = time.perf_counter()
    app = None
    work_dir = (work_root / f"{case.id}-{trial}") if case.no_sql and work_root else None
    try:
        app = build_application(settings, on_event=collect_sink(events), results=results, work_dir=work_dir)
        question = case.question
        if case.files:
            app.attach([CASES_DIR / f for f in case.files])
            question = app.with_uploads(question)
        try:
            t.answer = app.agent.run(question)
        finally:
            t.usage = app.agent.session_usage
            t.transcript = _transcript(app.agent.context.history)
    except Exception as exc:  # noqa: BLE001
        t.error = f"{type(exc).__name__}: {exc}"
        t.transcript.append({"error": traceback.format_exc(limit=5)})
    t.elapsed_s = round(time.perf_counter() - started, 1)
    answered = list(events)               # 回答那一轮的事件；主分数、步数、token 只看这些
    if submit and not t.error:
        t.submission = submit_sql(app, submit, events)
        t.transcript = _transcript(app.agent.context.history)   # 带上提交轮，失败分析要看
    if app is not None:
        app.close()                       # 沙箱是个容器
    digest(t, answered)
    show(t, results)
    grade(t, case, db, gold)
    return t


def submit_sql(app, prompt: str, events: list[Event]) -> Submission:
    """追问一句，收下 Agent 这一轮里最后一条执行成功的 SQL。"""
    s = Submission()
    before, n = app.agent.session_usage, len(events)
    try:
        app.agent.run(prompt)
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    s.usage = _minus(app.agent.session_usage, before)
    s.steps = sum(isinstance(e, LLMResponded) for e in events[n:])
    ok = [c.sql for c in extract_sql_calls(events[n:]) if c.ok]
    s.sql = ok[-1] if ok else ""
    return s


def digest(t: Trial, events: list[Event]) -> None:
    """从一轮的事件流里取出判分和统计要用的东西，写回 t。"""
    t.sql_calls = extract_sql_calls(events)
    t.refs = len(REF.findall(t.answer))
    edited = compacted = False
    for e in events:
        if isinstance(e, ContextEdited):
            t.context_edits += 1
            edited = True
            compacted |= e.kind == "HistoryCompacted"
            if e.usage.prompt_tokens:
                t.summary_calls.append(e.usage)
        elif isinstance(e, ContextEditFailed):
            t.compaction_failures += 1
            if e.usage.prompt_tokens:
                t.summary_calls.append(e.usage)
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
    digest_sandbox(t, events)


SANDBOX_TOOLS = ("run_python", "run_r")


def digest_sandbox(t: Trial, events: list[Event]) -> None:
    """沙箱代码、出的图、看图：自己画了图（代码里没调 fh_ 模板）之后，有没有 view_image 看一眼。"""
    pending: ToolStarted | None = None
    unviewed = False                      # 最近一次自己画的图还没看过
    for e in events:
        if isinstance(e, ToolStarted):
            pending = e
            continue
        if not isinstance(e, ToolFinished) or pending is None:
            continue
        if pending.name in SANDBOX_TOOLS:
            code = str(pending.arguments.get("code", ""))
            t.code.append(code)
            figures = e.details.figures if isinstance(e.details, Execution) else []
            t.figures += [_relative_figure(f) for f in figures]
            if figures and "fh_" not in code:
                t.custom_plots += 1
                unviewed = True
        elif pending.name == "view_image" and not e.is_error and unviewed:
            t.viewed_after += 1
            unviewed = False
        pending = None


def _relative_figure(path: Path) -> str:
    parts = path.parts
    return "/".join(parts[parts.index("figures"):]) if "figures" in parts else path.name


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
    def summary_calls(self) -> list[Usage]:
        return [u for t in self.turns for u in t.summary_calls]

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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionTrial":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["turns"] = [Trial.from_dict(t) for t in d["turns"]]
        return cls(**d)


def run_session(session: Session, trial: int, settings: Settings, db: Database,
                golds: dict[str, Gold], forced: dict[str, Any] | None = None) -> SessionTrial:
    """同一个 Agent 按顺序回答每一轮。某一轮出错（Agent.run 是事务，历史会回滚）就记下来接着问。

    配置的优先级：forced（命令行 --set）> 会话自带的 settings > .env。
    """
    st = SessionTrial(session.id, trial)
    events: list[Event] = []
    started = time.perf_counter()
    try:
        app = build_application(settings.model_copy(update={**session.settings, **(forced or {})}),
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
        if case.graded:
            show(t, app.results)     # 整段会话的结果：回答可以引用前面几轮的编号
        st.edits += [{"turn": n, "kind": e.kind, "before": e.tokens_before, "after": e.tokens_after}
                     for e in turn_events if isinstance(e, ContextEdited)]
        if case.graded:
            grade(t, case, db, golds[case.id])
        st.turns.append(t)

    st.elapsed_s = round(time.perf_counter() - started, 1)
    st.transcript = _transcript(app.agent.context.history)
    app.close()
    return st


def show(t: Trial, results: ResultStore) -> None:
    """算出用户看到的回答：{{r3}} 展开成整张表。判分按它来 —— 表里的数用户看得到，就算说过了。

    只给判分的轮次算：填充轮的清单动辄几百行，展开了只会撑大存档。
    """
    if t.refs:
        t.shown = results.expand(t.answer, lambda r: markdown_table(r.result.columns, r.result.rows))


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
                calls.append(SqlCall(str(args.get("sql", "")), not e.is_error, str(args.get("purpose", ""))))
            pending = None
    return calls


def grade(t: Trial, case: Case, db, gold: Gold) -> None:
    """给一次 trial 判分，结果写回 t。db 只需要有 query(sql, max_rows=) 方法。

    判分只写下面这些字段，先清掉 —— 改了判分规则重判（--regrade）时不留上一次的结论。
    """
    t.result = t.answer_check = None
    t.final_strict = t.text_ok = False
    t.matched_sql = t.grade_error = ""
    if t.submission is not None:
        t.submission.strict = False
    if case.no_sql:
        grade_files(t, case)
        return
    if case.match == "answer":
        t.answer_check = check_answer(gold.answer or [], t.shown or t.answer)
        ok = t.answer_check.ok is True
        t.result = ResultMatch(ok, ok, "" if ok else "回答里没说到标准答案的数")
        return
    if case.match == "empty":
        # 该查不到东西的题按回答判：说了「没有」就算对。Agent 常常先查数据覆盖哪几年
        # 来证明没有（冒烟测试里就是这样），这比硬跑一条返回空的 SQL 更好。
        says_none = any(w in (t.shown or t.answer) for w in NO_DATA_WORDS)
        t.result = ResultMatch(says_none, says_none, "" if says_none else "回答里没说查不到数据")
        return

    # Agent 跑过的每一条成功的 SQL 都拿来比，有一条查出了标准答案就算对（去重，先比后面的）
    best: tuple[ResultMatch, int] | None = None
    for n, sql in enumerate(dict.fromkeys(c.sql for c in reversed(t.sql_calls) if c.ok)):
        rows = _rerun(t, db, sql)
        if rows is None:
            continue
        for i, g in enumerate(gold.alternatives):
            m = compare_results(g, rows, case.match)
            # reversed 之后第一条就是最后执行的那条。BIRD 判法只认原版标准答案（第一条）
            if n == 0 and i == 0:
                t.final_strict = m.strict
            if best is None or _better(m, best[0]):
                best = (m, i)
                if m.lenient:
                    t.matched_sql = sql
        if best and best[0].strict:
            break
    if t.submission is not None:
        rows = _rerun(t, db, t.official_sql) if t.official_sql else None
        t.submission.strict = rows is not None and compare_results(gold.alternatives[0], rows, case.match).strict
    if best is not None:
        t.result, alt = best
        t.answer_check = check_answer(gold.answer if gold.answer is not None else gold.alternatives[alt],
                                      t.shown or t.answer)
    if not t.result_ok:
        t.text_ok = any(said_scalar(g, t.shown or t.answer) for g in gold.alternatives)


NO_DATA_WORDS = ("没有", "无数据", "不存在", "为空", "暂无", "查不到")


FINAL = re.compile(r"最终答案\s*[:：]\s*(.+)")


def extract_final(answer: str) -> str:
    """回答里最后一个「最终答案：」后面的内容，去掉 Markdown 的加粗、反引号。"""
    found = FINAL.findall(answer)
    return found[-1].strip().strip("*`").strip() if found else ""


def grade_files(t: Trial, case: Case) -> None:
    """没有 SQL 的题：该做的做了没有（代码、图、回答里该指出的问题）记在 result，数字记在 answer_check。
    DABstep 的题只看「最终答案」那一行，按官方规则比。"""
    t.final_answer = extract_final(t.answer)
    if case.official_answer is not None:
        ok = bool(t.final_answer) and question_scorer(t.final_answer, case.official_answer)
        reason = "" if ok else (f"最终答案 {t.final_answer!r}，标准答案 {case.official_answer!r}" if t.final_answer
                                else "回答里没有「最终答案：」那一行")
        t.result = ResultMatch(ok, ok, reason)
        return
    code = "\n".join(t.code)
    problems = [f"代码里没有 {p}" for p in case.expect_code if not re.search(p, code, re.DOTALL)]
    answer = t.shown or t.answer          # {{r5}} 展开成整张表之后：表里的数用户看得到，就算说过了
    problems += [f"回答里没有 {p}" for p in case.expect_text if not re.search(p, answer)]
    if case.expect_figure and not t.figures:
        problems.append("没有画出图")
    t.result = ResultMatch(not problems, not problems, "；".join(problems))
    t.answer_check = check_values(case.gold_values, answer) if case.gold_values else None


def _better(a: ResultMatch, b: ResultMatch) -> bool:
    return (a.lenient, a.strict) > (b.lenient, b.strict)


def _rerun(t: Trial, db, sql: str) -> list[tuple] | None:
    """重跑 Agent 的 SQL 拿结构化结果。工具返回给模型的是截断过的文本表格，没法精确比。"""
    try:
        return db.query(sql, max_rows=GRADE_MAX_ROWS).rows
    except Exception as exc:  # noqa: BLE001
        t.grade_error = f"{type(exc).__name__}: {exc}"
        return None


def run_gold(case: Case, db: Database | None) -> Gold:
    """跑标准答案。每道题只跑一次，所有 trial 共用。上传文件的题没有 SQL，标准值写在题里。"""
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
