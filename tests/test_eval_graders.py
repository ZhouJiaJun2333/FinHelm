"""评测的判分器、事件解析和题库格式。不需要数据库和模型。

判分器有 bug，所有分数就都没意义了 —— 每条「算对 / 算错」的规则都要有测试钉住。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from data_agent.core.events import LLMResponded, ToolFinished, ToolStarted
from evals.cases import Case, load_cases
from evals.graders import answer_numbers, check_answer, compare_results
from evals.runner import Gold, SqlCall, Trial, extract_sql_calls, grade

REGION_SALES = [("华东", Decimal("8100531.46509")), ("华北", Decimal("5612478.51907"))]


# ================================================================ 结果比对
def test_顺序列名都不管_只看值():
    pred = [(Decimal("5612478.52"), "华北"), (Decimal("8100531.47"), "华东")]   # 列换了位置，ROUND 过
    m = compare_results(REGION_SALES, pred)
    assert m.strict and m.lenient


def test_多出几列_宽松算对_严格算错():
    pred = [("华东", 586, Decimal("8100531.47")), ("华北", 430, Decimal("5612478.52"))]
    m = compare_results(REGION_SALES, pred)
    assert m.lenient and not m.strict


@pytest.mark.parametrize("pred, reason", [
    ([("华东", 8100531.47)], "行数不对"),
    ([("华东", 8100531.47), ("华北", 5612000.00)], "找不到对应的列"),
    ([("华东", 5612478.52), ("华北", 8100531.47)], "整行对不上"),            # 数对了，区域串了
    ([("华东",), ("华北",)], "列不够"),
    ([], "空的"),
])
def test_这些都算错(pred, reason):
    m = compare_results(REGION_SALES, pred)
    assert not m.lenient and reason in m.reason


def test_两个整数差1就是不一样():
    """年龄 91 和 90：Decimal 整数按「写到个位」各有 0.5 的舍入误差，但一个不会是另一个舍入来的。"""
    gold = [(9, Decimal("91"))]
    assert not compare_results(gold, [(9, Decimal("90"))]).lenient
    assert compare_results([(Decimal("90.4"),)], [(Decimal("90"),)]).lenient, "ROUND 到个位照样算对"


def test_容差只容得下ROUND到两位():
    gold = [(Decimal("0.3318"),)]
    assert compare_results(gold, [(0.33,)]).lenient
    assert not compare_results(gold, [(0.34,)]).lenient, "差 0.008 的比例是真的不一样"


def test_容差按写出来的精度算_小比例不能差一截():
    """固定容差 0.005 对 0.05 量级的退货率等于容了 10% —— 2024 年的 5.61% 被当成全年的 5.57%。"""
    gold = [("线上", Decimal("0.05573248407643312102"))]
    assert compare_results(gold, [("线上", Decimal("0.06"))]).lenient, "ROUND(x, 2) 的舍入照样算对"
    assert compare_results(gold, [("线上", Decimal("5.57"))]).lenient, "百分数写到两位小数"
    assert not compare_results(gold, [("线上", Decimal("0.0561"))]).lenient
    assert not compare_results(gold, [("线上", Decimal("5.61"))]).lenient
    assert not compare_results(gold, [("线上", 0.05608555399719495)]).lenient, "全精度的 float 没有舍入余地"


def test_回答核对_小比例也按写出来的精度():
    gold = [(Decimal("0.05573248407643312102"),)]
    assert check_answer(gold, "线上退货率 5.57%").ok
    assert check_answer(gold, "约 5.6%").ok
    assert not check_answer(gold, "线上退货率 5.61%").ok


def test_比例乘了100也算对():
    gold = [("线上", Decimal("0.0557")), ("线下", Decimal("0.0598"))]
    assert compare_results(gold, [("线上", 5.57), ("线下", 5.98)]).lenient


def test_ordered要看顺序_set不看():
    gold = [("线上", 3.0), ("线下", 2.0)]
    flipped = [("线下", 2.0), ("线上", 3.0)]
    assert compare_results(gold, flipped, "set").lenient
    assert not compare_results(gold, flipped, "ordered").lenient


def test_top只看前几名_后面多列几名不算错():
    gold = [("客户0071", 346022.41)]
    assert compare_results(gold, [("客户0071", 346022.41), ("客户0174", 338369.89)], "top").lenient
    assert not compare_results(gold, [("客户0174", 338369.89), ("客户0071", 346022.41)], "top").lenient


def test_distinct去重之后再比_BIRD的判法():
    """BIRD 的标准 SQL 常常不写 DISTINCT：461 行全是 DISPONENT。Agent 写了 DISTINCT 只有 1 行。"""
    gold = [("DISPONENT",)] * 461
    assert compare_results(gold, [("DISPONENT",)], "distinct").strict
    assert not compare_results(gold, [("DISPONENT",)], "set").lenient, "set 要求重复的行一一对上"
    assert not compare_results(gold, [("DISPONENT",), ("OWNER",)], "distinct").lenient


def test_contains多出的行不算错():
    gold = [("政府", 12181016.01), ("企业", 9452261.85)]
    pred = [("个人", 10333264.28), ("企业", 9452261.85), ("政府", 12181016.01)]
    assert compare_results(gold, pred, "contains").lenient
    assert not compare_results(gold, pred, "set").lenient


@pytest.mark.parametrize("pred, ok", [
    ([], True), ([(None,)], True), ([(0,)], True), ([(Decimal("0.00"),)], True),
    ([(123.4,)], False), ([(None,), (None,)], False),
])
def test_empty_查不到才算对(pred, ok):
    assert compare_results([], pred, "empty").lenient is ok


def test_NULL单列成文字标签也算对():
    """COALESCE(region, '未填写') 是很正常的展示方式（冒烟测试里 Agent 就这么做的）。"""
    gold = [("华东", 61), (None, 9), ("西南", 37)]
    assert compare_results(gold, [("华东", 61), (None, 9), ("西南", 37)]).lenient
    assert compare_results(gold, [("未填写", 9), ("华东", 61), ("西南", 37)]).lenient


def test_NULL的放宽只对文字_而且只朝一个方向():
    assert not compare_results([(None,)], [(0.0,)]).lenient, "数字的 NULL 不能当成 0"
    assert not compare_results([("华东", 61)], [(None, 61)]).lenient, "Agent 给 NULL 不能冒充标准答案的文字"


def test_带NULL的行最后配_不会抢走别的行():
    """NULL 能对上任何文字；先配它的话，它可能把「华北」那行占掉，后面华北就配不上了。"""
    gold = [(None, 5), ("华北", 5)]
    assert compare_results(gold, [("华北", 5), ("未填写", 5)]).lenient


# ================================================================ 回答核对
def test_回答里的数字_各种写法都认():
    said = answer_numbers("销售额 8,100,531.47 元（约 810.05 万），占比 33.18%，共 1.2 亿")
    for v in (8100531.47, 8100500.0, 0.3318, 1.2e8):
        assert any(abs(v - s.value) < 1e-6 * max(1, v) for s in said), v


def test_回答核对_四舍五入到万也算说对了():
    assert check_answer(REGION_SALES, "华东 810.05 万，华北 561.25 万").ok
    assert check_answer([(Decimal("0.3318"),)], "增长了 33.2%").ok
    assert not check_answer([(586,)], "一共 588 单").ok, "整数要准"


def test_回答核对_漏说的数字会列出来():
    c = check_answer(REGION_SALES, "华东 8,100,531.47 元")
    assert c.ok is False and c.missing == [pytest.approx(5612478.51907)]


def test_结果太多行或者没有数字时不核对():
    assert check_answer([(i,) for i in range(50)], "…").ok is None
    assert check_answer([("客户0114",)], "最好的客户是客户0114").ok is None


# ================================================================ 事件解析
def test_从事件里取出每条SQL和成败():
    events = [
        LLMResponded(1, "", ["run_sql"]),
        ToolStarted("run_sql", {"sql": "SELECT bad", "purpose": "试试"}),
        ToolFinished("run_sql", "列不存在", is_error=True, elapsed_ms=3),
        ToolStarted("describe_table", {"table": "orders"}),
        ToolFinished("describe_table", "…", is_error=False, elapsed_ms=2),
        ToolStarted("run_sql", {"sql": "SELECT good"}),
        ToolFinished("run_sql", "| 1 |", is_error=False, elapsed_ms=5),
    ]
    calls = extract_sql_calls(events)
    assert [(c.sql, c.ok) for c in calls] == [("SELECT bad", False), ("SELECT good", True)]
    assert calls[0].purpose == "试试"


def test_失败归类的优先级():
    assert Trial("x", 1, error="APIError").failure == "运行出错"
    assert Trial("x", 1).failure == "没有执行成功的 SQL"
    t = Trial("x", 1, step_limit=True)
    assert t.failure == "步数耗尽"


# ================================================================ 题库格式
def test_题库能读_match都合法_每题至少一条标准SQL():
    cases = load_cases("shop").cases
    assert len(cases) >= 20
    for c in cases:
        assert c.match in ("set", "ordered", "top", "contains", "empty"), c.id
        assert c.gold_sql and all(s.strip().upper().startswith("SELECT") for s in c.gold_sql), c.id
        assert c.tags, f"{c.id} 没有标签，报告里没法分类"


def test_BIRD题库_题库级配置选场景包_按去重集合判():
    cs = load_cases("bird_financial")
    assert cs.settings == {"domain": "financial"}
    assert len(cs.cases) == 32
    assert all(c.match == "distinct" and "提示（外部知识）" in c.question for c in cs.cases)


def test_只看最后一条_BIRD官方判法():
    """我们的主分数看任何一条；BIRD 只看最后一条。先查对、最后又查了一条明细 → 主分数对，BIRD 判法错。"""
    case = Case("c", "q", ("gold",))
    db = FakeDB({"SELECT 答案": [(61,)], "SELECT 明细": [("华东", 61), ("华北", 42)]})
    t = trial_with([("SELECT 答案", True), ("SELECT 明细", True)], "61")
    grade(t, case, db, Gold([[(61,)]]))
    assert t.result_ok and not t.final_strict

    t = trial_with([("SELECT 明细", True), ("SELECT 答案", True)], "61")
    grade(t, case, db, Gold([[(61,)]]))
    assert t.result_ok and t.final_strict


# ================================================================ 一次 trial 怎么判
class FakeDB:
    """按 SQL 文本返回写死的结果；没登记的 SQL 当成执行出错。"""

    def __init__(self, results: dict[str, list[tuple]]) -> None:
        self.results = results

    def query(self, sql, max_rows=0):
        if sql not in self.results:
            raise RuntimeError(f"relation does not exist: {sql}")
        return type("R", (), {"rows": self.results[sql]})()


def trial_with(sqls: list[tuple[str, bool]], answer: str) -> Trial:
    return Trial("x", 1, answer=answer, sql_calls=[SqlCall(s, ok) for s, ok in sqls])


def test_最后一条是给你对比的明细_前面那条查对了也算对():
    """冒烟测试里的真实情况：先查出 2024 年销售额，又按三种状态列了一遍给用户对比口径。"""
    case = Case("shop-003", "2024 年的销售额是多少？", ("gold",), match="contains")
    db = FakeDB({
        "SELECT sum": [(Decimal("31966542.15"),)],
        "SELECT by status": [("completed", 2438, Decimal("31966542.15")),
                             ("cancelled", 264, Decimal("3360777.19"))],
    })
    t = trial_with([("SELECT sum", True), ("SELECT by status", True)], "2024 年销售额 ¥31,966,542.15")
    grade(t, case, db, Gold([[(Decimal("31966542.14761"),)]]))
    assert t.result_ok and t.answer_ok


def test_执行失败的SQL不参与判分():
    case = Case("c", "q", ("gold",))
    t = trial_with([("SELECT bad", False), ("SELECT good", True)], "61")
    grade(t, case, FakeDB({"SELECT good": [(61,)]}), Gold([[(61,)]]))
    assert t.result_ok and t.matched_sql == "SELECT good"


def test_几种标准答案_回答核对用对上的那一种():
    case = Case("shop-007", "各区域客户数", ("含NULL", "剔除NULL"))
    gold = [[("华东", 61), (None, 9)], [("华东", 61)]]
    t = trial_with([("SELECT", True)], "华东 61 个")
    grade(t, case, FakeDB({"SELECT": [("华东", 61)]}), Gold(gold))
    assert t.result_ok and t.answer_ok, "对上的是「剔除 NULL」那种，回答里就不该要求出现 9"


def test_重跑出错记下来_不影响别的SQL():
    case = Case("c", "q", ("gold",))
    t = trial_with([("SELECT ok", True), ("SELECT gone", True)], "1")   # 从后往前比，gone 先被重跑
    grade(t, case, FakeDB({"SELECT ok": [(1,)]}), Gold([[(1,)]]))
    assert t.result_ok and "relation does not exist" in t.grade_error


@pytest.mark.parametrize("answer, ok", [
    ("库里只有 2024、2025 年的数据，2030 年没有任何订单。", True),
    ("2030 年销售额为 12,345 元", False),
])
def test_该查不到的题按回答判(answer, ok):
    case = Case("shop-014", "2030 年的销售额是多少？", ("gold",), match="empty")
    t = trial_with([("SELECT years", True)], answer)
    grade(t, case, FakeDB({"SELECT years": [(2024, 2853), (2025, 2147)]}), Gold([[(None,)]]))
    assert t.result_ok is ok


def test_查出分量在回答里自己算_取数算对_但回答里必须有最终的数():
    """全量评测里的真实情况：Agent 查出线上、线下各多少，在回答里减出差值。"""
    case = Case("shop-017", "线上比线下多多少？", ("差值", "分量"), match="contains", answer_sql="差值")
    gold = Gold([[(3963255.10,)], [("线上", 12178614.66), ("线下", 8215359.56)]], answer=[(3963255.10,)])
    db = FakeDB({"SELECT by channel": [("线上", 12178614.66), ("线下", 8215359.56), ("分销", 3774094.19)]})

    right = trial_with([("SELECT by channel", True)], "线上比线下多约 396.33 万元")
    grade(right, case, db, gold)
    assert right.result_ok and right.answer_ok

    wrong = trial_with([("SELECT by channel", True)], "线上 1217.86 万，线下 821.54 万，多了 400 万")
    grade(wrong, case, db, gold)
    assert wrong.result_ok and not wrong.answer_ok, "分量查对了、减错了：取数对，结论错"


# ================================================================ 缓存统计
def test_缓存命中率按token加权_首次和后续分开算():
    from data_agent.core.messages import Usage
    from evals.report import cache_stats

    one_step = Trial("a", 1, calls=[Usage(input=100, cache_read=900)])
    two_steps = Trial("b", 1, calls=[Usage(input=1000), Usage(input=100, cache_read=1900)])
    for t in (one_step, two_steps):
        t.usage = sum(t.calls, Usage())

    s = cache_stats([one_step, two_steps])
    assert s["命中率"] == 2800 / 4000                 # 不是 (0.9 + 0.475) / 2
    assert s["首次调用命中率"] == 900 / 2000
    assert s["后续调用命中率"] == 1900 / 2000
    assert cache_stats([one_step])["后续调用命中率"] is None   # 没有后续调用，不是 0%


# ================================================================ 多轮会话
def test_多轮题库能读_轮次id和填充轮():
    cs = load_cases("shop_multi")
    assert cs.sessions and not cs.cases
    first = cs.sessions[0]
    assert first.turns[0].id == f"{first.id}/1"
    assert first.settings["context_clear_trigger_tokens"] > 0
    graded = cs.graded_cases
    assert graded and all(not c.filler for c in graded)
    assert any(c.match == "answer" for c in graded)


def test_只看回答的题_说到数就对_不需要SQL():
    case = Case("m/8", "第一个问题里华东是多少？", (), match="answer", answer_sql="SELECT 1")
    gold = Gold([], [(Decimal("10273505.48803"),)])
    ok = Trial("m/8", 1, answer="华东 2024 年是 1027.35 万元。")
    grade(ok, case, None, gold)
    assert ok.answer_ok and ok.failure == ""
    bad = Trial("m/8", 1, answer="大约 900 万。")
    grade(bad, case, None, gold)
    assert not bad.answer_ok and bad.failure == "结果不对"


def test_填充轮不算失败():
    assert Trial("m/3", 1, graded=False).failure == ""


def test_记下哪几次调用紧跟在整理之后():
    from data_agent.core.events import ContextEdited
    from data_agent.core.messages import Usage
    from evals.runner import SessionTrial, digest

    t1, t2 = Trial("m/1", 1), Trial("m/2", 1)
    digest(t1, [LLMResponded(1, ""), ContextEdited("清理", 10, 5), LLMResponded(2, "")])
    digest(t2, [ContextEdited("压缩", 10, 5, kind="HistoryCompacted"), LLMResponded(1, "")])
    assert t1.after_edit == [1] and t1.context_edits == 1
    assert t2.after_edit == [0]
    st = SessionTrial("m", 1, turns=[t1, t2])
    assert st.after_edit == [1, 2]            # 第二轮的第 0 次 = 整段会话的第 2 次

    from evals.report import cache_stats
    t1.calls = [Usage(input=1000), Usage(input=600, cache_read=400)]
    t2.calls = [Usage(input=900, cache_read=100)]
    s = cache_stats([st])
    assert t2.after_compact == [0] and st.after_compact == [2]
    assert s["清理后调用次数"] == 1 and s["清理后调用命中率"] == 400 / 1000
    assert s["压缩后调用次数"] == 1 and s["压缩后调用命中率"] == 100 / 1000


def test_命令行临时配置_数字布尔按JSON解析_字段名要存在():
    from evals.run import _parse_sets

    got = _parse_sets(["context_clear_trigger_tokens=1000", "CONTEXT_COMPACT_MAX_TOKENS=16000",
                       "openai_native_thinking=false", "openai_model=deepseek-flash"])
    assert got == {"context_clear_trigger_tokens": 1000, "context_compact_max_tokens": 16000,
                   "openai_native_thinking": False, "openai_model": "deepseek-flash"}
    with pytest.raises(SystemExit):
        _parse_sets(["context_clear_trigger=1"])          # 少了 _tokens，字段名不对


def test_运行记录存下再读回来一模一样():
    """--rebuild 靠它：报告那一步崩了，能从 jsonl 重新出报告。"""
    import json

    from data_agent.core.messages import Usage
    from evals.graders import AnswerCheck, ResultMatch
    from evals.runner import SessionTrial

    t = Trial("m/1", 2, answer="华东 810 万", sql_calls=[SqlCall("SELECT 1", True, "试试")],
              usage=Usage(10, 2, 30, 0), calls=[Usage(5, 1), Usage(5, 1, 30)], after_edit=[1],
              result=ResultMatch(True, False, ""), answer_check=AnswerCheck(False, [8.1e6], 2))
    st = SessionTrial("m", 2, turns=[t, Trial("m/2", 2, graded=False, error="OutputTruncated: …")],
                      edits=[{"turn": 1, "kind": "ToolResultsCleared", "before": 9, "after": 3}])
    back = SessionTrial.from_dict(json.loads(json.dumps(st.to_dict())))
    assert back == st
    assert back.turns[0].answer_ok is False and back.turns[1].failure == ""


def test_找上一次运行_只认同一题库同一标签(tmp_path, monkeypatch):
    import json

    import evals.run as run

    def fake(name, cases, label):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"cases": cases, "label": label}), encoding="utf-8")
        (d / "summary.json").write_text("{}", encoding="utf-8")
        return d

    monkeypatch.setattr(run, "RUNS_DIR", tmp_path)
    fake("20260101-000000_shop_multi_m", "shop_multi", "")
    want = fake("20260102-000000_shop_multi_m_现状", "shop_multi", "现状")
    fake("20260103-000000_shop_multi_m_不清理", "shop_multi", "不清理")
    fake("20260104-000000_shop_m", "shop", "")
    assert run._previous_run("last", "shop_multi", "现状", tmp_path / "now") == want


def test_回答引用了结果_按用户看到的整张表判分():
    """模型写 {{r1}}，用户在那个位置看到整张表 —— 表里的数算说过了。"""
    from data_agent.db.connection import QueryResult
    from data_agent.tools.sql.results import ResultStore
    from evals.runner import digest, show

    rows = [("华东", 61), ("华北", 42)]
    results = ResultStore()
    table = results.add("SELECT", QueryResult(["region", "n"], rows, False, 1))
    events = [ToolStarted("run_sql", {"sql": "SELECT"}),
              ToolFinished("run_sql", "预览", is_error=False, elapsed_ms=1, details=table)]
    case = Case("x", "各区域多少单？", ("gold",), match="set")

    t = Trial("x", 1, answer="各区域订单数如下：\n{{r1}}")
    digest(t, events)
    show(t, results)
    grade(t, case, FakeDB({"SELECT": rows}), Gold([rows]))
    assert t.refs == 1 and "| 华北 | 42 |" in t.shown
    assert t.answer_ok


def test_多一行合计不算错_但只认文字标签():
    gold = [("线上", 100), ("线下", 60), ("分销", 40)]
    with_total = gold + [("合计", 200)]
    assert compare_results(gold, with_total, "set").lenient
    assert compare_results(gold, with_total, "ordered").lenient
    # NULL 那一行可能是真实的分组（区域为空的客户），不能当合计去掉
    assert not compare_results(gold, gold + [(None, 200)], "set").lenient
    # 多出来的不是合计：照样算错
    assert not compare_results(gold, gold + [("其他", 5)], "set").lenient


# ================================================================ BIRD：兜底、存疑题、提交轮
def test_回答里把单个数算对了_兜底算回答对():
    """q169：Agent 查出两年的总额，在回答里自己算出增长率 25.30%。"""
    from evals.graders import said_scalar

    gold = [(25.300191222790616,)]
    assert said_scalar(gold, "增长率 = (12,635,988 − 10,084,572) / 10,084,572 ≈ **+25.30%**")
    assert said_scalar(gold, "增长率大约 25%"), "写到整数，25.30 取整就是 25"
    assert not said_scalar(gold, "增长率 26.1%")
    assert not said_scalar([(13,)], "1. 共有 13 个账户"), "整数（个数、ID）不兜底"
    assert not said_scalar([(2836,)], "答案是账户 1372。候选：| 3428 | 2836 |"), "真跑出来的误判：列过 ≠ 答的是它"
    assert not said_scalar([(25.3, 1)], "25.30"), "多个数不兜底"
    assert not said_scalar([("POPLATEK MESICNE",)], "POPLATEK MESICNE"), "文字不兜底"


def test_兜底判对_不算失败_但结果对不变():
    case = Case("c", "q", ("gold",), match="distinct")
    db = FakeDB({"SELECT 两年": [(1996, 10084572), (1997, 12635988)]})
    t = trial_with([("SELECT 两年", True)], "增长率 25.30%")
    grade(t, case, db, Gold([[(25.300191222790616,)]]))
    assert t.text_ok and t.answer_ok and t.failure == ""
    assert not t.result_ok, "结果对只看 SQL"


def test_BIRD判法只认原版标准答案_主分数哪种都认():
    """标注存疑的题：Agent 按我们补的写法查对了 —— 主分数算对，BIRD 判法（最后一条、提交）算错。"""
    from evals.runner import Submission

    case = Case("c", "q", ("原版", "补的"), match="distinct")
    db = FakeDB({"SELECT 对的": [(40.0,)]})
    t = trial_with([("SELECT 对的", True)], "40.0%")
    t.final_sql, t.submission = "SELECT 对的", Submission("SELECT 对的")
    grade(t, case, db, Gold([[(44.26229508196721,)], [(40.0,)]]))
    assert t.result_ok and t.answer_ok
    assert not t.final_strict and not t.submission.strict


def test_重判不留上一次的结论():
    """--regrade：同一个 trial 按新规则再判一遍，上次判对的字段要清掉。"""
    case = Case("c", "q", ("gold",), match="distinct")
    t = trial_with([("SELECT 旧", True)], "增长率 25.30%")
    grade(t, case, FakeDB({"SELECT 旧": [(25.300191222790616,)]}), Gold([[(25.300191222790616,)]]))
    assert t.result_ok and t.final_strict
    grade(t, case, FakeDB({"SELECT 旧": [(1,)]}), Gold([[(99.5,)]]))
    assert not t.result_ok and not t.final_strict and not t.text_ok and not t.matched_sql


def test_提交轮按交的那条判_没交就用最后一条():
    from evals.runner import Submission

    case = Case("c", "q", ("gold",), match="distinct")
    db = FakeDB({"SELECT 明细": [(10451, 482940), (6034, 464520)], "SELECT 只要ID": [(10451,)]})
    t = trial_with([("SELECT 明细", True)], "账户 10451")
    t.final_sql, t.submission = "SELECT 明细", Submission("SELECT 只要ID")
    grade(t, case, db, Gold([[(10451,)]]))
    assert t.submission.strict and t.official_sql == "SELECT 只要ID"

    t = trial_with([("SELECT 明细", True)], "账户 10451")
    t.final_sql, t.submission = "SELECT 明细", Submission("")
    grade(t, case, db, Gold([[(10451,)]]))
    assert t.official_sql == "SELECT 明细" and not t.submission.strict


def test_提交轮_收这一轮最后一条成功的SQL_出错不算这题错():
    from data_agent.core.messages import Usage
    from evals.runner import submit_sql

    class FakeAgent:
        def __init__(self, events, fail=False):
            self.events, self.fail, self.session_usage = events, fail, Usage(100, 10)

        def run(self, prompt):
            self.events += [
                LLMResponded(Usage(120, 5), "", ["run_sql"]),
                ToolStarted("run_sql", {"sql": "SELECT 交的"}),
                ToolFinished("run_sql", "| 1 |", is_error=False, elapsed_ms=1),
                ToolStarted("run_sql", {"sql": "SELECT 写错"}),
                ToolFinished("run_sql", "列不存在", is_error=True, elapsed_ms=1),
            ]
            self.session_usage = Usage(220, 15)
            if self.fail:
                raise RuntimeError("超时")

    before = [ToolStarted("run_sql", {"sql": "SELECT 回答那轮"}),
              ToolFinished("run_sql", "| 1 |", is_error=False, elapsed_ms=1)]
    for fail in (False, True):
        events = list(before)
        app = type("App", (), {"agent": FakeAgent(events, fail)})()
        s = submit_sql(app, "请交一条", events)
        assert s.sql == "SELECT 交的", "只看提交轮，失败的那条不算"
        assert s.steps == 1 and s.usage == Usage(120, 5)
        assert bool(s.error) is fail


def test_题库级配置_submit和不认识的键(tmp_path, monkeypatch):
    import evals.cases as cases_mod

    monkeypatch.setattr(cases_mod, "CASES_DIR", tmp_path)
    (tmp_path / "x.jsonl").write_text(
        '{"settings": {"domain": "financial"}, "submit": "交一条"}\n'
        '{"id": "a", "question": "q", "gold_sql": "SELECT 1"}\n', encoding="utf-8")
    cs = load_cases("x")
    assert cs.settings == {"domain": "financial"} and cs.submit == "交一条"

    (tmp_path / "y.jsonl").write_text('{"setting": {"domain": "financial"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="不认识的键"):
        load_cases("y")


def test_BIRD题库_有提交轮_存疑题原版在第一条():
    cs = load_cases("bird_financial")
    assert "run_sql" in cs.submit
    disputed = [c for c in cs.cases if "标注存疑" in c.tags]
    assert [c.id for c in disputed] == [f"bird-fin-0{n}" for n in (115, 129, 152, 186, 194)]
    assert all(len(c.gold_sql) >= 2 and "标注存疑：" in c.note for c in disputed)
    assert all(len(c.gold_sql) == 1 for c in cs.cases if c not in disputed)


def test_带提交轮的运行记录存下再读回来一模一样():
    import json

    from data_agent.core.messages import Usage
    from evals.runner import Submission

    t = Trial("c", 1, final_sql="SELECT 1", text_ok=True,
              submission=Submission("SELECT 2", True, 2, Usage(50, 3), ""))
    d = json.loads(json.dumps(t.to_dict()))
    assert d["official_sql"] == "SELECT 2"
    assert Trial.from_dict(d) == t
