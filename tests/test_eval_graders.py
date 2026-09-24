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


def test_容差只容得下ROUND到两位():
    gold = [(Decimal("0.3318"),)]
    assert compare_results(gold, [(0.33,)]).lenient
    assert not compare_results(gold, [(0.34,)]).lenient, "差 0.008 的比例是真的不一样"


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
        assert any(abs(v - s) < 1e-6 * max(1, v) for s in said), v


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
        ToolFinished("run_sql", False, "列不存在", 3),
        ToolStarted("describe_table", {"table": "orders"}),
        ToolFinished("describe_table", True, "…", 2),
        ToolStarted("run_sql", {"sql": "SELECT good"}),
        ToolFinished("run_sql", True, "| 1 |", 5),
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
