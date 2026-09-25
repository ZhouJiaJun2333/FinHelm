"""research 题库（上传文件的题）：题库加载、判分、从事件里整理出代码 / 图 / 看图。不需要 key 和 Docker。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent.core.events import ToolFinished, ToolStarted
from data_agent.tools.sandbox import Execution
from evals.cases import CASES_DIR, Case, load_cases
from evals.graders import check_values
from evals.report import file_stats, render, summarize
from evals.runner import Trial, digest_sandbox, grade

WORK = Path("D:/x/work")


# ================================================================ 判分器
def test_标准值_按回答写到几位来容差():
    values = [0.48959, 0.34478, 0.695223, 0.921346]
    assert check_values(values, "RR 0.49（95% CI 0.34–0.70），I² = 92%").ok
    assert check_values(values, "RR 0.4896 (0.3448, 0.6952), I2 92.1%").ok
    check = check_values(values, "RR 0.49（95% CI 0.34–0.71），I² = 92%")
    assert not check.ok and check.missing == [0.695223]


def test_标准值_不看正负号():
    """「少住 13.98 天」、Unicode 减号、区间里的连字符都算说到了。"""
    values = [-13.981722, -24.029866, -3.933578]
    assert check_values(values, "卒中单元平均少住 13.98 天（95% CI 3.93–24.03）").ok
    assert check_values(values, "MD −13.98（−24.03 至 −3.93）").ok
    assert check_values(values, "MD -13.98 [-24.03, -3.93]").ok


def test_标准值_精确值不会因为写得短就放宽():
    """0.5 如果按写出来的精度算要容 0.05；标准值是算出来的，只按回答的精度容。"""
    assert not check_values([0.5], "0.54").ok
    assert check_values([0.5], "0.50").ok


def test_没有标准值就不核对():
    assert check_values([], "随便").ok is None


# ================================================================ 题库
def test_research题库能加载_文件都在():
    cs = load_cases("research")
    assert cs.settings["domain"] == "research"
    assert len(cs.cases) >= 10 and all(c.uses_files for c in cs.cases)
    for c in cs.cases:
        assert all((CASES_DIR / f).is_file() for f in c.files)
        assert c.gold_values or c.expect_code or c.expect_text or c.expect_figure, c.id


def test_标准值原样写进回答必须判对():
    """run.py 的自检也做这一步：生成的数（6 位小数）和判分器的容差对得上。"""
    for c in load_cases("research").cases:
        if c.gold_values:
            assert check_values(c.gold_values, " ".join(map(str, c.gold_values))).ok, c.id


def test_上传文件的题没有判分依据就报错(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.cases.CASES_DIR", tmp_path)
    (tmp_path / "a.xlsx").write_bytes(b"x")
    (tmp_path / "bad.jsonl").write_text(json.dumps({"id": "x", "files": ["a.xlsx"], "question": "q"}), encoding="utf-8")
    with pytest.raises(ValueError, match="判分依据"):
        load_cases("bad")
    (tmp_path / "missing.jsonl").write_text(
        json.dumps({"id": "x", "files": ["nope.xlsx"], "question": "q", "expect_figure": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="找不到文件"):
        load_cases("missing")


# ================================================================ 事件 → 代码、图、看图
def run(name: str, code: str = "", figures=(), path: str = "") -> list:
    args = {"code": code} if name != "view_image" else {"path": path}
    details = Execution(figures=[WORK / "figures" / f for f in figures]) if name != "view_image" else None
    return [ToolStarted(name=name, arguments=args),
            ToolFinished(name=name, content="", is_error=False, elapsed_ms=1, details=details)]


def test_自己画图之后看了图():
    t = Trial("rs-11", 1, uses_files=True)
    digest_sandbox(t, [*run("run_r", "m <- fh_meta_bin(d)\nfh_forest(m)", ["forest.png"]),
                       *run("run_python", "plt.scatter(x, y)", ["fig-1.png"]),
                       *run("view_image", path="figures/fig-1.png")])
    assert t.figures == ["figures/forest.png", "figures/fig-1.png"]
    assert (t.custom_plots, t.viewed_after) == (1, 1), "模板画的不算自己画的"
    assert len(t.code) == 2


def test_画了两次只看了一次():
    t = Trial("rs-11", 1, uses_files=True)
    digest_sandbox(t, [*run("run_python", "plt.plot(x)", ["fig-1.png"]),
                       *run("run_python", "plt.plot(y)", ["fig-2.png"]),
                       *run("view_image", path="figures/fig-2.png"),
                       *run("view_image", path="figures/fig-1.png")])
    assert (t.custom_plots, t.viewed_after) == (2, 1), "连看两次只算最近那张图看过"


# ================================================================ 判分
def case(**kw) -> Case:
    return Case(id="rs-x", question="q", gold_sql=(), files=("a.xlsx",), **kw)


def graded(c: Case, answer: str = "", code: str = "", figures=()) -> Trial:
    t = Trial(c.id, 1, uses_files=True, answer=answer, code=[code], figures=list(figures))
    grade(t, c, None, None)
    return t


def test_数字都说到了_步骤都做了才算对():
    c = case(gold_values=(1.238485, 1.129224, 1.358319), expect_code=(r"fh_meta_gen\(",))
    assert graded(c, "OR 1.24（1.13–1.36）", "fh_meta_gen(d, ...)").answer_ok
    t = graded(c, "OR 1.24（1.13–1.36）", "metagen(...)")
    assert t.failure == "要求的步骤没做" and "fh_meta_gen" in t.result.reason
    assert graded(c, "OR 1.25（1.13–1.36）", "fh_meta_gen(d)").failure == "回答里的数字不对"


def test_要求出图和指出问题():
    c = case(expect_figure=True, expect_text=("Heard", "写反|超过"))
    assert graded(c, "Heard 1998 的感染数超过了置管数，请核对", figures=["figures/a.png"]).answer_ok
    t = graded(c, "合并 RR 0.40")
    assert not t.answer_ok and "没有画出图" in t.result.reason and "Heard" in t.result.reason


def test_报告里有看图统计_没有SQL那几行():
    c = case(expect_figure=True)
    t = graded(c, "好了", figures=["figures/fig-1.png"])
    t.custom_plots, t.viewed_after = 1, 1
    s = summarize([c], [t])
    assert s["上传文件"] == file_stats([t]) == {"自己画图的trial": 1, "画完看了图的trial": 1,
                                               "自己画图次数": 1, "画完看图次数": 1}
    meta = {"cases": "research", "model": "m", "started": "", "git": "x", "dirty": False,
            "cases_sha1": "", "prompt_sha1": "", "trials": 1}
    report = render(meta, s)
    assert "要求的步骤都做了" in report and "## 看图" in report
    assert "最后一条 SQL" not in report and "列数也一样" not in report


def test_步数耗尽就不算对_哪怕图画出来了():
    c = case(expect_figure=True)
    t = graded(c, "已达到最大步数 20 仍未得出结论。", figures=["figures/fig-1.png"])
    t.step_limit = True
    assert t.result_ok and not t.answer_ok and t.failure == "步数耗尽"
    t.step_limit = False
    assert t.answer_ok
