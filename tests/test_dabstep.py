"""DABstep 接入：payments 场景包、/data 挂载、「最终答案」怎么抽、按官方规则判、导出提交文件。不需要 key 和 Docker。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent.domains import get_domain
from data_agent.prompts import build_system_prompt
from data_agent.tools.python import PYTHON_KERNEL
from data_agent.tools.sandbox import Sandbox
from evals.cases import Case, load_cases
from evals.dabstep.submission import export
from evals.report import render, summarize
from evals.runner import Trial, extract_final, grade


# ================================================================ 场景包
def test_payments场景包_讲数据目录_不叫用户上传():
    prompt = build_system_prompt(get_domain("payments"), ["run_python"])
    assert "`/data/`" in prompt and "manual.md" in prompt
    assert "用户上传" not in prompt and "/attach" not in prompt
    assert "run_sql" not in prompt


def test_数据目录只读挂进容器():
    sb = Sandbox.docker("img", PYTHON_KERNEL, Path("work"), lambda ref: {}, data_dir=Path("data/dabstep/context"))
    mounts = [sb.command[i + 1] for i, a in enumerate(sb.command) if a == "--mount"]
    data = [m for m in mounts if m.endswith("target=/data,readonly")]
    assert len(data) == 1 and "dabstep" in data[0]
    assert not any("/data" in m for m in Sandbox.docker("img", PYTHON_KERNEL, Path("w"), lambda r: {}).command)


def test_数据目录不存在_组装时就报错(monkeypatch, tmp_path):
    from data_agent.app import build_application
    from data_agent.settings import Settings

    from fakes import ScriptedProvider
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="数据目录"):
        build_application(Settings(domain="payments"), llm=ScriptedProvider())


# ================================================================ 最终答案
@pytest.mark.parametrize("answer, final", [
    ("算下来是 NL。\n\n最终答案：NL", "NL"),
    ("最终答案: **0.120132**", "0.120132"),
    ("先写一个 最终答案：A\n改一下\n最终答案：`B. BE`", "B. BE"),
    ("没写", None),
    ("列表为空。\n\n最终答案：\n", ""),                        # 官方要求：空列表回答空字符串
    ("最终答案：\n下一段说明", ""),                            # 只取同一行，不把下一行当答案
])
def test_抽最后一个最终答案(answer, final):
    assert extract_final(answer) == final


def test_空答案也算写了_标准答案是空的题能判对():
    c = dab(official_answer="")
    t = graded(c, "没有商户受影响。\n\n最终答案：")
    assert t.wrote_final and t.final_answer == "" and t.answer_ok


def dab(**kw) -> Case:
    return Case(id="dab-1", question="q", gold_sql=(), **kw)


def graded(c: Case, answer: str) -> Trial:
    t = Trial(c.id, 1, no_sql=True, graded=c.graded, official=True, answer=answer)
    grade(t, c, None, None)
    return t


def test_按官方规则判_列表不看顺序_数字有容差():
    assert graded(dab(official_answer="384, 394, 276"), "最终答案：276, 384, 394").answer_ok
    assert graded(dab(official_answer="0.120132"), "最终答案：0.12013").answer_ok
    t = graded(dab(official_answer="B. BE"), "最终答案：A. NL")
    assert t.failure == "最终答案不对" and "A. NL" in t.result.reason
    assert "没有「最终答案" in graded(dab(official_answer="NL"), "是 NL").result.reason


def test_答案不公开的题不判分_但记下最终答案():
    c = dab(answer_hidden=True)
    assert not c.graded and c.no_sql
    t = graded(c, "最终答案：42")
    assert t.final_answer == "42" and t.failure == ""


def test_题库能加载_dev有答案_正式题没有():
    dev, full = load_cases("dabstep_dev"), load_cases("dabstep")
    assert dev.settings["domain"] == "payments" == full.settings["domain"]
    assert len(dev.cases) == 10 and all(c.official_answer is not None for c in dev.cases)
    assert len(full.cases) == 450 and not any(c.graded for c in full.cases)
    assert "最终答案：" in dev.cases[0].question


def test_报告里不显示SQL和步骤那几行():
    c = dab(official_answer="NL")
    s = summarize([c], [graded(c, "最终答案：NL")])
    meta = {"cases": "dabstep_dev", "model": "m", "started": "", "git": "x", "dirty": False,
            "cases_sha1": "", "prompt_sha1": "", "trials": 1}
    report = render(meta, s)
    assert "要求的步骤都做了" not in report and "最后一条 SQL" not in report and "**100%**" in report


def test_答案不公开_报告里不出现对错的百分比():
    c = dab(answer_hidden=True, tags=("难度:hard",))
    report = render({"cases": "dabstep", "model": "m", "started": "", "git": "x", "dirty": False,
                     "cases_sha1": "", "prompt_sha1": "", "trials": 1},
                    summarize([c], [graded(c, "最终答案：42")]))
    assert "答案不公开" in report and "pass^" not in report
    assert "| 难度:hard | 1 | — | — |" in report and "0/1" not in report


# ================================================================ 提交文件
def test_导出提交文件_每题取第一次_按题号排(tmp_path):
    rows = [
        {"case_id": "dab-12", "trial": 2, "final_answer": "b", "answer": "…最终答案：b"},
        {"case_id": "dab-12", "trial": 1, "final_answer": "a", "answer": "…最终答案：a"},
        {"case_id": "dab-5", "trial": 1, "final_answer": "", "answer": "步数耗尽"},
    ]
    (tmp_path / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    lines = [json.loads(line) for line in export(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert [(r["task_id"], r["agent_answer"]) for r in lines] == [("5", ""), ("12", "a")]
    assert set(lines[0]) == {"task_id", "agent_answer", "reasoning_trace"}


def test_补跑只填没写出答案的题_不覆盖已有答案(tmp_path):
    first, retry = tmp_path / "a", tmp_path / "b"
    for d, rows in ((first, [{"case_id": "dab-1", "trial": 1, "answer": "最终答案：x"},
                             {"case_id": "dab-2", "trial": 1, "answer": "已达到最大步数"}]),
                    (retry, [{"case_id": "dab-1", "trial": 1, "answer": "最终答案：y"},
                             {"case_id": "dab-2", "trial": 1, "answer": "最终答案：z"}])):
        d.mkdir()
        (d / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    lines = [json.loads(line) for line in export(first, retry).read_text(encoding="utf-8").splitlines()]
    assert [(r["task_id"], r["agent_answer"]) for r in lines] == [("1", "x"), ("2", "z")]
