"""BIRD 官方判分的导出格式。官方脚本出了错只会静默记 0 分，格式错了看不出来，所以这里钉死。不需要数据库。"""

from __future__ import annotations

import json

import pytest

from evals.bird import official
from evals.cases import load_cases

SEP = "\t----- bird -----\t"


def _fake_run(tmp_path, trials):
    (tmp_path / "meta.json").write_text(json.dumps({"cases": "bird_financial", "domain": "financial"}))
    (tmp_path / "trials.jsonl").write_text(
        "".join(json.dumps({"case_id": cid, "trial": k, "final_sql": sql}) + "\n" for cid, k, sql in trials),
        encoding="utf-8")
    return tmp_path


def test_预测按题库顺序写_不排序_缺的交空串(tmp_path):
    cases = load_cases("bird_financial").graded_cases
    # trials.jsonl 里的顺序是乱的（并发跑完的先写），导出要按题库顺序
    run = _fake_run(tmp_path, [(cases[11].id, 1, "SELECT 11"), (cases[0].id, 1, "SELECT 0 -- 中文")])
    out, trials = official.export(run)
    assert trials == [1]

    predict = json.loads((out / "predict_1.json").read_text(encoding="utf-8"))
    # 官方按顺序配对、不看 key：key 必须是 0,1,2… 且就按这个顺序出现（"10" 在 "9" 后面，不是字典序）
    assert list(predict) == [str(i) for i in range(len(cases))]
    assert predict["0"] == "SELECT 0 -- 中文" + SEP + "financial"
    assert predict["11"] == "SELECT 11" + SEP + "financial"
    assert predict["1"] == SEP + "financial", "没有成功的 SQL 交空串，官方执行出错记 0 分"


def test_标准答案和难度一行一题_自检文件交的就是标准答案(tmp_path):
    cases = load_cases("bird_financial").graded_cases
    out, _ = official.export(_fake_run(tmp_path, [(cases[0].id, 1, "SELECT 1")]))

    gold = (out / "gold.sql").read_text(encoding="utf-8").splitlines()
    assert gold == [f"{c.gold_sql[0]}\tfinancial" for c in cases]
    levels = [json.loads(l)["difficulty"] for l in (out / "difficulty.jsonl").read_text().splitlines()]
    assert len(levels) == len(cases) and set(levels) == {"simple", "moderate", "challenging"}
    self_check = json.loads((out / "predict_gold.json").read_text(encoding="utf-8"))
    assert [v.split(SEP)[0] for v in self_check.values()] == [c.gold_sql[0] for c in cases]


def test_官方脚本只换连接那一处(tmp_path, monkeypatch):
    src = tmp_path / "official"
    src.mkdir()
    (src / "evaluation_ex.py").write_text("# ex")
    (src / "evaluation_utils.py").write_text(f"import psycopg2\ndb = psycopg2.connect(\n    {official.OFFICIAL_CONNECT}\n)\n")
    monkeypatch.setattr(official, "OFFICIAL", src)

    dst = tmp_path / "patched"
    dst.mkdir()
    official._patched_copy(dst, "financial")
    text = (dst / "evaluation_utils.py").read_text()
    assert text.startswith("import os\n")
    assert 'os.environ["BIRD_EVAL_DSN"], options="-c search_path=financial"' in text
    assert "li123911" not in text

    # 官方改了脚本、连接那行对不上：停下来，不要猜着改
    (src / "evaluation_utils.py").write_text("db = psycopg2.connect(host='x')\n")
    with pytest.raises(SystemExit):
        official._patched_copy(dst, "financial")
