"""用 BIRD 官方的判分脚本给一次评测打分（EX），得到能和公开榜单比的分数。

    python -m evals.bird.official evals/runs/<一次 bird_financial 的运行目录>

报告里「只看最后一条 SQL」是我们照 BIRD 的规则自己算的，这里是真拿官方的 evaluation_ex.py 跑。
两个数对不上，差别就是我们的判分器和官方的差别：官方要求去重后的结果集合完全相等 ——
数值没有容差（我们按小数位数容差），列顺序要一样，**连 Python 类型都要一样**：
标准 SQL 里常写 CAST(... AS REAL)，查出来是 float；Agent 写 100.0 * ...，查出来是 Decimal。
430.45454545454544 和 Decimal('430.4545454545454545') 前 15 位相同，set 比较照样不相等。
提交轮那次运行里，我们判对、官方判错的 10 次有 9 次是这个（另 1 次是我们判分器的容差 bug，已修）。

提交哪条 SQL：官方一道题只收一条，Agent 却会跑好几条（探查、查答案、核对）。
题库有提交轮的，交提交轮那条；没有的（或者提交轮没跑出 SQL）交最后一条执行成功的
（Trial.official_sql）。每个 trial 各交一份，报每份的 EX 和平均。
标准答案用题目里的第一条 gold_sql，也就是 BIRD 原版（以后补的替代写法排在后面，官方不认）。

官方脚本要先下载到 data/bird/official/（见 README「BIRD」一节）。只改了一处：
数据库连接写死在 evaluation_utils.connect_postgresql 里（官方 README 让用户自己改），
这里复制一份到临时目录，把连接换成我们的只读账号再加上 search_path。判分逻辑一个字不动。

官方脚本的三个坑：
- 预测和标准答案按文件里的**顺序**配对，不看 key（mini_dev issue #41），
  key 排了序就会静默地几乎零分。所以预测文件按题库顺序写，不能用 sort_keys。
- 读文件没指定编码，Windows 上默认按 GBK 读，SQL 里有中文就会炸。子进程开 UTF-8 模式。
- 连不上库、SQL 出错一律静默记 0 分。所以先自检：把标准答案本身当预测交上去，必须 100 分。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from data_agent.domains import get_domain
from data_agent.settings import Settings

from ..cases import load_cases

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "data" / "bird" / "official"
SCRIPTS = ("evaluation_ex.py", "evaluation_utils.py")
DB_ID = "financial"
SEPARATOR = "\t----- bird -----\t"      # 官方预测文件里 SQL 和库名之间的分隔
LEVELS = ("simple", "moderate", "challenging", "total")

# 官方脚本里写死的连接串，替换前先核对，官方改了脚本就停下来让人看
OFFICIAL_CONNECT = '"dbname=bird user=postgres host=localhost password=li123911 port=5432"'
OUR_CONNECT = 'os.environ["BIRD_EVAL_DSN"], options="-c search_path={schema}"'


def export(run_dir: Path) -> tuple[Path, list[int]]:
    """把一次运行写成官方要的三种文件，返回目录和有哪几个 trial。

    gold.sql          一行一题：SQL<TAB>库名
    difficulty.jsonl  一行一题：{"difficulty": ...}，官方按它分难度统计
    predict_<k>.json  第 k 个 trial 的预测：{"0": "SQL<分隔>库名", ...}
    predict_gold.json 标准答案本身，自检用
    """
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cases = load_cases(meta["cases"]).graded_cases
    by_case: dict[str, dict[int, str]] = {}
    for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        # 早先的运行没有 official_sql 这个字段，那时也没有提交轮
        by_case.setdefault(t["case_id"], {})[t["trial"]] = t.get("official_sql", t["final_sql"])
    trials = sorted({k for per_case in by_case.values() for k in per_case})

    out = run_dir / "bird_official"
    out.mkdir(exist_ok=True)
    gold_lines, difficulty_lines = [], []
    for c in cases:
        gold = c.gold_sql[0]
        # 官方按行读、按 TAB 切，标准 SQL 里不能有换行和 TAB
        assert "\n" not in gold and "\t" not in gold, c.id
        gold_lines.append(f"{gold}\t{DB_ID}\n")
        level = next(t.split(":", 1)[1] for t in c.tags if t.startswith("难度:"))
        difficulty_lines.append(json.dumps({"difficulty": level}) + "\n")
    (out / "gold.sql").write_text("".join(gold_lines), encoding="utf-8")
    (out / "difficulty.jsonl").write_text("".join(difficulty_lines), encoding="utf-8")

    # 没有成功的 SQL 就交空串：官方执行出错记 0 分，和答错一样
    predictions = {k: [by_case.get(c.id, {}).get(k, "") for c in cases] for k in trials}
    predictions["gold"] = [c.gold_sql[0] for c in cases]
    for k, sqls in predictions.items():
        predict = {str(i): sql + SEPARATOR + DB_ID for i, sql in enumerate(sqls)}
        (out / f"predict_{k}.json").write_text(
            json.dumps(predict, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, trials


def evaluate(out: Path, trial: int | str, schema: str, dsn: str) -> dict[str, float]:
    """跑一次官方 evaluation_ex.py，返回 {simple, moderate, challenging, total} 的 EX（百分数）。"""
    with tempfile.TemporaryDirectory() as tmp:
        _patched_copy(Path(tmp), schema)
        proc = subprocess.run(
            [sys.executable, str(Path(tmp) / "evaluation_ex.py"),
             "--predicted_sql_path", str(out / f"predict_{trial}.json"),
             "--ground_truth_path", str(out / "gold.sql"),
             "--db_root_path", "unused/",          # 只有 SQLite 用得上
             "--diff_json_path", str(out / "difficulty.jsonl"),
             "--sql_dialect", "PostgreSQL",
             "--num_cpus", "4",
             "--output_log_path", str(out / f"ex_{trial}.log")],
            env={**os.environ, "PYTHONUTF8": "1", "BIRD_EVAL_DSN": dsn},
            capture_output=True, text=True, encoding="utf-8",
        )
    # 官方把结果打在一行里：「EX  <simple> <moderate> <challenging> <total>」
    m = re.search(r"^EX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", proc.stdout, re.M)
    if proc.returncode != 0 or m is None:
        sys.exit(f"官方脚本没跑成（trial {trial}）：\n{proc.stdout}\n{proc.stderr}")
    return dict(zip(LEVELS, map(float, m.groups())))


def _patched_copy(dst: Path, schema: str) -> None:
    for name in SCRIPTS:
        if not (OFFICIAL / name).exists():
            sys.exit(f"找不到 {OFFICIAL / name}，先按 README「BIRD」一节下载官方脚本")
        shutil.copy(OFFICIAL / name, dst / name)
    utils = dst / "evaluation_utils.py"
    text = utils.read_text(encoding="utf-8")
    if text.count(OFFICIAL_CONNECT) != 1:
        sys.exit("官方的 connect_postgresql 和预期的不一样，脚本可能更新过，先看一眼再改 OFFICIAL_CONNECT")
    utils.write_text("import os\n" + text.replace(OFFICIAL_CONNECT, OUR_CONNECT.format(schema=schema)),
                     encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="用 BIRD 官方脚本给一次评测打分")
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)

    meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
    schema = get_domain(meta["domain"]).schema
    out, trials = export(args.run_dir)
    dsn = Settings().database_url
    if (check := evaluate(out, "gold", schema, dsn))["total"] != 100:
        sys.exit(f"自检没过：标准答案交上去只得了 {check['total']:.1f} 分，连接或配对有问题，看 {out / 'ex_gold.log'}")
    scores = {k: evaluate(out, k, schema, dsn) for k in trials}
    mean = {lv: sum(s[lv] for s in scores.values()) / len(scores) for lv in LEVELS}
    (out / "ex.json").write_text(json.dumps({"trials": scores, "mean": mean}, indent=2), encoding="utf-8")

    print(f"{'':10}" + "".join(f"{lv:>13}" for lv in LEVELS))
    for k, s in [*scores.items(), ("平均", mean)]:
        label = f"trial {k}" if isinstance(k, int) else k
        print(f"{label:10}" + "".join(f"{s[lv]:12.1f}%" for lv in LEVELS))
    ours = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))["pass@1"]
    mine = f"提交 {ours['提交']:.1%}，" if "提交" in ours else ""
    print(f"\n对照我们的判分器：{mine}只看最后一条 {ours['最后一条']:.1%}，主分数（结果对）{ours['结果对']:.1%}")
    print(f"文件在 {out}")


if __name__ == "__main__":
    main()
