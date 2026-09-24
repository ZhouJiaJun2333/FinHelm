"""命令行入口：python -m evals.run --cases shop --trials 3

常用参数：
    --model qwen3.6-flash     换个模型比一比（token plan 里的都行）
    --only shop-003,shop-011  只跑这几道（调试某道题时用）
    --workers 4               同时跑几个 trial（太多会撞 API 限流）
    --compare last|none|路径   和哪次运行对比，默认和同一题库的上一次比
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_agent.db.connection import Database
from data_agent.prompts import SYSTEM_PROMPT
from data_agent.settings import Settings

from .cases import load_cases
from .graders import compare_results
from .report import compare, render, summarize
from .runner import Trial, run_gold, run_trial

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(__file__).parent / "runs"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="跑评测")
    ap.add_argument("--cases", default="shop")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", help="覆盖 .env 里的 OPENAI_MODEL")
    ap.add_argument("--only", help="逗号分隔的题目 id")
    ap.add_argument("--compare", default="last")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env")
    overrides = {"openai_model": args.model} if args.model else {}
    settings = Settings(**overrides)
    model = settings.openai_model if settings.provider == "openai" else settings.anthropic_model

    only = set(args.only.split(",")) if args.only else None
    case_set = load_cases(args.cases, only)
    if not case_set.cases:
        sys.exit(f"题库 {args.cases} 里没有要跑的题")

    # 标准答案先全跑一遍：标准 SQL 本身有错，要在花钱跑 Agent 之前就发现
    db = Database(settings.database_url, statement_timeout_ms=settings.db_statement_timeout_ms)
    gold = {c.id: run_gold(c, db) for c in case_set.cases}
    # 自检：标准答案和它自己比必须算对。不对说明判分器或者题目的 match 写错了
    for c in case_set.cases:
        for rows in gold[c.id].alternatives:
            if c.match != "empty" and not compare_results(rows, rows, c.match).strict:
                sys.exit(f"{c.id} 的标准答案和它自己比都对不上，先检查判分器或 match 设置")

    started = datetime.now()
    run_dir = RUNS_DIR / f"{started:%Y%m%d-%H%M%S}_{args.cases}_{model}"
    run_dir.mkdir(parents=True)
    meta = {
        "cases": args.cases,
        "cases_sha1": case_set.sha1,
        "only": sorted(only) if only else None,
        "model": model,
        "provider": settings.provider,
        "trials": args.trials,
        "started": f"{started:%Y-%m-%d %H:%M:%S}",
        "prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "max_steps": settings.max_steps,
        **_git(),
    }

    jobs = [(c, i) for c in case_set.cases for i in range(1, args.trials + 1)]
    print(f"{len(case_set.cases)} 道题 × {args.trials} 次 = {len(jobs)} 个 trial，模型 {model}，"
          f"并发 {args.workers}\n记录写在 {run_dir.relative_to(ROOT)}")
    trials: list[Trial] = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(run_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        futures = [pool.submit(run_trial, c, i, settings, db, gold[c.id]) for c, i in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            t = fut.result()
            with lock:
                trials.append(t)
                f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")
                f.flush()
                mark = "✅" if t.answer_ok else "❌"
                print(f"[{done:>3}/{len(jobs)}] {mark} {t.case_id} #{t.trial}  "
                      f"{t.elapsed_s:>5.1f}s  {t.steps} 步  {t.failure}")
    meta["elapsed_s"] = round(time.perf_counter() - t0, 1)

    trials.sort(key=lambda t: (t.case_id, t.trial))
    summary = summarize(case_set.cases, trials)
    prev_dir = _previous_run(args.compare, args.cases, run_dir)
    diff = None
    if prev_dir is not None:
        prev_meta = json.loads((prev_dir / "meta.json").read_text(encoding="utf-8"))
        meta["compared_with"] = prev_dir.name
        if prev_meta.get("cases_sha1") != case_set.sha1:
            meta["compared_with"] += "（⚠️ 题库改过，逐题对比仅供参考）"
        diff = compare(json.loads((prev_dir / "summary.json").read_text(encoding="utf-8")), summary)

    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(meta, summary, diff, trials)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def _git() -> dict[str, object]:
    """记下代码版本：半个月后看到一个分数，得知道它是哪版代码跑出来的。"""
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"git": git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": bool(git("status", "--porcelain"))}


def _previous_run(spec: str, cases: str, current: Path) -> Path | None:
    if spec == "none":
        return None
    if spec != "last":
        p = Path(spec)
        return p if (p / "summary.json").exists() else None
    runs = sorted(
        d for d in RUNS_DIR.glob(f"*_{cases}_*")
        if d != current and (d / "summary.json").exists()
    )
    return runs[-1] if runs else None


if __name__ == "__main__":
    main()
