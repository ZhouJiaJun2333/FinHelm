"""命令行入口：python -m evals.run --cases shop --trials 3

常用参数：
    --model qwen3.6-flash     换个模型比一比（token plan 里的都行）
    --only shop-003,shop-011  只跑这几道（调试某道题时用）
    --workers 4               同时跑几个 trial（太多会撞 API 限流）
    --compare last|none|路径   和哪次运行对比，默认和同一题库、同一标签的上一次比
    --set KEY=VALUE           临时改一项配置（可以写多次），优先级高于会话自带的 settings
    --label 名字              给这次运行起个名，写进目录名和报告，比较几种配置时用
    --rebuild 目录            不跑模型，用存下的 trials/sessions.jsonl 重新出报告
                              （报告那一步崩了，或者改了报告格式想重出一遍）
    --regrade 目录            不跑模型，用存下的 SQL 和回答按现在的判分规则重新判分、出报告
                              （改了判分器 / 补了标准答案，不用再花钱跑一遍）。只支持单题库

比较几种配置（几个进程可以同时跑）：
    python -m evals.run --cases shop_multi --trials 2 --label 现状
    python -m evals.run --cases shop_multi --trials 2 --label 不清理 --set context_clear_trigger_tokens=1000000000

多轮题库（python -m evals.run --cases shop_multi --trials 2）：一个 trial = 整段会话跑一遍，
--only 填会话 id。
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
from data_agent.domains import get_domain
from data_agent.app import build_application
from data_agent.settings import Settings

from .cases import CaseSet, load_cases
from .graders import check_answer, check_values, compare_results
from .report import compare, render, summarize, summarize_sessions
from .runner import SessionTrial, Trial, grade, run_gold, run_session, run_trial

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
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="sets")
    ap.add_argument("--label", default="")
    ap.add_argument("--rebuild", type=Path, metavar="目录")
    ap.add_argument("--regrade", type=Path, metavar="目录")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.rebuild or args.regrade:
        load_dotenv(ROOT / ".env")
        rebuild(args.rebuild or args.regrade, args.compare, regrade=bool(args.regrade))
        return
    load_dotenv(ROOT / ".env")
    only = set(args.only.split(",")) if args.only else None
    case_set = load_cases(args.cases, only)
    if not case_set.cases and not case_set.sessions:
        sys.exit(f"题库 {args.cases} 里没有要跑的题")
    if unknown := set(case_set.settings) - set(Settings.model_fields):
        sys.exit(f"题库 {args.cases} 的 settings 里有不认识的配置：{sorted(unknown)}")

    forced = _parse_sets(args.sets)
    overrides = {"openai_model": args.model} if args.model else {}
    # 优先级：.env < 题库级配置（比如 BIRD 用 financial 场景包）< --model / --set
    settings = Settings(**{**case_set.settings, **overrides, **forced})
    domain = get_domain(settings.domain)
    model = settings.openai_model if settings.provider == "openai" else settings.anthropic_model
    graded = case_set.graded_cases
    # 会话覆盖的配置名写错了，model_copy 会悄悄忽略 —— 门槛没调低，整段会话就白跑了
    for s in case_set.sessions:
        if unknown := set(s.settings) - set(Settings.model_fields):
            sys.exit(f"{s.id} 的 settings 里有不认识的配置：{sorted(unknown)}")

    # 标准答案先全跑一遍：标准 SQL 本身有错，要在花钱跑 Agent 之前就发现
    db = _database(settings, domain.schema, graded)
    gold = {c.id: run_gold(c, db) for c in graded}
    # 自检：标准答案和它自己比必须算对。不对说明判分器或者题目的 match 写错了
    for c in graded:
        if c.no_sql:
            # 标准值写进回答里必须判对（容差、正负号的处理有问题的话这里就能发现）
            if c.gold_values and not check_values(c.gold_values, " ".join(map(str, c.gold_values))).ok:
                sys.exit(f"{c.id} 的 gold_values 原样写进回答都判不对，先检查判分器")
            continue
        for rows in gold[c.id].alternatives:
            if c.match != "empty" and not compare_results(rows, rows, c.match).strict:
                sys.exit(f"{c.id} 的标准答案和它自己比都对不上，先检查判分器或 match 设置")
        # 只看回答的题：answer_sql 得查出能核对的数，不然这题谁都答不对
        if c.match == "answer" and check_answer(gold[c.id].answer or [], "").ok is not False:
            sys.exit(f"{c.id} 的 answer_sql 查不出可核对的数字（空、太多行或没有数）")

    started = datetime.now()
    label = f"_{args.label}" if args.label else ""
    run_dir = RUNS_DIR / f"{started:%Y%m%d-%H%M%S}_{args.cases}_{model}{label}"
    run_dir.mkdir(parents=True)
    meta = {
        "cases": args.cases,
        "cases_sha1": case_set.sha1,
        "only": sorted(only) if only else None,
        "model": model,
        "provider": settings.provider,
        # 同一个模型在不同端点上缓存规则、限流都不一样（百炼 vs DeepSeek 官方）
        "base_url": settings.openai_base_url if settings.provider == "openai" else None,
        "label": args.label,
        "overrides": forced,
        "trials": args.trials,
        "started": f"{started:%Y-%m-%d %H:%M:%S}",
        "domain": domain.name,
        "prompt_sha1": prompt_fingerprint(settings),
        "max_steps": settings.max_steps,
        **_git(),
    }
    # 先写一份：跑了半小时、最后出报告时崩了，有 meta + jsonl 就能 --rebuild
    _write_json(run_dir / "meta.json", meta)

    multi = bool(case_set.sessions)
    units = case_set.sessions if multi else case_set.cases
    jobs = [(u, i) for u in units for i in range(1, args.trials + 1)]
    print(f"{len(units)} {'段会话' if multi else '道题'} × {args.trials} 次 = {len(jobs)} 个 trial，"
          f"模型 {model}，并发 {args.workers}\n记录写在 {run_dir.relative_to(ROOT)}")
    results: list = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(run_dir / ("sessions.jsonl" if multi else "trials.jsonl"), "w", encoding="utf-8") as f:
        futures = [
            pool.submit(run_session, u, i, settings, db, gold, forced) if multi
            else pool.submit(run_trial, u, i, settings, db, gold[u.id], case_set.submit, run_dir / "work")
            for u, i in jobs
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with lock:
                results.append(r)
                f.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(f"[{done:>3}/{len(jobs)}] {_progress(r)}")
    meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
    finish(run_dir, meta, case_set, results, args.compare)


def finish(run_dir: Path, meta: dict, case_set: CaseSet, results: list, compare_spec: str) -> None:
    """汇总、和上一次比、写 summary.json 和 report.md。"""
    if case_set.sessions:
        results.sort(key=lambda st: (st.session_id, st.trial))
        summary = summarize_sessions(case_set, results)
        trials = [t for st in results for t in st.turns if t.graded]
    else:
        trials = sorted(results, key=lambda t: (t.case_id, t.trial))
        summary = summarize(case_set.cases, trials)
    prev_dir = _previous_run(compare_spec, meta["cases"], meta.get("label", ""), run_dir)
    diff = None
    if prev_dir is not None:
        prev_meta = json.loads((prev_dir / "meta.json").read_text(encoding="utf-8"))
        meta["compared_with"] = prev_dir.name
        if prev_meta.get("cases_sha1") != case_set.sha1:
            meta["compared_with"] += "（⚠️ 题库改过，逐题对比仅供参考）"
        diff = compare(json.loads((prev_dir / "summary.json").read_text(encoding="utf-8")), summary)

    _write_json(run_dir / "meta.json", meta)
    _write_json(run_dir / "summary.json", summary)
    report = render(meta, summary, diff, trials)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def rebuild(run_dir: Path, compare_spec: str, regrade: bool = False) -> None:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    only = set(meta["only"]) if meta.get("only") else None
    case_set = load_cases(meta["cases"], only)
    if case_set.sha1 != meta["cases_sha1"]:
        print(f"⚠️ 题库 {meta['cases']} 在这次运行之后改过，按现在的题库汇总")
    if case_set.sessions:
        rows = (run_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
        results = [SessionTrial.from_dict(json.loads(r)) for r in rows if r.strip()]
    else:
        rows = (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        results = [Trial.from_dict(json.loads(r)) for r in rows if r.strip()]
    if regrade:
        _regrade(run_dir, meta, case_set, results)
    meta.pop("compared_with", None)
    finish(run_dir, meta, case_set, results, compare_spec)


def _regrade(run_dir: Path, meta: dict, case_set: CaseSet, trials: list[Trial]) -> None:
    """按现在的判分规则和题库重判，覆盖 trials.jsonl。模型的输出（SQL、回答）一个字不动。"""
    if case_set.sessions:
        sys.exit("--regrade 只支持单题库：多轮会话的判分依赖每轮当时的上下文")
    settings = Settings(**case_set.settings)
    db = _database(settings, get_domain(meta.get("domain", settings.domain)).schema, case_set.cases)
    cases = {c.id: c for c in case_set.cases}
    gold = {cid: run_gold(c, db) for cid, c in cases.items()}
    for t in trials:
        grade(t, cases[t.case_id], db, gold[t.case_id])
    with open(run_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")
    meta["regraded"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}（题库 {case_set.sha1}，判分器 {_git()['git']}）"
    print(f"按现在的规则重判了 {len(trials)} 个 trial")


def prompt_fingerprint(settings: Settings) -> str:
    """系统提示词 + 工具定义 + 收尾提示的指纹，按这次真正会组装出来的 Agent 算。

    提示词按注册了哪些工具拼（沙箱、view_image 开没开），只按场景包算的话，
    开关不同的两次运行指纹一样，报告里就看不出提示词变过。不启动沙箱（第一次调用才起容器）。
    """
    app = build_application(settings)
    try:
        text = (app.agent.system_prompt + json.dumps(app.tools.schemas(), ensure_ascii=False, sort_keys=True)
                + app.agent.wrap_up_prompt)
    finally:
        app.close()
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _database(settings: Settings, schema: str | None, cases: list) -> Database | None:
    """有 SQL 题才连库：research 场景不连数据库，数据库没开也能跑。"""
    if not any(c.gold_sql or c.answer_sql for c in cases):
        return None
    return Database(settings.database_url, statement_timeout_ms=settings.db_statement_timeout_ms,
                    search_path=schema)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_sets(items: list[str]) -> dict[str, object]:
    """--set KEY=VALUE → {key: value}。值按 JSON 解析（数字、true/false），解析不了就当字符串。"""
    out: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip().lower()
        if not sep or key not in Settings.model_fields:
            sys.exit(f"--set {item}：要写成 KEY=VALUE，KEY 是 Settings 里的字段名")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _progress(r: Trial | SessionTrial) -> str:
    if isinstance(r, Trial):
        mark = "·" if not r.graded else "✅" if r.answer_ok else "❌"
        return f"{mark} {r.case_id} #{r.trial}  {r.elapsed_s:>5.1f}s  {r.steps} 步  {r.failure}"
    graded = [t for t in r.turns if t.graded]
    marks = "".join("·" if not t.graded else "✅" if t.answer_ok else "❌" for t in r.turns)
    return (f"{r.session_id} #{r.trial}  {marks}  {sum(t.answer_ok for t in graded)}/{len(graded)} 轮对  "
            f"{r.elapsed_s:>5.1f}s  整理 {len(r.edits)} 次  {r.error}")


def _git() -> dict[str, object]:
    """记下代码版本：半个月后看到一个分数，得知道它是哪版代码跑出来的。"""
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"git": git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": bool(git("status", "--porcelain"))}


def _previous_run(spec: str, cases: str, label: str, current: Path) -> Path | None:
    if spec == "none":
        return None
    if spec != "last":
        p = Path(spec)
        return p if (p / "summary.json").exists() else None
    # 不能只靠目录名：「*_shop_*」也会匹配到 shop_multi 的运行
    runs = sorted(
        d for d in RUNS_DIR.glob(f"*_{cases}_*")
        if d != current and (d / "summary.json").exists()
        and _same_kind(json.loads((d / "meta.json").read_text(encoding="utf-8")), cases, label)
    )
    return runs[-1] if runs else None


def _same_kind(meta: dict, cases: str, label: str) -> bool:
    return meta.get("cases") == cases and meta.get("label", "") == label


if __name__ == "__main__":
    main()
