"""FinanceBench 端到端评测：四种给模型看文档的做法，同一批题、同一个模型、同一种判分。在项目根目录跑：

    python -m evals.financebench.e2e --dry-run                    不调模型：检查上下文怎么拼、多长，判分器自检
    python -m evals.financebench.e2e                              四种都跑（50 道数值题，4 路并发）
    python -m evals.financebench.e2e --modes rag,oracle --limit 5 先拿便宜的试几道
    python -m evals.financebench.e2e --resume evals/runs/<目录>   断了接着跑（跑完的不重跑）

四种做法（--modes）：
    agentic   Agent + list_docs / search_docs / read_doc + run_python：自己找文档、决定搜什么搜几次（Agentic RAG）
    rag       传统 RAG：拿题目原文在题目那份文档里检索一次（BM25 + bge-m3 → 重排，前 5 片），一次调用回答。
              已知是哪份文档 = 元数据过滤做对了，比 agentic 占便宜（agentic 要自己用 list_docs 找）
    oracle    直接给证据页（数据集标注的那几页，我们自己解析的整页文字）：检索满分时的上限
    fulldoc   整份 10-K 放进上下文（平均约 12 万 token，最大约 29 万）：不检索，看长上下文够不够

题目：metrics-generated 那 50 道（答案都是一个数，都来自 10-K）。另外 100 道答案是文字，要用大模型判，以后再做。
判分见 grade.py：只看最后一行「Final answer: <数>」，容差 max(1%, 标准答案末位的一半)。
单次调用的三种没有工具，要心算；agentic 有 run_python。报告里写明，比较时要记得。

结果写到 evals/runs/<时间>_financebench_e2e[_标签]/：results.jsonl（每做完一题追加一行）、report.md。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_agent.app import build_application
from data_agent.core.events import Event, LLMResponded, ToolStarted
from data_agent.core.messages import Message, Usage
from data_agent.rag import IndexSpec, SearchSpec
from data_agent.rag.chunk import approx_tokens
from data_agent.settings import Settings, build_provider

from ..runner import answer
from . import data
from .grade import grade
from .retrieval import RERANKER, RUNS

MODES = ("agentic", "rag", "oracle", "fulldoc")
EMBEDDER = "BAAI/bge-m3"
TOP_K = 5
MAX_STEPS = 20
WAITS = (10, 30, 60, 120)             # 单次调用出错（网络、限流）后等多久重试

SUFFIX = ("\n\nAnswer in the units the question asks for. End your reply with exactly one line:\n"
          "Final answer: <number>\n(only the number, with % if the question asks for a percentage)")
SYSTEM = ("You are a careful financial analyst. Answer the question using only the document content provided "
          "by the user. Numbers in parentheses in financial statements are negative. Show the figures you used "
          "and your calculation briefly.")

AGENT_SETTINGS = {
    "project_dir": "evals/projects/financebench", "docs_dirs": str(data.PDFS), "database_url": "",
    "python_sandbox": True, "r_sandbox": False, "memory_enabled": False, "ask_user": False,
    "max_steps": MAX_STEPS, "wrap_up": "best_guess",     # 步数用完也交一个答案：和单次调用的三种一样总有答案
}


@dataclass(slots=True)
class Result:
    mode: str
    id: str
    doc: str
    question: str
    gold: str
    reply: str = ""
    ok: bool = False
    got: float | None = None
    note: str = ""
    error: str = ""
    seconds: float = 0.0
    context_tokens: int = 0           # 单次调用：塞进去的文档内容估了多少 token
    usage: dict = field(default_factory=dict)
    steps: int = 0                    # agentic：请求了几次模型
    tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 上下文
def context(mode: str, q: data.Question, collection) -> str:
    """单次调用的三种：给模型看的文档内容。"""
    if mode == "rag":
        hits = collection.search(q.question, SearchSpec(("bm25", "dense"), reranker=RERANKER, top_k=TOP_K), [q.doc])
        return "\n\n".join(f"[Excerpt {n}] {h.chunk.doc}, page {h.chunk.page + 1}"
                           + (f" | {' > '.join(h.chunk.section)}" if h.chunk.section else "") + f"\n{h.chunk.text}"
                           for n, h in enumerate(hits, 1))
    doc = collection.document(q.doc)
    pages = sorted({p for d, p in q.evidence if d == q.doc}) if mode == "oracle" else range(doc.pages)
    return "\n\n".join(f"=== {q.doc}, page {p + 1} ===\n{doc.page_text(p)}" for p in pages)


def prompt(mode: str, q: data.Question, ctx: str) -> str:
    what = {"rag": "Excerpts retrieved from", "oracle": "Relevant pages of", "fulldoc": "Full text of"}[mode]
    return f"{what} the document {q.doc}:\n\n{ctx}\n\n---\n\nQuestion: {q.question}{SUFFIX}"


# ---------------------------------------------------------------- 跑一题
def run_single(mode: str, q: data.Question, settings: Settings, collection) -> Result:
    r = Result(mode, q.id, q.doc, q.question, q.answer)
    started = time.perf_counter()
    ctx = context(mode, q, collection)
    r.context_tokens = approx_tokens(ctx)
    llm = build_provider(settings)
    for n, wait in enumerate((*WAITS, None)):
        try:
            resp = llm.chat(messages=[Message.user(prompt(mode, q, ctx))], system=SYSTEM)
            r.reply, r.usage, r.steps = resp.text, asdict(resp.usage), 1
            break
        except Exception as exc:              # noqa: BLE001 网络、限流：等一会儿再试
            if wait is None:
                raise
            print(f"  {mode} {q.id} 第 {n + 1} 次出错（{type(exc).__name__}），{wait}s 后重试", flush=True)
            time.sleep(wait)
    r.seconds = time.perf_counter() - started
    return r


def run_agentic(q: data.Question, settings: Settings, work: Path) -> Result:
    r = Result("agentic", q.id, q.doc, q.question, q.answer)
    events: list[Event] = []
    started = time.perf_counter()
    app = build_application(settings, on_event=events.append, work_dir=work / q.id)
    try:
        r.reply = answer(app.agent, q.question + SUFFIX)
    finally:
        app.close()
    usage = sum((e.usage for e in events if isinstance(e, LLMResponded)), Usage())
    r.usage, r.seconds = asdict(usage), time.perf_counter() - started
    r.steps = sum(isinstance(e, LLMResponded) for e in events)
    r.tools = [e.name for e in events if isinstance(e, ToolStarted)]
    return r


def run_one(mode: str, q: data.Question, settings: Settings, agent_settings: Settings, collection,
            work: Path) -> Result:
    try:
        r = run_agentic(q, agent_settings, work) if mode == "agentic" else run_single(mode, q, settings, collection)
    except Exception as exc:                  # noqa: BLE001 一题出错不影响别的，记下来，--resume 会重跑
        return Result(mode, q.id, q.doc, q.question, q.answer, error=f"{type(exc).__name__}: {exc}",
                      note=traceback.format_exc(limit=3))
    g = grade(q.answer, r.reply)
    r.ok, r.got, r.note = g.ok, g.got, g.note
    return r


# ---------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> None:
    load_dotenv(data.ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--workers", type=int, default=4, help="同时跑几题（几种做法共用）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前几题（试跑用）")
    ap.add_argument("--only", default="", help="只跑这几题，逗号隔开的 financebench_id")
    ap.add_argument("--label", default="")
    ap.add_argument("--resume", type=Path, help="接着跑这个目录：已经做完（没出错）的不重跑")
    ap.add_argument("--dry-run", action="store_true", help="不调模型：拼一遍上下文、看多长，判分器自检")
    args = ap.parse_args(argv)

    modes = [m for m in args.modes.split(",") if m]
    if bad := set(modes) - set(MODES):
        sys.exit(f"没有这几种做法：{sorted(bad)}，可选 {','.join(MODES)}")
    qs = [q for q in data.questions() if q.type == "metrics-generated"]
    if args.only:
        qs = [q for q in qs if q.id in set(args.only.split(","))]
    if args.limit:
        qs = qs[:args.limit]
    for q in qs:                               # 判分器自检：标准答案原样写成最后一行必须判对
        if not grade(q.answer, f"Final answer: {q.answer}").ok:
            sys.exit(f"{q.id} 的标准答案 {q.answer!r} 原样写上去都判不对，先检查 grade.py")

    collection = data.collection(IndexSpec(embedder=EMBEDDER))
    settings = Settings(database_url="", memory_enabled=False, ask_user=False)
    agent_settings = Settings(**AGENT_SETTINGS)
    if args.dry_run:
        return dry_run(modes, qs, collection, agent_settings)

    folder = args.resume or RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_financebench_e2e{'_' + args.label if args.label else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    results_file = folder / "results.jsonl"
    done = {(d["mode"], d["id"]) for d in _read(results_file) if not d["error"]}
    jobs = [(m, q) for q in qs for m in modes if (m, q.id) not in done]      # 按题交错：几种做法一起往前走
    meta = {"started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "model": settings.openai_model
            if settings.provider == "openai" else settings.anthropic_model, "modes": modes, "questions": len(qs),
            "embedder": EMBEDDER, "reranker": RERANKER, "top_k": TOP_K, "max_steps": MAX_STEPS}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(qs)} 题 × {len(modes)} 种，要跑 {len(jobs)} 个（已完成 {len(done)}），结果在 {folder}", flush=True)

    collection.index(progress=True)            # 先在主线程加载好，不让第一批题一起等
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, m, q, settings, agent_settings, collection, folder / "work")
                   for m, q in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            with lock, results_file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            mark = "出错" if r.error else ("✓" if r.ok else "✗")
            print(f"[{n}/{len(jobs)}] {r.mode:8s} {r.id} {mark} 答 {r.got} / 标准 {r.gold}"
                  f"（{r.seconds:.0f}s）{r.error[:80]}", flush=True)
    write_report(folder, meta, qs, modes)


def dry_run(modes: list[str], qs: list[data.Question], collection, agent_settings: Settings) -> None:
    print(f"{len(qs)} 题，判分器自检通过")
    for mode in modes:
        if mode == "agentic":
            app = build_application(agent_settings)
            try:
                print(f"  agentic：工具 {[t.name for t in app.tools]}，最多 {MAX_STEPS} 步")
            finally:
                app.close()
            continue
        sizes = [approx_tokens(context(mode, q, collection)) for q in qs]
        print(f"  {mode:8s} 上下文平均 {sum(sizes) // len(sizes):,} token，最大 {max(sizes):,}，合计 {sum(sizes):,}")
    print(f"\n示例（rag，{qs[0].id}）：\n{prompt('rag', qs[0], context('rag', qs[0], collection))[:1500]}…")


# ---------------------------------------------------------------- 报告
def _cell(r: dict | None) -> str:
    if r is None:
        return "—"
    if r["error"]:
        return "出错"
    return ("✓" if r["ok"] else "✗") + ("" if r["got"] is None else f" {r['got']:g}")


def _why(r: dict) -> str:
    """答错的那题：出错信息，或者没写最终答案，或者回答最后一行。"""
    if r["error"]:
        return r["error"][:200]
    last = r["reply"].strip().splitlines()[-1][:200] if r["reply"].strip() else "（空回答）"
    return f"{r['note']}｜{last}" if r["note"] else last


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(folder: Path, meta: dict, qs: list[data.Question], modes: list[str]) -> None:
    latest = {(d["mode"], d["id"]): d for d in _read(folder / "results.jsonl")}      # 重跑过的按最后一次
    ids = [q.id for q in qs]
    lines = [f"# FinanceBench 端到端{('：' + folder.name.split('_e2e_')[-1]) if '_e2e_' in folder.name else ''}", "",
             f"- {meta['started']}　模型 {meta['model']}　{len(ids)} 道数值题（metrics-generated）",
             f"- 判分：最后一行「Final answer」，容差 max(1%, 标准答案末位的一半)，不看正负号",
             f"- rag：已知文档，BM25 + {meta['embedder'].split('/')[-1]} → {meta['reranker'].split('/')[-1]}，"
             f"前 {meta['top_k']} 片；agentic 最多 {meta['max_steps']} 步，有 run_python，另外三种没有工具（心算）", "",
             "| 做法 | 答对 | 出错 | 没写最终答案 | 平均输入 token | 平均输出 token | 平均秒数 | 平均步数 |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for m in modes:
        rs = [latest[(m, i)] for i in ids if (m, i) in latest]
        if not rs:
            continue
        n = len(rs)
        prompt_tokens = [r["usage"].get("input", 0) + r["usage"].get("cache_read", 0) for r in rs]
        lines.append(f"| {m} | {sum(r['ok'] for r in rs)}/{n} = {sum(r['ok'] for r in rs) / n:.0%} | "
                     f"{sum(bool(r['error']) for r in rs)} | {sum(r['note'] == '没写最终答案' for r in rs)} | "
                     f"{sum(prompt_tokens) // n:,} | {sum(r['usage'].get('output', 0) for r in rs) // n:,} | "
                     f"{sum(r['seconds'] for r in rs) / n:.0f} | {sum(r['steps'] for r in rs) / n:.1f} |")
    lines += ["", "## 逐题", "", "| 题 | 标准答案 | " + " | ".join(modes) + " |",
              "|:--|--:|" + "--:|" * len(modes)]
    for q in qs:
        cells = [_cell(latest.get((m, q.id))) for m in modes]
        lines.append(f"| {q.id.removeprefix('financebench_id_')} {q.doc} | {q.answer} | " + " | ".join(cells) + " |")
    lines += ["", "## 答错的", ""]
    for m in modes:
        for i in ids:
            r = latest.get((m, i))
            if r and not r["ok"]:
                lines.append(f"- **{m}** {i}（{r['doc']}）标准 {r['gold']}，答 {r['got']}：{_why(r)}")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{folder / 'report.md'}")


if __name__ == "__main__":
    main()
