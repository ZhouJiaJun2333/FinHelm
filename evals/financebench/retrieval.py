"""检索层评测：只看证据页有没有被搜出来，不调大模型，不花钱。在项目根目录跑：

    python -m evals.financebench.retrieval                                  默认：structure 分片、只用 BM25
    python -m evals.financebench.retrieval --embedder BAAI/bge-m3 \\
        --search bm25 --search dense --search bm25+dense --search "bm25+dense>BAAI/bge-reranker-v2-m3"
    python -m evals.financebench.retrieval --chunk fixed --chunk page --chunk structure     几种分片一起比

建索引时的配置（--parser --chunk --max-tokens --no-context --embedder）每种组合一个索引，增量同步：
只处理新增、改过、删掉的 PDF（--docs 换了范围，索引也跟着增删），
查询时的配置（--search）在每个索引上都跑一遍。两种范围：
    doc  只在这道题问的那份文档里找（相当于元数据过滤做对了）
    all  全部文档混在一起找（368 份）

指标（按题算，再平均）：
    hit@k     前 k 片覆盖的页里有证据页（有几页证据的，命中一页就算）
    all@5     前 5 片覆盖了全部证据页
    MRR       第一片命中证据页的名次的倒数（前 10 片都没命中记 0）
    doc@1     （all 范围）第一片来自正确的文档
    tokens@5  前 5 片一共多少 token：给模型看的上下文有多长，分片大小不同的时候要一起看

结果写到 evals/runs/<时间>_rag_<标签>/：report.md 和 results.json（每题的名次，失败分析用）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_agent.rag import ChunkSpec, Index, IndexSpec, ParsedCache, SearchSpec
from data_agent.rag.chunk import approx_tokens

from . import data

RUNS = data.ROOT / "evals" / "runs"
RAG = data.ROOT / "data" / "rag"
KS = (1, 3, 5, 10)
RERANKER = "BAAI/bge-reranker-v2-m3"


def parse_search(text: str, top_k: int = 10) -> SearchSpec:
    """「bm25+dense>模型名」：+ 连接几路召回，> 后面是重排模型（只写 > 就用默认的）。"""
    retrievers, _, reranker = text.partition(">")
    return SearchSpec(tuple(retrievers.split("+")), reranker=(reranker or RERANKER) if ">" in text else "",
                      top_k=top_k)


def evaluate(index: Index, qs: list[data.Question], search: SearchSpec, scope: str) -> dict:
    per_q = []
    started = time.perf_counter()
    for q in qs:
        hits = index.search(q.question, search, docs=[q.doc] if scope == "doc" else None)
        gold = set(q.evidence)
        ranks = [i for i, h in enumerate(hits, 1) if any((h.chunk.doc, p) in gold for p in h.chunk.pages)]
        covered = lambda k: {(h.chunk.doc, p) for h in hits[:k] for p in h.chunk.pages}  # noqa: E731
        per_q.append({
            "id": q.id, "type": q.type, "first": ranks[0] if ranks else None,
            "all5": gold <= covered(5), "doc1": bool(hits) and hits[0].chunk.doc == q.doc,
            "tokens5": sum(approx_tokens(h.chunk.search_text) for h in hits[:5]),
            "top": [f"{h.chunk.doc}#{','.join(map(str, h.chunk.pages))}" for h in hits[:5]],
        })
    n = len(per_q) or 1
    summary = {f"hit@{k}": sum(1 for r in per_q if r["first"] and r["first"] <= k) / n for k in KS}
    summary.update({
        "all@5": sum(r["all5"] for r in per_q) / n,
        "MRR": sum(1 / r["first"] for r in per_q if r["first"]) / n,
        "doc@1": sum(r["doc1"] for r in per_q) / n,
        "tokens@5": round(sum(r["tokens5"] for r in per_q) / n),
        "秒/题": round((time.perf_counter() - started) / n, 3),
    })
    by_type = defaultdict(list)
    for r in per_q:
        by_type[r["type"]].append(r)
    summary["按题型 hit@5"] = {t: round(sum(1 for r in rs if r["first"] and r["first"] <= 5) / len(rs), 3)
                              for t, rs in sorted(by_type.items())}
    return {"summary": summary, "questions": per_q}


def main(argv: list[str] | None = None) -> None:
    load_dotenv(data.ROOT / ".env")          # 按项目根目录找：从别处调用时也读得到模型目录（HF_HUB_CACHE）
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", default="pymupdf")
    ap.add_argument("--chunk", action="append", choices=["fixed", "page", "structure"])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-context", action="store_true", help="片前不加「文档名 > 章节」前缀")
    ap.add_argument("--embedder", default="", help="比如 BAAI/bge-m3；不写就只建 BM25")
    ap.add_argument("--search", action="append", help="bm25 / dense / bm25+dense / bm25+dense>重排模型")
    ap.add_argument("--scope", default="doc,all")
    ap.add_argument("--docs", choices=["all", "used"], default="all",
                    help="all = 368 份都建索引；used = 只建题目用到的 84 份（索引会删成这 84 份）")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    qs = data.questions()
    meta = data.doc_meta()
    pdfs = data.pdfs(None if args.docs == "all" else {q.doc for q in qs})
    searches = [parse_search(s) for s in (args.search or ["bm25"])]
    scopes = args.scope.split(",")
    parsed = ParsedCache(RAG / "parsed")

    results = []
    for kind in args.chunk or ["structure"]:
        spec = IndexSpec(args.parser, ChunkSpec(kind, max_tokens=args.max_tokens, contextualize=not args.no_context),
                         args.embedder)
        t = time.perf_counter()
        index = Index.sync(spec, pdfs, RAG / "index", parsed, meta, progress=True)
        print(f"索引 {spec.label}：{len(index.chunks)} 片，{index.changes}，{time.perf_counter() - t:.0f}s")
        for search in searches:
            for scope in scopes:
                r = evaluate(index, qs, search, scope)
                s = r["summary"]
                print(f"  {search.label:32s} {scope:3s}  hit@1 {s['hit@1']:.0%}  hit@5 {s['hit@5']:.0%}  "
                      f"hit@10 {s['hit@10']:.0%}  MRR {s['MRR']:.3f}  tokens@5 {s['tokens@5']}")
                results.append({"index": asdict(spec), "index_label": spec.label, "chunks": len(index.chunks),
                                "search": asdict(search), "search_label": search.label, "scope": scope, **r})
    write(results, args, len(pdfs), len(qs))


def write(results: list[dict], args, n_docs: int, n_q: int) -> None:
    started = datetime.now()
    folder = RUNS / f"{started:%Y%m%d-%H%M%S}_rag_{args.label or 'retrieval'}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# FinanceBench 检索评测{('：' + args.label) if args.label else ''}", "",
             f"- {started:%Y-%m-%d %H:%M}　{n_q} 题　{n_docs} 份文档建索引　解析器 {args.parser}", "",
             "| 索引 | 片数 | 检索 | 范围 | hit@1 | hit@3 | hit@5 | hit@10 | all@5 | MRR | doc@1 | tokens@5 | 秒/题 |",
             "|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['index_label']} | {r['chunks']:,} | {r['search_label']} | {r['scope']} | "
                     f"{pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | {pct(s['hit@10'])} | "
                     f"{pct(s['all@5'])} | {s['MRR']:.3f} | {pct(s['doc@1']) if r['scope'] == 'all' else '—'} | "
                     f"{s['tokens@5']:,} | {s['秒/题']} |")
    lines += ["", "## 按题型 hit@5", ""]
    for r in results:
        lines.append(f"- {r['index_label']} / {r['search_label']} / {r['scope']}：" +
                     "，".join(f"{t} {v:.0%}" for t, v in r["summary"]["按题型 hit@5"].items()))
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{folder / 'report.md'}")


if __name__ == "__main__":
    main()
