"""检索失败分析：证据页没进前 k 的题，错在哪一环。在项目根目录跑：

    python -m evals.financebench.failures                 默认：bge-m3 结构化分片索引，BM25+向量+重排，本文档范围
    python -m evals.financebench.failures --k 5

对每道没命中的题，逐环检查：
    解析    证据页的文字在不在我们解析出来的这一页里（拿 FinanceBench 给的证据原文比，按词算覆盖率）
    页码    证据原文其实在前一页 / 后一页（数据集标注和 PDF 页码差一位）
    召回    BM25、向量各自在这份文档里把证据页排第几；两路的前 50 都没有 → 召回失败
    融合    某一路前 50 里有，融合后截到 50 条时被挤掉了
    重排    在候选里，重排后掉到了 k 名以外

结果写到 evals/runs/<时间>_rag_失败分析/：report.md（每题一段，带题目、证据原文、我们排在前面的是什么）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv

from data_agent.rag import Index, IndexSpec, SearchSpec
from data_agent.rag.bm25 import tokenize
from data_agent.rag.chunk import approx_tokens

from . import data
from .retrieval import RERANKER, RUNS

CANDIDATES = 50


def evidence_texts() -> dict[str, list[dict]]:
    out = {}
    for line in (data.DATA / "data" / "financebench_open_source.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        out[q["financebench_id"]] = q["evidence"]
    return out


def coverage(evidence: str, page: str) -> float:
    """证据原文的词有多少出现在这一页里（去重后按词算）。"""
    want = set(tokenize(evidence))
    return len(want & set(tokenize(page))) / len(want) if want else 1.0


def gold_rank(chunks_in_order, gold: set[tuple[str, int]]) -> int | None:
    for rank, c in enumerate(chunks_in_order, 1):
        if any((c.doc, p) in gold for p in c.pages):
            return rank
    return None


def main(argv: list[str] | None = None) -> None:
    load_dotenv(data.ROOT / ".env")          # 按项目根目录找：从别处调用时也读得到模型目录（HF_HUB_CACHE）
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", default="BAAI/bge-m3")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    index = data.collection(IndexSpec(embedder=args.embedder)).index(progress=True)
    print(f"索引 {index.spec.label}（{len(index.chunks)} 片）")
    pages: dict[tuple[str, int], str] = {}
    for c in index.chunks:
        key = (c.doc, c.page)
        pages[key] = pages.get(key, "") + "\n" + c.text
    evidence = evidence_texts()

    rows = []
    for q in data.questions():
        gold = set(q.evidence)
        full = SearchSpec(("bm25", "dense"), candidates=CANDIDATES, reranker=RERANKER, top_k=CANDIDATES)
        final = index.search(q.question, full, docs=[q.doc])
        rank = gold_rank([h.chunk for h in final], gold)
        if rank is not None and rank <= args.k:
            continue
        per_doc = len(index._by_doc.get(q.doc, []))
        bm = index.search(q.question, SearchSpec(("bm25",), candidates=per_doc, top_k=per_doc), docs=[q.doc])
        dn = index.search(q.question, SearchSpec(("dense",), candidates=per_doc, top_k=per_doc), docs=[q.doc])
        fused = index.search(q.question, SearchSpec(("bm25", "dense"), candidates=CANDIDATES, top_k=CANDIDATES),
                             docs=[q.doc])
        ranks = {"bm25": gold_rank([h.chunk for h in bm], gold), "dense": gold_rank([h.chunk for h in dn], gold),
                 "fused": gold_rank([h.chunk for h in fused], gold), "rerank": rank}
        # 解析：证据原文在我们这一页里的覆盖率；前后页更高就是页码差一位
        checks = []
        for ev in evidence[q.id]:
            doc, page = ev["doc_name"], ev["evidence_page_num"]
            here = coverage(ev["evidence_text"], pages.get((doc, page), ""))
            near = {d: coverage(ev["evidence_text"], pages.get((doc, page + d), "")) for d in (-1, 1)}
            checks.append({"page": page, "here": here, "near": near, "text": ev["evidence_text"]})
        stage = _stage(ranks, checks)
        rows.append({"q": q, "ranks": ranks, "checks": checks, "stage": stage,
                     "top": [(h.chunk, h.score) for h in final[:3]]})
        print(f"  {q.id} {stage:12s} {ranks}")
    write(rows, index, args.k)


def _stage(ranks: dict, checks: list[dict]) -> str:
    if all(c["here"] < 0.5 for c in checks):
        if any(max(c["near"].values()) >= 0.8 for c in checks):
            return "页码差一位"
        return "解析"
    if all(r is None or r > CANDIDATES for r in (ranks["bm25"], ranks["dense"])):
        return "召回"
    if ranks["fused"] is None:
        return "融合截断"
    return "重排"


def write(rows: list[dict], index: Index, k: int) -> None:
    folder = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_rag_失败分析"
    folder.mkdir(parents=True, exist_ok=True)
    stages = Counter(r["stage"] for r in rows)
    types = Counter((r["stage"], r["q"].type) for r in rows)
    out = [f"# FinanceBench 检索失败分析（证据页不在前 {k}）", "",
           f"- 索引：{index.spec.label}　检索：BM25 + 向量 → 重排，本文档范围　没命中 {len(rows)} / 150 题", "",
           "| 错在哪一环 | 题数 | 题型 |", "|:--|--:|:--|"]
    for stage, n in stages.most_common():
        out.append(f"| {stage} | {n} | " + "，".join(f"{t} {c}" for (s, t), c in types.items() if s == stage) + " |")
    out += ["", "名次都是在这份文档里的第几片（None = 没排进来）：bm25 / dense 在全文档里排，fused 是融合后的前 50，"
            "rerank 是重排后。", ""]
    for r in sorted(rows, key=lambda r: r["stage"]):
        q = r["q"]
        out += [f"## {q.id}（{r['stage']}，{q.type}）", "", f"**问**：{q.question}", "", f"**答**：{q.answer[:300]}", "",
                f"- 名次：{r['ranks']}"]
        for c in r["checks"]:
            near = "，".join(f"{d:+d} 页 {v:.0%}" for d, v in c["near"].items())
            out.append(f"- 证据第 {c['page']} 页：原文词覆盖 {c['here']:.0%}（{near}）")
            out.append(f"  - 证据原文：{' '.join(c['text'].split())[:400]}")
        out.append("- 我们排在前面的：")
        for chunk, score in r["top"]:
            snippet = " ".join(chunk.text.split())[:220]
            out.append(f"  - 第 {','.join(map(str, chunk.pages))} 页（{score:.3f}，{approx_tokens(chunk.text)} token）"
                       f"{' > '.join(chunk.section[-1:])}：{snippet}")
        out.append("")
    (folder / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n{dict(stages)}\n报告：{folder / 'report.md'}")


if __name__ == "__main__":
    main()
