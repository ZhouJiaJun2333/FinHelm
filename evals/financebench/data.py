"""FinanceBench（Patronus AI）开源的 150 题：美股公司的 10-K / 10-Q / 8-K / 财报电话会，每题标了证据在哪份文档的哪一页。

数据在 data/financebench/（不进 git），从 GitHub 只拉题目和 PDF：

    git clone --depth 1 --filter=blob:none --sparse https://github.com/patronus-ai/financebench.git data/financebench
    cd data/financebench && git sparse-checkout set data pdfs

evidence_page_num 从 0 数，和 PyMuPDF 的页码一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "financebench"
PDFS = DATA / "pdfs"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    doc: str
    question: str
    answer: str
    type: str                               # metrics-generated / domain-relevant / novel-generated
    reasoning: str
    evidence: tuple[tuple[str, int], ...]   # (文档, 页码)


def questions() -> list[Question]:
    out = []
    for line in (DATA / "data" / "financebench_open_source.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        out.append(Question(q["financebench_id"], q["doc_name"], q["question"], q["answer"], q["question_type"],
                            q.get("question_reasoning") or "", tuple((e["doc_name"], e["evidence_page_num"])
                                                                     for e in q["evidence"])))
    return out


def doc_meta() -> dict[str, dict]:
    """文档名 → {company, period, doc_type}，拼进每片的上下文前缀。"""
    out = {}
    for line in (DATA / "data" / "financebench_document_information.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["doc_name"]] = {"company": d["company"], "period": d["doc_period"], "doc_type": d["doc_type"]}
    return out


def pdfs(only: set[str] | None = None) -> list[Path]:
    return sorted(p for p in PDFS.glob("*.pdf") if only is None or p.stem in only)
