"""RAG：解析（版面分析）、分片、BM25、索引和检索。不需要模型和网络：PDF 现造，只测 BM25 这一路。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("numpy")

import numpy as np

from data_agent.rag import ChunkSpec, Document, Element, Index, IndexSpec, ParsedCache, SearchSpec
from data_agent.rag import index as index_mod
from data_agent.rag import parsers
from data_agent.rag.embed_cache import EmbeddingCache, text_key
from data_agent.rag.bm25 import BM25, _stem, tokenize
from data_agent.rag.chunk import chunk_document
from data_agent.rag.index import _fuse
from data_agent.rag.parsers import pymupdf_layout


# ================================================================ 解析
def _pdf(path: Path, pages: int = 4) -> Path:
    """几页的 10-K 样子：每页顶上一行页眉、底下页码；第 2 页有一张现金流量表，某一年缺一个数。"""
    import pymupdf

    pdf = pymupdf.open()
    for n in range(pages):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((57, 30), "Table of Contents", fontsize=8)
        page.insert_text((300, 770), str(n + 1), fontsize=8)
        if n == 0:
            page.insert_text((57, 100), "PART II", fontsize=8, fontname="hebo")
            page.insert_text((57, 115), "Item 8. Financial Statements", fontsize=8, fontname="hebo")
            page.insert_text((57, 140), "The company sells tape and adhesives to many customers around the world.",
                             fontsize=8)
            page.insert_text((57, 150), "Sales grew in every region during the year.", fontsize=8)
        if n == 1:
            page.insert_text((57, 100), "Consolidated Statement of Cash Flows", fontsize=8, fontname="hebo")
            rows = [("(Millions)", "2018", "2017", "2016"),
                    ("Depreciation and amortization", "1,488", "1,544", "1,474"),
                    ("Purchases of property, plant and equipment", "(1,577)", "(1,373)", "(1,420)"),
                    ("Other - net", "9", "", "(4)")]
            for i, row in enumerate(rows):
                y = 130 + i * 10
                page.insert_text((57, y), row[0], fontsize=8)
                for x, cell in zip((300, 380, 460), row[1:]):
                    if cell:
                        page.insert_text((x + 40 - pymupdf.get_text_length(cell, fontsize=8), y), cell, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)
    return path


def test_解析_页眉页码去掉_标题有章节路径_表格按列对齐(tmp_path):
    doc = pymupdf_layout.parse(_pdf(tmp_path / "3M_2018_10K.pdf"))
    assert doc.pages == 4 and doc.name == "3M_2018_10K"
    noise = {e.text for e in doc.elements if e.type == "header_footer"}
    assert "Table of Contents" in noise and "2" in noise, "每页重复的页眉、页面最后的孤立页码"

    titles = [e for e in doc.elements if e.type == "title"]
    assert [t.text for t in titles][:2] == ["PART II", "Item 8. Financial Statements"]
    para = next(e for e in doc.elements if e.type == "paragraph")
    assert para.section == ("PART II", "Item 8. Financial Statements")
    assert "tape and adhesives" in para.text and "every region" in para.text, "行距正常的两行合成一段"

    [table] = [e for e in doc.elements if e.type == "table"]
    assert table.page == 1 and table.section[-1] == "Consolidated Statement of Cash Flows"
    assert table.rows[0] == ("(Millions)", "2018", "2017", "2016")
    assert table.rows[2] == ("Purchases of property, plant and equipment", "(1,577)", "(1,373)", "(1,420)")
    assert table.rows[3] == ("Other - net", "9", "", "(4)"), "2017 缺数：按右边缘对列，(4) 不会挪到 2017 下面"
    assert table.text.splitlines()[1] == "|---|---|---|---|"


def test_解析结果缓存_按内容不按文件名(tmp_path, monkeypatch):
    pdf = _pdf(tmp_path / "pdfs" / "X_2018_10K.pdf")
    cache = ParsedCache(tmp_path / "parsed")
    first = cache.get(pdf, meta={"company": "X"})
    assert len(list((tmp_path / "parsed").glob("pymupdf-v*/*.jsonl"))) == 1
    again = cache.get(pdf, meta={"company": "X"})
    assert again.elements == first.elements and again.meta["company"] == "X"

    calls = []
    real = parsers.parse
    monkeypatch.setattr(parsers, "parse", lambda path, parser="pymupdf": calls.append(path.name) or real(path, parser))
    renamed = pdf.rename(pdf.with_name("Y_2018_10K.pdf"))
    moved = cache.get(renamed)
    assert calls == [] and moved.name == "Y_2018_10K", "改名不用重新解析，名字按现在的文件"

    _pdf(pdf, pages=2)                                  # 原来的名字，换了内容
    assert cache.get(pdf).pages == 2 and calls == ["X_2018_10K.pdf"], "同名换了内容要重新解析"


# ================================================================ 分片
def _doc() -> Document:
    sec = ("PART II", "Item 8", "Cash Flows")
    rows = [("(Millions)", "2018")] + [(f"Line item {i}", str(i)) for i in range(40)]
    table = "\n".join(["| (Millions) | 2018 |", "|---|---|"] + [f"| Line item {i} | {i} |" for i in range(40)])
    return Document("3M_2018_10K", [
        Element("header_footer", "Table of Contents", 0),
        Element("title", "Item 8", 0, section=("PART II", "Item 8")),
        Element("paragraph", "Intro text about the statements.", 0, section=("PART II", "Item 8")),
        Element("title", "Cash Flows", 1, section=sec),
        Element("table", table, 1, section=sec, rows=tuple(rows)),
        Element("paragraph", "Notes are an integral part.", 2, section=sec),
    ], pages=3, meta={"company": "3M", "period": 2018, "doc_type": "10k"})


def test_按结构分片_标题跟着内容_表格拆开都带表头_不跨页_带章节前缀():
    chunks = chunk_document(_doc(), ChunkSpec("structure", max_tokens=80))
    assert all("Table of Contents" not in c.text for c in chunks), "页眉页脚不进索引"
    assert chunks[0].text.startswith("Item 8\n\nIntro text"), "标题不单独成片"
    tables = [c for c in chunks if "Line item" in c.text]
    assert len(tables) > 1 and all("| (Millions) | 2018 |" in c.text for c in tables), "长表按行拆，每块带表头"
    assert tables[0].text.startswith("Cash Flows"), "表的标题跟着第一块"
    assert all(len(c.pages) == 1 for c in chunks), "默认不跨页"
    assert chunks[-1].pages == (2,)
    assert tables[0].context == "3M 2018 10k > PART II > Item 8 > Cash Flows"
    assert tables[0].search_text.startswith("3M 2018 10k > ")
    assert not chunk_document(_doc(), ChunkSpec("structure", contextualize=False))[0].context


def test_整页和朴素滑窗():
    pages = chunk_document(_doc(), ChunkSpec("page"))
    assert [c.pages for c in pages] == [(0,), (1,), (2,)]
    windows = chunk_document(_doc(), ChunkSpec("fixed", size=20, overlap=5))
    assert len(windows) > 3 and all(len(c.text.split()) <= 20 for c in windows)
    with pytest.raises(ValueError, match="overlap"):
        ChunkSpec("fixed", size=10, overlap=10)


# ================================================================ BM25
def test_分词_学Lucene的英文分析器():
    assert tokenize("What is the FY2018 capital expenditure (in USD millions)?") == \
        ["fy", "2018", "capital", "expenditure", "usd", "million"]
    assert tokenize("Purchases of PP&E (1,577)") == ["purchase", "pp", "e", "1577"]
    assert [_stem(w) for w in ("companies", "statements", "status", "class", "goes")] == \
        ["company", "statement", "status", "class", "goes"]
    assert tokenize("资本开支") == ["资", "本", "开", "支", "资本", "本开", "开支"]


def test_BM25_按文档过滤():
    bm = BM25(["capital expenditure purchases", "tape adhesives", "capital expenditure capital"])
    assert [i for i, _ in bm.search("capital expenditure", 3)][:2] == [2, 0]
    assert [i for i, _ in bm.search("capital", 3, allowed=[True, True, False])] == [0]


def test_RRF融合_只看名次():
    fused = _fuse({"bm25": [(1, 30.0), (2, 20.0)], "dense": [(2, 0.9), (3, 0.8)]}, k=60)
    assert [i for i, _, _ in fused][0] == 2, "两路都排前面的最靠前"
    assert fused[0][2] == {"bm25": 2, "dense": 1}


# ================================================================ 索引
class FakeEmbedder:
    """不加载模型：向量由文本哈希决定，记下每次编码了哪些文本。"""
    encoded: list[str] = []

    def __init__(self, name: str, max_length: int = 1024) -> None:
        self.name, self.max_length = name, max_length

    def embed(self, texts, *, query=False, progress=False):
        type(self).encoded += list(texts)
        rng = [np.random.default_rng(int(text_key(t)[:8], 16)).standard_normal(8) for t in texts]
        return np.array(rng, dtype=np.float16).reshape(len(texts), 8)


def _fake_docs(monkeypatch, tmp_path, docs: dict[str, Document]) -> list[Path]:
    """PDF 文件只是占位（内容 = 文档正文，用来算哈希），解析直接返回 docs 里的。"""
    monkeypatch.setattr(ParsedCache, "get",
                        lambda self, path, parser="pymupdf", meta=None, sha="": docs[path.stem])
    pdfs = []
    for name, doc in docs.items():
        path = tmp_path / "pdfs" / f"{name}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(e.text for e in doc.elements), encoding="utf-8")
        pdfs.append(path)
    return pdfs


def _sync(spec: IndexSpec, pdfs: list[Path], tmp_path: Path) -> Index:
    return Index.sync(spec, {p.stem: p for p in pdfs}, tmp_path / "index" / spec.folder_name,
                      ParsedCache(tmp_path), tmp_path / "embeddings")


def _one(text: str, name: str) -> Document:
    return Document(name, [Element("paragraph", text, 0)], pages=1)


def test_建索引_再同步没变的不动_按文档过滤检索(tmp_path, monkeypatch):
    docs = {"A_2018_10K": _doc(), "B_2018_10K": _one("Line item capital purchases for company B.", "B_2018_10K")}
    pdfs = _fake_docs(monkeypatch, tmp_path, docs)
    spec = IndexSpec(chunk=ChunkSpec("structure", max_tokens=80))
    index = _sync(spec, pdfs, tmp_path)
    assert sorted(index.changes.added) == ["A_2018_10K", "B_2018_10K"]
    assert [c.id for c in index.chunks] == list(range(len(index.chunks))), "全局 id 连续"

    chunked = []
    monkeypatch.setattr(index_mod, "chunk_document", lambda *a, **k: chunked.append(1) or [])
    again = _sync(spec, pdfs, tmp_path)
    assert not chunked and str(again.changes) == "新增 0、更新 0、删除 0", "没变的不重新分片"
    assert [c.text for c in again.chunks] == [c.text for c in index.chunks]

    search = SearchSpec(("bm25",), top_k=3)
    only_b = index.search("capital purchases", search, docs=["B_2018_10K"])
    assert only_b and {h.chunk.doc for h in only_b} == {"B_2018_10K"}
    with pytest.raises(ValueError, match="没有向量"):
        index.search("x", SearchSpec(("dense",)))


def test_增量同步_增删改_嵌入缓存复用(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "Embedder", FakeEmbedder)
    FakeEmbedder.encoded = []
    spec = IndexSpec(chunk=ChunkSpec("page"), embedder="fake/model")
    docs = {n: _one(f"{n} revenue grew", n) for n in ("A", "B", "C")}
    pdfs = _fake_docs(monkeypatch, tmp_path, docs)
    first = _sync(spec, pdfs, tmp_path)
    assert len(FakeEmbedder.encoded) == 3 and first.vectors.shape == (3, 8)
    vec_a = first.vectors[0].copy()

    # B 换内容、C 删掉、D 新增、A 只是修改时间变了
    docs2 = {"A": docs["A"], "B": _one("B margin fell", "B"), "D": _one("D cash flow", "D")}
    pdfs2 = _fake_docs(monkeypatch, tmp_path, docs2)
    (tmp_path / "pdfs" / "C.pdf").unlink()
    FakeEmbedder.encoded = []
    second = _sync(spec, pdfs2, tmp_path)
    assert (second.changes.added, second.changes.changed, second.changes.removed) == (["D"], ["B"], ["C"])
    assert FakeEmbedder.encoded == ["B\nB margin fell", "D\nD cash flow"], "只编码新文本"
    assert [c.doc for c in second.chunks] == ["A", "B", "D"] and second.vectors.shape == (3, 8)
    assert (second.vectors[0] == vec_a).all()
    folder = next((tmp_path / "index").glob("page+ctx-*"))
    assert sorted(p.name for p in (folder / "docs").iterdir()) == ["A.jsonl", "A.npy", "B.jsonl", "B.npy",
                                                                    "D.jsonl", "D.npy"]

    # 换回 B 的旧内容：片文本和第一次一样，从嵌入缓存拿，不重新编码
    _fake_docs(monkeypatch, tmp_path, {**docs2, "B": docs["B"]})
    FakeEmbedder.encoded = []
    third = _sync(spec, pdfs2, tmp_path)
    assert third.changes.changed == ["B"] and FakeEmbedder.encoded == []
    assert (third.vectors[1] == first.vectors[1]).all()


def test_嵌入缓存_一批里重复的只编码一次_重开还在(tmp_path):
    FakeEmbedder.encoded = []
    cache = EmbeddingCache(tmp_path, FakeEmbedder("BAAI/bge-m3", 512))
    vecs, new = cache.embed(["a", "b", "a"])
    assert new == 2 and FakeEmbedder.encoded == ["a", "b"] and (vecs[0] == vecs[2]).all()
    assert (tmp_path / "BAAI--bge-m3-L512" / "pack-0001.keys").is_file()
    reopened = EmbeddingCache(tmp_path, FakeEmbedder("BAAI/bge-m3", 512))
    again, new = reopened.embed(["b", "c"])
    assert new == 1 and (again[0] == vecs[1]).all() and len(reopened) == 3
    assert len(EmbeddingCache(tmp_path, FakeEmbedder("BAAI/bge-m3", 1024))) == 0, "截断长度不同是另一份缓存"
