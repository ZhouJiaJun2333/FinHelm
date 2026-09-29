"""知识库和 list_docs / search_docs / read_doc：PDF 现造，只用 BM25（不加载模型）。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from data_agent.app import build_application
from data_agent.rag import Collection, IndexSpec, SearchSpec
from data_agent.settings import Settings
from data_agent.tools.docs import ListDocsTool, ReadDocTool, SearchDocsTool

from fakes import ScriptedProvider

BM25 = SearchSpec(("bm25",))


def _pdf(path: Path, pages: list[list[str]]) -> Path:
    """每页几行字。"""
    import pymupdf

    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page(width=612, height=792)
        for i, line in enumerate(lines):
            page.insert_text((57, 100 + i * 14), line, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)
    return path


def _library(root: Path) -> Path:
    docs = root / "reports"
    _pdf(docs / "3M_2018_10K.pdf", [["Net sales grew in every region during the year."],
                                   ["Capital spending was driven by new factories.",
                                    "Purchases of property, plant and equipment were 1,577 million."],
                                   ["Dividends were paid every quarter."]])
    _pdf(docs / "AMD_2022_10K.pdf", [["Revenue from data center products increased."]])
    _pdf(docs / "内部" / "预算说明.pdf", [["Budget plan for property purchases next year."]])
    (docs / "metadata.jsonl").write_text(
        '{"doc": "3M_2018_10K", "company": "3M", "period": 2018, "doc_type": "10k"}\n'
        '{"doc": "AMD_2022_10K", "company": "AMD", "period": 2022, "doc_type": "10k"}\n', encoding="utf-8")
    return docs


@pytest.fixture
def library(tmp_path) -> Collection:
    return Collection("reports", _library(tmp_path), tmp_path / "rag", IndexSpec())


# ================================================================ 知识库
def test_知识库_子目录的文档名带路径_元数据进前缀_索引跟着目录走(library, tmp_path):
    index = library.index()
    assert sorted(index.docs) == ["3M_2018_10K", "AMD_2022_10K", "内部/预算说明"]
    assert index.docs["3M_2018_10K"]["pages"] == 3 and index.docs["3M_2018_10K"]["meta"]["company"] == "3M"
    assert {c.doc for c in index.chunks} == set(index.docs)
    assert next(c for c in index.chunks if c.doc == "3M_2018_10K").context.startswith("3M 2018 10k")
    assert library.folder.is_relative_to(tmp_path / "rag" / "collections")
    assert (library.folder / "docs" / "内部" / "预算说明.jsonl").is_file()

    same_dir = Collection("另一个项目起的名字", library.docs_dir, library.root, IndexSpec())
    assert same_dir.folder == library.folder and same_dir.index() is index, "同一个目录共用一份索引，进程里只加载一次"
    assert library.document("3M_2018_10K").page_text(1).startswith("Capital spending")


# ================================================================ 工具
def test_list_docs_按关键词筛(library):
    tool = ListDocsTool([library])
    out = tool.execute({"contains": "3m 2018"}).content
    assert "共 1 份" in out and "- 3M_2018_10K（3M，2018，10k，3 页）" in out
    assert "共 3 份" in tool.execute({}).content
    assert "一份都没有" in tool.execute({"contains": "tesla"}).content


def test_search_docs_限定文档_页码从1数_名字写错给提示(library):
    tool = SearchDocsTool([library], BM25)
    out = tool.execute({"query": "purchases of property", "docs": ["3m_2018_10k"]})
    assert not out.is_error
    assert "（限 3M_2018_10K）" in out.content and "[1] 3M_2018_10K 第 2 页" in out.content
    assert "内部/预算说明" not in out.content
    assert "3M_2018_10K 第 2 页" in out.summary

    everywhere = tool.execute({"query": "property purchases", "top_k": 3}).content
    assert "内部/预算说明 第 1 页" in everywhere

    wrong = tool.execute({"query": "x", "docs": ["3M_2018_10-K"]})
    assert wrong.is_error and "是不是：3M_2018_10K" in wrong.content


def test_read_doc_读整页_连着读_越界报错(library):
    tool = ReadDocTool([library])
    out = tool.execute({"doc": "3M_2018_10K", "page": 2, "pages": 2}).content
    assert out.startswith("3M_2018_10K（共 3 页）")
    assert "=== 第 2 页 ===\nCapital spending" in out and "=== 第 3 页 ===\nDividends" in out
    assert "第 1 页" not in out
    assert "只有 3 页" in tool.execute({"doc": "3M_2018_10K", "page": 9}).content


def test_挂了两个库_要写collection(library, tmp_path):
    other = Collection("contracts", _library(tmp_path / "b"), tmp_path / "rag", IndexSpec())
    tool = ListDocsTool([library, other])
    assert "要写 collection" in tool.execute({}).content
    assert "共 3 份" in tool.execute({"collection": "contracts"}).content
    assert "没有叫 x 的知识库" in tool.execute({"collection": "x"}).content


# ================================================================ 组装
def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(**{"project_dir": str(tmp_path), "database_url": "", "python_sandbox": False,
                       "r_sandbox": False, "rag_dir": str(tmp_path / "rag"), "rag_embedder": "",
                       "rag_reranker": "", **kw})


def test_配了文档目录才有三个工具_提示词讲怎么搜(tmp_path):
    docs = _library(tmp_path)
    app = build_application(_settings(tmp_path, docs_dirs=str(docs)), llm=ScriptedProvider())
    names = [t.name for t in app.tools]
    assert {"list_docs", "search_docs", "read_doc"} <= set(names)
    assert "知识库里的文档（reports，3 份）" in app.agent.system_prompt
    assert "写明出处（文档名、页码）" in app.agent.system_prompt
    assert not (tmp_path / "rag" / "collections").exists(), "第一次搜的时候才建索引"

    plain = build_application(_settings(tmp_path), llm=ScriptedProvider())
    assert "search_docs" not in [t.name for t in plain.tools] and "知识库" not in plain.agent.system_prompt


def test_文档目录不存在_或者重名(tmp_path):
    with pytest.raises(FileNotFoundError, match="文档目录不存在"):
        build_application(_settings(tmp_path, docs_dirs=str(tmp_path / "nope")), llm=ScriptedProvider())
    a, b = tmp_path / "a" / "docs", tmp_path / "b" / "docs"
    a.mkdir(parents=True), b.mkdir(parents=True)
    with pytest.raises(ValueError, match="同名"):
        build_application(_settings(tmp_path, docs_dirs=f"{a};{b}"), llm=ScriptedProvider())


# ================================================================ 通过 MCP
def test_知识库的MCP服务器_子进程起起来_客户端连上去搜(tmp_path):
    """真的起 python -m data_agent.mcp.server，用我们的客户端走 stdio 协议。只用 BM25，不加载模型。"""
    import os
    import sys

    from data_agent.mcp import McpClient, McpTool, ServerConfig

    docs = _library(tmp_path)
    root = Path(__file__).resolve().parents[1]
    env = {"PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]), "RAG_DIR": str(tmp_path / "rag"),
           "RAG_EMBEDDER": "", "RAG_RERANKER": ""}
    client = McpClient(ServerConfig("finhelm", sys.executable, ("-X", "utf8", "-m", "data_agent.mcp.server",
                                                                "--docs-dir", str(docs)), env), timeout=60).start()
    try:
        assert "知识库里的文档（reports，3 份）" in client.instructions
        tools = {t.remote_name: t for t in (McpTool(client, s) for s in client.list_tools())}
        assert set(tools) == {"list_docs", "search_docs", "read_doc"} and tools["search_docs"].rerunnable
        out = tools["search_docs"].execute({"query": "purchases of property", "docs": ["3M_2018_10K"]})
        assert not out.is_error and "[1] 3M_2018_10K 第 2 页" in out.content
        assert tools["read_doc"].execute({"doc": "3M_2018_10K", "page": 9}).is_error
    finally:
        client.close()
