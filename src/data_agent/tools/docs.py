"""list_docs / search_docs / read_doc：在挂上的知识库（DOCS_DIRS）里找文档、搜片段、读整页。

做成工具让模型自己决定搜不搜、搜什么、搜几次（Agentic RAG），不在每轮开头先检索一遍塞进提示词。
为什么是三个：
    list_docs    先定到是哪份文档。检索评测里，只在题目那份里找 hit@5 76%，368 份混在一起只有 54%
    search_docs  在一份或几份里搜，返回几个片段（BM25 + 向量 → 重排，按设置）
    read_doc     读整页：片段只是一页的一块，表格常被拆开，要看全表、上下文就读整页
失败分析里剩下的两类错也要靠模型自己：用词对不上（问 capex，报表写 Purchases of PP&E）换说法再搜；
派生指标（速动比率）页上没这个词，搜它的组成项。

页码给模型看的是从 1 数的（和 PDF 阅读器一致），内部从 0 数。都是 rerunnable：只读，清掉了可以再查。
"""

from __future__ import annotations

import difflib
from typing import Sequence

from pydantic import BaseModel, Field

from ..core.tools import Tool, ToolOutput
from ..rag import Collection, SearchSpec

LIST_LIMIT = 100
READ_CHARS = 40_000          # read_doc 一次最多给这么多字符（约 1 万 token），和 read_file 一样
MAX_PAGES = 5

_COLLECTION = "知识库名。只挂了一个库时不用写"


class _DocsTool(Tool):
    rerunnable = True

    def __init__(self, collections: Sequence[Collection]) -> None:
        self.collections = {c.name: c for c in collections}

    def _collection(self, name: str) -> Collection:
        if not name and len(self.collections) == 1:
            return next(iter(self.collections.values()))
        if name in self.collections:
            return self.collections[name]
        names = "、".join(self.collections)
        raise ValueError(f"要写 collection，是这几个之一：{names}" if not name else f"没有叫 {name} 的知识库，有：{names}")

    @staticmethod
    def _doc(collection: Collection, name: str) -> str:
        """模型写的文档名 → 索引里的文档名。大小写不对也认；认不出就给几个像的。"""
        docs = collection.index().docs
        if name in docs:
            return name
        lower = {d.lower(): d for d in docs}
        if name.lower() in lower:
            return lower[name.lower()]
        close = difflib.get_close_matches(name, list(docs), n=5, cutoff=0.5)
        hint = f"是不是：{'、'.join(close)}？" if close else "用 list_docs 查文档名。"
        raise ValueError(f"{collection.name} 里没有文档 {name}。{hint}")


class ListDocsTool(_DocsTool):
    name = "list_docs"
    description = (
        "列出知识库里有哪些文档（文档名、公司、期间、类型、页数）。要在某份文档里搜之前，先用它找到准确的文档名。"
        "contains 按关键词筛（空格隔开的几个词都要出现，不分大小写），比如「3M 2018」。"
    )

    class Args(BaseModel):
        contains: str = Field(default="", description="筛选关键词，比如公司名、年份、10K；不写就列全部")
        collection: str = Field(default="", description=_COLLECTION)

    def run(self, args: Args) -> ToolOutput:
        collection = self._collection(args.collection)
        words = args.contains.lower().split()
        rows = []
        for name, entry in collection.index().docs.items():
            meta = [str(v) for v in entry["meta"].values()]
            text = " ".join([name, *meta]).lower()
            if all(w in text for w in words):
                rows.append(f"- {name}（{'，'.join([*meta, str(entry['pages']) + ' 页'])}）")
        where = f"里包含「{args.contains}」的" if words else "里"
        if not rows:
            return ToolOutput(f"{collection.name} {where}文档一份都没有。换个关键词，或者不写 contains 列全部。",
                              summary=f"list_docs {args.contains}：0 份")
        head = f"{collection.name} {where}文档共 {len(rows)} 份" + (
            f"，只列了前 {LIST_LIMIT} 份（用 contains 缩小范围）" if len(rows) > LIST_LIMIT else "") + "："
        return ToolOutput("\n".join([head, *rows[:LIST_LIMIT]]), summary=f"list_docs {args.contains}：{len(rows)} 份")


class SearchDocsTool(_DocsTool):
    name = "search_docs"
    description = (
        "在知识库里搜和问题相关的片段，返回文档名、页码、章节和原文。知道是哪份文档就用 docs 限定，准得多。"
        "query 用文档里会出现的写法：财报里的科目名、英文原词（问资本开支，报表写的是 Purchases of property, plant and equipment）。"
        "搜不到就换说法再搜；比率、增长率这类派生指标文档里往往没有，分别搜它的组成项。"
        "片段只是一页的一部分，表格看不全、要看上下文就用 read_doc 读整页。"
    )
    max_output_chars = 30_000

    class Args(BaseModel):
        query: str = Field(description="要找的内容，写成文档里会出现的样子")
        docs: list[str] = Field(default_factory=list, description="只在这几份文档里搜（list_docs 给的文档名）；不写就搜整个库")
        top_k: int = Field(default=5, ge=1, le=10, description="返回几个片段")
        collection: str = Field(default="", description=_COLLECTION)

    def __init__(self, collections: Sequence[Collection], search: SearchSpec) -> None:
        super().__init__(collections)
        self.search = search

    def run(self, args: Args) -> ToolOutput:
        collection = self._collection(args.collection)
        docs = [self._doc(collection, d) for d in args.docs] or None
        spec = SearchSpec(self.search.retrievers, self.search.candidates, self.search.rrf_k, self.search.reranker,
                          top_k=args.top_k)
        hits = collection.search(args.query, spec, docs)
        scope = f"（限 {'、'.join(docs)}）" if docs else ""
        if not hits:
            return ToolOutput(f"在 {collection.name}{scope} 里没搜到「{args.query}」。", summary="search_docs：0 条")
        parts = [f"在 {collection.name}{scope} 里搜「{args.query}」，前 {len(hits)} 条（按相关度）："]
        for n, h in enumerate(hits, 1):
            section = " > ".join(h.chunk.section)
            parts.append(f"[{n}] {h.chunk.doc} {_pages(h.chunk.pages)}" + (f"｜{section}" if section else "")
                         + f"\n{h.chunk.text}")
        found = "、".join(dict.fromkeys(f"{h.chunk.doc} {_pages(h.chunk.pages)}" for h in hits))
        return ToolOutput("\n\n".join(parts), summary=f"search_docs「{args.query}」{scope}：{found}")


class ReadDocTool(_DocsTool):
    name = "read_doc"
    description = (
        f"读知识库里一份文档的整页（从 page 开始连着读 pages 页，一次最多 {MAX_PAGES} 页），表格是 Markdown。"
        "search_docs 找到的片段不全（表格被拆开、要看前后文、要核对数字）时用。页码和 search_docs 给的一样，从 1 数。"
    )
    max_output_chars = READ_CHARS + 1000

    class Args(BaseModel):
        doc: str = Field(description="文档名（list_docs / search_docs 给的）")
        page: int = Field(ge=1, description="从第几页开始读（从 1 数）")
        pages: int = Field(default=1, ge=1, le=MAX_PAGES, description="连着读几页")
        collection: str = Field(default="", description=_COLLECTION)

    def run(self, args: Args) -> ToolOutput:
        collection = self._collection(args.collection)
        name = self._doc(collection, args.doc)
        doc = collection.document(name)
        if args.page > doc.pages:
            raise ValueError(f"{name} 只有 {doc.pages} 页，没有第 {args.page} 页。")
        shown: list[str] = []
        size = 0
        last = min(args.page + args.pages - 1, doc.pages)
        for page in range(args.page, last + 1):
            text = f"=== 第 {page} 页 ===\n{doc.page_text(page - 1) or '（这一页没有文字，可能是图片）'}"
            if shown and size + len(text) > READ_CHARS:
                break
            shown.append(text[:READ_CHARS])
            size += len(text)
        end = args.page + len(shown) - 1
        tail = f"\n\n（太长了，只给到第 {end} 页，接着读用 page={end + 1}）" if end < last else ""
        head = f"{name}（共 {doc.pages} 页）："
        return ToolOutput(head + "\n" + "\n\n".join(shown) + tail,
                          summary=f"读了 {name} {_pages(tuple(range(args.page - 1, end)))}")


def _pages(pages: Sequence[int]) -> str:
    """内部从 0 数 → 给模型看的从 1 数。"""
    first, last = pages[0] + 1, pages[-1] + 1
    return f"第 {first} 页" if first == last else f"第 {first}–{last} 页"
