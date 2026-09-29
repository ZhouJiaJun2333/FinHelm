"""分片：Document → Chunk。三种做法，评测时对比：

    fixed      朴素基线：整页文字按词滑窗切（很多教程里的 RecursiveCharacterTextSplitter 就是这个思路）
    page       一页一片（页眉页脚去掉）
    structure  按结构切（学 Docling 的 HybridChunker）：同一章节里相邻的元素合并，直到接近 max_tokens；
               换章节就另起一片；表格不从中间切，太长的按行拆、每一块都带上表头；超长的段落按句子拆

默认片不跨页（cross_page=False）：命中哪一片就知道是哪一页，引用、评测都按页。

上下文前缀（contextualize，学 Anthropic 的 Contextual Retrieval 的简化版）：每片前面加上
「文档名 > 章节路径」。「(1,577)」这种数单独看不知道是什么，带上「3M 2018 10-K > 现金流量表」才搜得到。
前缀参与检索（BM25、向量、重排都用 search_text），给模型看的时候也带着。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from .document import Document, Element

TokenCounter = Callable[[str], int]

_WORD = re.compile(r"\S+\s*")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+")


def approx_tokens(text: str) -> int:
    """没有分词器时的估算：英文约 1.3 token 一个词，中文一字一个。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return int(len(text.split()) * 1.3) + cjk


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    kind: Literal["fixed", "page", "structure"] = "structure"
    max_tokens: int = 512           # structure：一片最多多少 token
    size: int = 200                 # fixed：每片多少个词
    overlap: int = 50               # fixed：相邻两片重叠多少个词
    contextualize: bool = True      # 每片前面加「文档名 > 章节路径」
    cross_page: bool = False        # structure：允许一片跨页

    def __post_init__(self) -> None:
        if self.kind == "fixed" and not 0 <= self.overlap < self.size:
            raise ValueError(f"overlap 要小于 size（现在 size={self.size}, overlap={self.overlap}）")

    @property
    def label(self) -> str:
        base = {"fixed": f"fixed{self.size}-{self.overlap}", "page": "page",
                "structure": f"structure{self.max_tokens}"}[self.kind]
        return base + ("+ctx" if self.contextualize else "") + ("+xpage" if self.cross_page else "")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: int
    doc: str
    pages: tuple[int, ...]          # 覆盖了哪几页（不跨页时只有一页）
    text: str                       # 正文
    context: str = ""               # 前缀：「文档名 > 章节路径」
    section: tuple[str, ...] = ()

    @property
    def page(self) -> int:
        return self.pages[0]

    @property
    def search_text(self) -> str:
        """检索用的文字：前缀 + 正文。"""
        return f"{self.context}\n{self.text}" if self.context else self.text


def doc_title(doc: Document) -> str:
    """「3M 2018 10K」：数据集给了公司、年份、类型就用，没有就用文件名。"""
    parts = [str(doc.meta[k]) for k in ("company", "period", "doc_type") if doc.meta.get(k)]
    return " ".join(parts) or doc.name


def chunk_document(doc: Document, spec: ChunkSpec, count: TokenCounter = approx_tokens,
                   start_id: int = 0) -> list[Chunk]:
    title = doc_title(doc)
    elements = [e for e in doc.elements if e.type != "header_footer"]
    if spec.kind == "fixed":
        pieces = [(p, (), text) for p in range(doc.pages) for text in _windows(_page_text(elements, p), spec)]
    elif spec.kind == "page":
        pieces = [(p, (), text) for p in range(doc.pages) if (text := _page_text(elements, p))]
    else:
        pieces = list(_structure(elements, spec, count))
    out = []
    for pages, section, text in pieces:
        pages = pages if isinstance(pages, tuple) else (pages,)
        context = " > ".join((title, *section)) if spec.contextualize else ""
        out.append(Chunk(start_id + len(out), doc.name, pages, text, context, section))
    return out


# ---------------------------------------------------------------- fixed / page
def _page_text(elements: list[Element], page: int) -> str:
    return "\n\n".join(e.text for e in elements if e.page == page).strip()


def _windows(text: str, spec: ChunkSpec) -> list[str]:
    if not text:
        return []
    words = _WORD.findall(text)
    step = spec.size - spec.overlap
    return ["".join(words[s:s + spec.size]).strip() for s in range(0, max(len(words) - spec.overlap, 1), step)]


# ---------------------------------------------------------------- structure
def _structure(elements: list[Element], spec: ChunkSpec,
               count: TokenCounter) -> Iterable[tuple[tuple[int, ...], tuple[str, ...], str]]:
    """同一章节、同一页（除非 cross_page）的相邻元素合并，直到接近 max_tokens。
    标题不单独成片：它留在缓冲里，等下面的内容来了一起出。"""
    buf: list[str] = []
    pages: list[int] = []
    section: tuple[str, ...] = ()
    size, titles_only = 0, True

    for e in elements:
        boundary = buf and ((e.page != pages[-1] and not spec.cross_page)
                            or (e.type == "title" and not titles_only)       # 新的小节
                            or e.section[:2] != section[:2])                 # 换了 PART / Item
        if boundary:
            yield tuple(sorted(set(pages))), section, "\n\n".join(buf)
            buf, pages, size, titles_only = [], [], 0, True
        if not buf:
            section = e.section
        for piece in _fit(e, spec.max_tokens, count):
            n = count(piece)
            if buf and size + n > spec.max_tokens and not titles_only:
                yield tuple(sorted(set(pages))), section, "\n\n".join(buf)
                buf, pages, size, titles_only = [], [], 0, True
            buf.append(piece)
            pages.append(e.page)
            size += n
            titles_only = titles_only and e.type == "title"
    if buf:
        yield tuple(sorted(set(pages))), section, "\n\n".join(buf)


def _fit(e: Element, limit: int, count: TokenCounter) -> list[str]:
    """一个元素放不进一片时拆开：表格按行拆、每块带表头；段落按句子拆。"""
    if count(e.text) <= limit:
        return [e.text]
    if e.type == "table" and e.rows:
        return _split_table(e, limit, count)
    return _split_text(e.text, limit, count)


def _split_table(e: Element, limit: int, count: TokenCounter) -> list[str]:
    lines = e.text.splitlines()
    head = lines[:2]                                       # 表头行 + 分隔线
    out, cur = [], list(head)
    for line in lines[2:]:
        if len(cur) > 2 and count("\n".join(cur + [line])) > limit:
            out.append("\n".join(cur))
            cur = list(head)
        cur.append(line)
    out.append("\n".join(cur))
    return out


def _split_text(text: str, limit: int, count: TokenCounter) -> list[str]:
    out, cur = [], ""
    for sentence in _SENTENCE.split(text):
        if cur and count(cur + " " + sentence) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
        while count(cur) > limit:                          # 一句就超长（没有句号的长串）：按词硬切
            words = cur.split()
            keep = max(1, int(len(words) * limit / count(cur)))
            out.append(" ".join(words[:keep]))
            cur = " ".join(words[keep:])
    if cur:
        out.append(cur)
    return out
