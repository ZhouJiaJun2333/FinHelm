"""解析器：PDF → Document（一串有类型的元素）。用名字选，结果按「解析器 + 版本 + 文件内容」缓存。

    pymupdf   自己写的版面分析（pymupdf_layout.py），默认
    以后加    mineru（在装了它的环境里跑，结果转成 Document）、docling …

改了某个解析器的规则就把它的版本号加一：旧缓存自动作废，不会拿旧结果去评测新规则。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from ..document import Document
from . import pymupdf_layout

# 名字 → (解析函数, 版本)
PARSERS: dict[str, tuple[Callable[[Path], Document], int]] = {
    "pymupdf": (pymupdf_layout.parse, 3),         # v3：列名行（两列日期、Level 1/2/3）不再当标题
}


def parse(path: Path, parser: str = "pymupdf") -> Document:
    if parser not in PARSERS:
        raise ValueError(f"没有叫 {parser} 的解析器，有这些：{', '.join(PARSERS)}")
    return PARSERS[parser][0](path)


def file_sha(path: Path) -> str:
    """文件内容的 SHA-1（十六进制）。按块读，几百 MB 的 PDF 也不占内存。"""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


class ParsedCache:
    """<root>/<解析器>-v<版本>/<文件内容哈希>.jsonl。解析过的直接读。

    按内容不按文件名：同名文件换了内容会重新解析；改名、复制不用重新解析（读出来按现在的文件名）。
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, path: Path, parser: str = "pymupdf", meta: dict | None = None, sha: str = "") -> Document:
        """sha：调用方已经算过就传进来，省得再读一遍文件。"""
        _, version = PARSERS[parser]
        cached = self.root / f"{parser}-v{version}" / f"{sha or file_sha(path)}.jsonl"
        if cached.is_file():
            doc = Document.load(cached)
            doc.name = path.stem
        else:
            doc = parse(path, parser)
            doc.save(cached)
        if meta:
            doc.meta = {**doc.meta, **meta}
        return doc
