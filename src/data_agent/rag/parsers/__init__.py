"""解析器：PDF → Document（一串有类型的元素）。用名字选，结果按「解析器 + 版本」缓存。

    pymupdf   自己写的版面分析（pymupdf_layout.py），默认
    以后加    mineru（在装了它的环境里跑，结果转成 Document）、docling …

改了某个解析器的规则就把它的版本号加一：旧缓存自动作废，不会拿旧结果去评测新规则。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..document import Document
from . import pymupdf_layout

# 名字 → (解析函数, 版本)
PARSERS: dict[str, tuple[Callable[[Path], Document], int]] = {
    "pymupdf": (pymupdf_layout.parse, 2),
}


def parse(path: Path, parser: str = "pymupdf") -> Document:
    if parser not in PARSERS:
        raise ValueError(f"没有叫 {parser} 的解析器，有这些：{', '.join(PARSERS)}")
    return PARSERS[parser][0](path)


class ParsedCache:
    """<root>/<解析器>-v<版本>/<文档>.jsonl。解析过的直接读。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, path: Path, parser: str = "pymupdf", meta: dict | None = None) -> Document:
        _, version = PARSERS[parser]
        cached = self.root / f"{parser}-v{version}" / f"{path.stem}.jsonl"
        if cached.is_file():
            doc = Document.load(cached)
        else:
            doc = parse(path, parser)
            doc.save(cached)
        if meta:
            doc.meta = {**doc.meta, **meta}
        return doc
