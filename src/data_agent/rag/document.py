"""解析结果的统一格式：一份文档 = 一串有类型的元素（学 Unstructured / Docling 的文档模型）。

不管用哪个解析器（自己写的 PyMuPDF 版、MinerU、Docling），都转成这个格式，分片、索引、评测只认它。

    Element.type
        title          标题（10-K 的 PART / Item、报表名、加粗的小标题）
        paragraph      正文段落
        table          表格：rows 是按行按列还原的格子，text 是渲染好的 Markdown
        header_footer  页眉页脚、页码（每页都重复的东西）：保留下来方便排查，建索引时丢掉
    Element.section   所在章节的路径，比如 ("PART II", "Item 8. Financial Statements", "Consolidated Statement of Cash Flows")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ElementType = Literal["title", "paragraph", "table", "header_footer"]


@dataclass(frozen=True, slots=True)
class Element:
    type: ElementType
    text: str
    page: int                                   # 从 0 数
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    section: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()      # 只有 table 有


@dataclass(slots=True)
class Document:
    name: str                                   # 文件名去掉扩展名
    elements: list[Element] = field(default_factory=list)
    pages: int = 0
    parser: str = ""
    meta: dict[str, object] = field(default_factory=dict)    # 公司、年份、文档类型…（数据集给的）

    def page_text(self, page: int) -> str:
        """一页的正文（不含页眉页脚），按元素顺序拼起来。给「读整页」用。"""
        return "\n\n".join(e.text for e in self.elements if e.page == page and e.type != "header_footer")

    # ------------------------------------------------------------ 存取
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        head = {"name": self.name, "pages": self.pages, "parser": self.parser, "meta": self.meta}
        lines = [json.dumps(head, ensure_ascii=False)] + [json.dumps(asdict(e), ensure_ascii=False)
                                                         for e in self.elements]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Document":
        head, *rest = path.read_text(encoding="utf-8").splitlines()
        doc = cls(**json.loads(head))
        for line in rest:
            d = json.loads(line)
            doc.elements.append(Element(d["type"], d["text"], d["page"], tuple(d["bbox"]), tuple(d["section"]),
                                        tuple(tuple(r) for r in d["rows"])))
        return doc
