"""自己写的版面分析，底层用 PyMuPDF 取文字片段（带位置、字号、粗细）。几秒一份 10-K，不用模型。

流程（学 Unstructured / Docling 的思路，规则换成适合财报的）：

1. 行重建：同一条水平线上的片段拼成一行；片段之间空隙大（超过一个字高）就是分列。
   直接 get_text() 会把表格的每一格各放一行，科目名和数字就对不上了。
2. 页眉页脚：页面上下边缘、在很多页上重复出现的行（数字换成 # 再比），还有页面最后一行的孤立页码。
3. 每一行分类：
   表格行  至少两格，后面的格里有数（金额、百分比、年份、—）；或者是列名行：只有日期、年份、Level 1/2/3
           （「May 31, 2020 ⎮ May 26, 2019」常常加粗，以前被当成标题，把表的章节「Consolidated Balance Sheets」顶掉了）
   标题    整行加粗或字号比正文大，不太长；或者是 10-K 的 PART / Item 开头
   正文    其它
4. 合成元素：连续的表格行合成一张表（中间夹着的短小标签行，比如「Cash Flows from Investing Activities」、
   折行的科目名，也算表的一部分）；行距正常的正文行合成一段；相邻的标题行合成一个标题。
5. 表格按列对齐：数字右对齐，按右边缘聚出列的位置，每个数放进最近的一列 —— 某一年没有数时不会错位。
6. 章节路径：PART → Item → 其它标题，三层。

已知不处理：双栏排版（按从上到下读，两栏会混在一起），扫描件（没有文字层）。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..document import Document, Element

_CURRENCY = {"$", "€", "£", "¥", "US$"}
_NUMERIC = re.compile(r"^[(\-–—]?\s*(?:US)?[$€£¥]?\s*[(\-–—]?\s*\d[\d,]*(?:\.\d+)?\s*[%)]*\s*[)%]?$|^[—–-]{1,3}$|^n/?m$|^nm$",
                      re.IGNORECASE)
_YEAR = re.compile(r"^(19|20)\d\d$")
_PART = re.compile(r"^PART\s+[IVX]+\b", re.IGNORECASE)
_ITEM = re.compile(r"^ITEM\s+\d+[A-Z]?\b", re.IGNORECASE)
_DATE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d\d"
_HEADER_TOKEN = re.compile(rf"{_DATE}|\b(?:19|20)\d\d\b|\blevel\s+[1-3]\b", re.IGNORECASE)
# 列名行里除了日期年份，只能有这些词；有 compared / vs 的是 MD&A 的小标题（「Fiscal 2020 Compared to Fiscal 2019」）
_HEADER_FILLER = re.compile(r"\b(?:total|as|of|at|and|fiscal|year|years|ended|in|millions|thousands)\b|[,.\-–—()/&]",
                            re.IGNORECASE)
_PAGE_NO = re.compile(r"^(page\s+)?[\divxlc]+(\s+of\s+\d+)?$|^-\s*\d+\s*-$", re.IGNORECASE)

MARGIN = 0.08             # 页面上下各 8% 算边缘
REPEAT = 0.3              # 在至少 30% 的页上重复出现 → 页眉页脚
TITLE_MAX_CHARS = 150
TABLE_GAP_ROWS = 3        # 表里最多连着夹几行非表格行（折行的科目名、小标题）


@dataclass(slots=True)
class _Cell:
    text: str
    x0: float
    x1: float


@dataclass(slots=True)
class _Row:
    page: int
    cells: list[_Cell]
    y0: float
    y1: float
    size: float
    bold: bool
    kind: str = ""                    # table / title / text / header_footer

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.cells)

    @property
    def x0(self) -> float:
        return self.cells[0].x0

    @property
    def x1(self) -> float:
        return self.cells[-1].x1


@dataclass(slots=True)
class _State:
    section: list[str] = field(default_factory=lambda: ["", "", ""])

    def enter(self, title: str) -> None:
        level = 0 if _PART.match(title) else 1 if _ITEM.match(title) else 2
        self.section[level] = title
        for deeper in range(level + 1, 3):
            self.section[deeper] = ""

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(s for s in self.section if s)


def parse(path: Path) -> Document:
    import pymupdf

    with pymupdf.open(path) as pdf:
        pages = [(_rows(page, n), page.rect.height) for n, page in enumerate(pdf)]
    body = _body_size([r for rows, _ in pages for r in rows])
    _mark_header_footer(pages)
    for rows, _ in pages:
        for r in rows:
            if not r.kind:
                r.kind = _classify(r, body)
        _attach_table_headers(rows)
    doc = Document(path.stem, pages=len(pages), parser="pymupdf")
    state = _State()
    for rows, _ in pages:
        doc.elements += _elements(rows, state)
    return doc


# ---------------------------------------------------------------- 1. 行重建
def _rows(page, number: int) -> list[_Row]:
    spans = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", ()):
            for s in line["spans"]:
                if s["text"].strip():
                    bold = bool(s["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold|-bd", s["font"], re.I))
                    spans.append((s["text"], *s["bbox"], s["size"], bold))
    spans.sort(key=lambda s: ((s[2] + s[4]) / 2, s[1]))
    lines: list[list[tuple]] = []
    for s in spans:
        center = (s[2] + s[4]) / 2
        if lines and abs(_center(lines[-1]) - center) < s[5] * 0.5:
            lines[-1].append(s)
        else:
            lines.append([s])
    return [_row(sorted(ss, key=lambda s: s[1]), number) for ss in lines]


def _center(spans: list[tuple]) -> float:
    return sum((s[2] + s[4]) / 2 for s in spans) / len(spans)


def _row(spans: list[tuple], page: int) -> _Row:
    cells: list[_Cell] = []
    prev = None
    for text, x0, y0, x1, y1, size, _ in spans:
        gap = x0 - prev[3] if prev else 0
        if cells and gap <= size:                     # 空隙不到一个字高：同一格
            c = cells[-1]
            c.text = (c.text + ("" if gap < 0.5 else " ") + text).strip()
            c.x1 = x1
        else:
            cells.append(_Cell(text.strip(), x0, x1))
        prev = (text, x0, y0, x1)
    cells = _merge_currency(cells)
    size = Counter(round(s[5], 1) for s in spans).most_common(1)[0][0]
    return _Row(page, cells, min(s[2] for s in spans), max(s[4] for s in spans), size,
                bold=all(s[6] for s in spans))


def _merge_currency(cells: list[_Cell]) -> list[_Cell]:
    """「$ ⎮ 5,363」：单独的货币符号并进后面那格；「4,869 $」：格子末尾粘上的下一格的符号去掉。"""
    out: list[_Cell] = []
    for c in cells:
        c.text = re.sub(r"\s+[$€£¥]$", "", c.text)
        if out and out[-1].text in _CURRENCY:
            c = _Cell(c.text, out[-1].x0, c.x1)
            out[-1] = c
        else:
            out.append(c)
    return [c for c in out if c.text]


# ---------------------------------------------------------------- 2. 页眉页脚
def _mark_header_footer(pages: list[tuple[list[_Row], float]]) -> None:
    def key(r: _Row) -> str:
        return re.sub(r"\d+", "#", r.text.lower()).strip()

    edge: Counter[str] = Counter()
    for rows, height in pages:
        edge.update({key(r) for r in rows if _at_edge(r, height)})
    threshold = max(3, REPEAT * len(pages))
    for rows, height in pages:
        for r in rows:
            if _at_edge(r, height) and edge[key(r)] >= threshold:
                r.kind = "header_footer"
        body = [r for r in rows if r.kind != "header_footer"]
        if body and _PAGE_NO.match(body[-1].text.strip()):      # 页面最后一行的孤立页码
            body[-1].kind = "header_footer"


def _at_edge(r: _Row, height: float) -> bool:
    return r.y1 < height * MARGIN or r.y0 > height * (1 - MARGIN)


# ---------------------------------------------------------------- 3. 行分类
def _body_size(rows: list[_Row]) -> float:
    sizes = Counter()
    for r in rows:
        sizes[round(r.size, 1)] += len(r.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _classify(r: _Row, body: float) -> str:
    if len(r.cells) >= 2 and any(_NUMERIC.match(c.text) for c in r.cells[1:]):
        return "table"
    text = r.text.strip()
    if _column_header(text):                          # 列名行：几格的归进下面的表，一格的（副标题里的日期）当正文
        return "table" if len(r.cells) >= 2 else "text"
    if len(text) <= TITLE_MAX_CHARS and (_PART.match(text) or _ITEM.match(text)):
        return "title"
    if len(text) <= TITLE_MAX_CHARS and (r.bold or r.size > body * 1.15) and not text.endswith((",", ";")):
        return "title"
    return "text"


def _column_header(text: str) -> bool:
    """「May 31, 2020 May 26, 2019」「December 31, 2022 and 2021」「Level 1 Level 2 Level 3 Total」：
    至少两个日期 / 年份 / Level，除此之外只有 Total、Fiscal、Years Ended 这类词。"""
    if len(_HEADER_TOKEN.findall(text)) < 2:
        return False
    return not _HEADER_FILLER.sub(" ", _HEADER_TOKEN.sub(" ", text)).strip()


def _attach_table_headers(rows: list[_Row]) -> None:
    """表上面紧挨着的多格行（加粗的列名「Weighted Average / Number of…」）是表头，不是标题。"""
    for i, r in enumerate(rows):
        if r.kind != "table" or (i and rows[i - 1].kind == "table"):
            continue
        j = i - 1
        while j >= 0 and rows[j].kind in ("title", "text") and len(rows[j].cells) >= 2 \
                and rows[j + 1].y0 - rows[j].y1 < rows[j].size * 1.5:
            rows[j].kind = "table"
            j -= 1


# ---------------------------------------------------------------- 4. 合成元素
def _elements(rows: list[_Row], state: _State) -> list[Element]:
    out: list[Element] = []
    pitch = _line_gap(rows)
    left = min((r.x0 for r in rows if r.kind == "text"), default=0.0)
    right = max((r.x1 for r in rows if r.kind == "text"), default=0.0)
    i = 0
    while i < len(rows):
        r = rows[i]
        if r.kind == "header_footer":
            out.append(Element("header_footer", r.text, r.page, _bbox([r])))
            i += 1
        elif r.kind == "table":
            j = _table_end(rows, i)
            out.append(_table(rows[i:j], state))
            i = j
        elif r.kind == "title":
            j = i + 1
            while j < len(rows) and rows[j].kind == "title" and rows[j].y0 - rows[j - 1].y1 < r.size * 1.2 \
                    and not (_structural(rows[j - 1].text) or _structural(rows[j].text)):
                j += 1                                # PART / Item 单独成一个标题，不和上下合并
            title = re.sub(r"\s+", " ", " / ".join(x.text for x in rows[i:j]))
            state.enter(title)
            out.append(Element("title", title, r.page, _bbox(rows[i:j]), state.path))
            i = j
        else:
            j = i + 1
            while j < len(rows) and rows[j].kind == "text" and _same_paragraph(rows[j - 1], rows[j], pitch,
                                                                                  left, right):
                j += 1
            out.append(Element("paragraph", _join_lines(rows[i:j]), r.page, _bbox(rows[i:j]), state.path))
            i = j
    return out


def _structural(text: str) -> bool:
    return bool(_PART.match(text.strip()) or _ITEM.match(text.strip()))


def _line_gap(rows: list[_Row]) -> float:
    """这一页正文的行间空白（中位数）。双倍行距的附件和单倍行距的正文不一样，不能用固定值。"""
    gaps = sorted(b.y0 - a.y1 for a, b in zip(rows, rows[1:]) if a.kind == b.kind == "text" and b.y0 > a.y1)
    return gaps[len(gaps) // 2] if gaps else 0.0


def _same_paragraph(prev: _Row, row: _Row, pitch: float, left: float, right: float) -> bool:
    """行距正常就接着算一段；上一行没写满、这一行又缩进了，是新的一段。"""
    if row.y0 - prev.y1 > max(pitch * 1.4, prev.size * 0.5):
        return False
    return not (row.x0 > left + 15 and prev.x1 < right - 40)


def _table_end(rows: list[_Row], start: int) -> int:
    """表从 start 开始，到哪一行结束（不含）。表里可以夹几行短的非表格行，只要后面还有表格行。"""
    end = start + 1
    j = start + 1
    while j < len(rows):
        r = rows[j]
        if r.kind == "header_footer":
            break
        gap = r.y0 - rows[j - 1].y1
        if r.kind == "table" and gap < r.size * 3:
            end = j = j + 1
            continue
        if r.kind in ("text", "title") and len(r.cells) >= 2 and gap < r.size * 2:
            j += 1                                    # 多格的行（几层的表头）：明显是表的一部分
            continue
        if r.kind in ("text", "title") and len(r.text) < TITLE_MAX_CHARS and gap < r.size * 2 and \
                any(x.kind == "table" for x in rows[j + 1:j + 1 + TABLE_GAP_ROWS]):
            j += 1
            continue
        break
    return end


def _table(rows: list[_Row], state: _State) -> Element:
    grid = _grid(rows)
    return Element("table", _markdown(grid), rows[0].page, _bbox(rows), state.path, tuple(map(tuple, grid)))


def _grid(rows: list[_Row]) -> list[list[str]]:
    """第一列是左边的文字（科目名），数字按右边缘对到最近的列。
    列的位置只从数字算：表头的年份常常居中，右边缘和下面的数对不齐，拿它算会多出一列。"""
    numbers = [c for r in rows for c in r.cells[1:] if _NUMERIC.match(c.text)]
    values = [c for c in numbers if not _YEAR.match(c.text)] or numbers
    rights = sorted(c.x1 for c in values)
    anchors: list[float] = []
    for x in rights:
        if anchors and x - anchors[-1] < 12:
            anchors[-1] = (anchors[-1] + x) / 2
        else:
            anchors.append(x)
    grid = []
    for r in rows:
        line = [""] * (len(anchors) + 1)
        label = []
        header = _column_header(r.text)              # 列名行：日期、年份那几格对到列上（「(Millions)」还是标签）
        for c in r.cells:
            if anchors and (header and _HEADER_TOKEN.search(c.text)
                            or (_NUMERIC.match(c.text) or c.x0 > anchors[0] - 60) and c is not r.cells[0]):
                k = min(range(len(anchors)), key=lambda a: abs(anchors[a] - c.x1))
                line[k + 1] = (line[k + 1] + " " + c.text).strip()
            else:
                label.append(c.text)
        line[0] = " ".join(label)
        grid.append(line)
    return grid


def _markdown(grid: list[list[str]]) -> str:
    width = max(len(r) for r in grid)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in grid]
    return "\n".join([lines[0], "|" + "---|" * width, *lines[1:]])


def _join_lines(rows: list[_Row]) -> str:
    text = ""
    for r in rows:
        line = r.text.strip()
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line                  # 行尾连字符断开的词接回去
        else:
            text = f"{text} {line}" if text else line
    return text


def _bbox(rows: list[_Row]) -> tuple[float, float, float, float]:
    return (min(r.x0 for r in rows), min(r.y0 for r in rows), max(r.x1 for r in rows), max(r.y1 for r in rows))
