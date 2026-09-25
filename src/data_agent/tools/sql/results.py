"""结果仓库：run_sql 查出过的结果、沙箱里 save_result() 存下的表，按编号（r1、r2…）各存一份，只增不减。

run_sql 和沙箱存进来；导出时 SQL 结果按 SQL 重跑、沙箱的表直接写存下的行；
界面和评测把回答里的 {{r3}} 展开成整张表；沙箱 load_result("r3") 从这里取。
不放进对话历史：历史会随失败的一轮回滚，但用户已经看过 r5、可能接着 /save r5，编号也不能回收。
给了 path 就边查边追加到 results.jsonl，恢复会话后编号接着往下编。
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from ...db.connection import QueryResult

# 回答里的结果引用：模型写 {{r3}}，界面在这里展示整张表
REF = re.compile(r"\{\{\s*(r\d+)\s*\}\}")


# 沙箱存一张表最多多少行（和 run_sql 取回的上限一样），再多该先聚合
SAVE_ROWS = 10_000


@dataclass(frozen=True, slots=True)
class StoredResult:
    """一个编了号的结果。run_sql 的也是它给界面的 details。"""

    ref: str                 # 结果编号，r1、r2…
    sql: str                 # run_sql 的 SQL；沙箱存的表是空的（导出时没法重跑，直接写存下的行）
    result: QueryResult
    source: str = "sql"      # sql / python / r
    title: str = ""          # 沙箱存的时候起的名字


class ResultStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path              # None = 只在内存里
        self._results: dict[str, StoredResult] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                table = _decode(json.loads(line))
                self._results[table.ref] = table
        last = max((int(ref[1:]) for ref in self._results), default=0)
        self._numbers = itertools.count(last + 1)

    def add(self, sql: str, result: QueryResult, *, source: str = "sql", title: str = "") -> StoredResult:
        """存一个结果，编上下一个号。"""
        table = StoredResult(f"r{next(self._numbers)}", sql, result, source, title)
        self._results[table.ref] = table
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_encode(table), ensure_ascii=False, default=str) + "\n")
        return table

    def get(self, ref: str) -> StoredResult | None:
        return self._results.get(ref.strip())

    def refs(self) -> list[str]:
        """按先后排，最后一个是最近的。"""
        return list(self._results)

    def expand(self, text: str, render: Callable[[StoredResult], str]) -> str:
        """把 {{r3}} 换成 render(结果)。怎么画由界面定。"""
        def one(m: re.Match) -> str:
            table = self.get(m.group(1))
            return render(table) if table else f"（找不到结果 {m.group(1)}）"
        return REF.sub(one, text)


def _encode(t: StoredResult) -> dict:
    r = t.result
    extra = {"source": t.source, "title": t.title} if t.source != "sql" else {}
    return {"ref": t.ref, "sql": t.sql, "columns": r.columns, "rows": r.rows,
            "truncated": r.truncated, "elapsed_ms": r.elapsed_ms, **extra}


def _decode(d: dict) -> StoredResult:
    result = QueryResult(d["columns"], [tuple(row) for row in d["rows"]], d["truncated"], d["elapsed_ms"])
    return StoredResult(d["ref"], d["sql"], result, d.get("source", "sql"), d.get("title", ""))


def result_resolver(results: ResultStore) -> Callable[[str], dict[str, Any]]:
    """沙箱内核 load_result("r3") 时，宿主机按编号从这里取数据。"""
    def resolve(ref: str) -> dict[str, Any]:
        table = results.get(ref)
        if table is None:
            known = "、".join(results.refs()) or "还没有"
            return {"error": f"没有编号为 {ref} 的结果。本次对话里的编号：{known}"}
        return encode_result(table.result)
    return resolve


def result_saver(results: ResultStore, source: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """沙箱内核 save_result(df) 时，宿主机把表存进仓库、编上号，把编号回给内核。"""
    def save(payload: dict[str, Any]) -> dict[str, Any]:
        columns = [str(c) for c in payload.get("columns", [])]
        rows = [tuple(r) for r in payload.get("rows", [])]
        if not columns or not rows:
            return {"error": "表是空的，没有存。"}
        result = QueryResult(columns, rows[:SAVE_ROWS], len(rows) > SAVE_ROWS, 0)
        table = results.add("", result, source=source, title=str(payload.get("title", "")))
        return {"ref": table.ref, "rows": result.row_count, "truncated": result.truncated}
    return save


def encode_result(result: QueryResult) -> dict[str, Any]:
    """转成能过 JSON 的样子。Decimal 转 float（DataFrame 里本来也是 float）；日期列记下来，内核那头再转回去。"""
    rows = [[_plain(v) for v in row] for row in result.rows]
    dates = [
        col for i, col in enumerate(result.columns)
        if any(isinstance(row[i], dt.date) for row in result.rows)
        and all(row[i] is None or isinstance(row[i], dt.date) for row in result.rows)
    ]
    return {"columns": result.columns, "rows": rows, "dates": dates, "truncated": result.truncated}


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return str(v)


def markdown_table(columns: list[str], rows: list[tuple]) -> str:
    """结果表 → Markdown 表格（模型看的预览、表结构样例、界面展示都用它）。"""
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
