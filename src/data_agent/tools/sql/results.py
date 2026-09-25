"""查询结果仓库：run_sql 查出过的结果，按编号（r1、r2…）各存一份，只增不减。

run_sql 存进来；export_csv 按编号找到 SQL 重跑；界面和评测把回答里的 {{r3}} 展开成整张表。
不放进对话历史：历史会随失败的一轮回滚，但用户已经看过 r5、可能接着 /save r5，编号也不能回收。
给了 path 就边查边追加到 results.jsonl，恢复会话后编号接着往下编。
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...db.connection import QueryResult

# 回答里的结果引用：模型写 {{r3}}，界面在这里展示整张表
REF = re.compile(r"\{\{\s*(r\d+)\s*\}\}")


@dataclass(frozen=True, slots=True)
class SqlResult:
    """一次 run_sql 的完整结果，也是它给界面的 details。"""

    ref: str                 # 结果编号，r1、r2…
    sql: str
    result: QueryResult


class ResultStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path              # None = 只在内存里
        self._results: dict[str, SqlResult] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                table = _decode(json.loads(line))
                self._results[table.ref] = table
        last = max((int(ref[1:]) for ref in self._results), default=0)
        self._numbers = itertools.count(last + 1)

    def add(self, sql: str, result: QueryResult) -> SqlResult:
        """存一个结果，编上下一个号。"""
        table = SqlResult(f"r{next(self._numbers)}", sql, result)
        self._results[table.ref] = table
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_encode(table), ensure_ascii=False, default=str) + "\n")
        return table

    def get(self, ref: str) -> SqlResult | None:
        return self._results.get(ref.strip())

    def refs(self) -> list[str]:
        """按先后排，最后一个是最近的。"""
        return list(self._results)

    def expand(self, text: str, render: Callable[[SqlResult], str]) -> str:
        """把 {{r3}} 换成 render(结果)。怎么画由界面定。"""
        def one(m: re.Match) -> str:
            table = self.get(m.group(1))
            return render(table) if table else f"（找不到结果 {m.group(1)}）"
        return REF.sub(one, text)


def _encode(t: SqlResult) -> dict:
    r = t.result
    return {"ref": t.ref, "sql": t.sql, "columns": r.columns, "rows": r.rows,
            "truncated": r.truncated, "elapsed_ms": r.elapsed_ms}


def _decode(d: dict) -> SqlResult:
    result = QueryResult(d["columns"], [tuple(row) for row in d["rows"]], d["truncated"], d["elapsed_ms"])
    return SqlResult(d["ref"], d["sql"], result)


def markdown_table(columns: list[str], rows: list[tuple]) -> str:
    """结果表 → Markdown 表格（模型看的预览、表结构样例、界面展示都用它）。"""
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
