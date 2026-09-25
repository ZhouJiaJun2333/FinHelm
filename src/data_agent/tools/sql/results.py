"""查询结果仓库：这次会话里 run_sql 查出过的结果，按编号（r1、r2…）各存一份。

    run_sql       查完存进来，拿到编号
    export_csv    按编号找到当时的 SQL，重跑、写文件（终端的 /save 走同一条路）
    界面 / 评测   回答里的 {{r3}} 按编号展开成整张表

以前这份数据存了三处：CLI 的 tables、run_sql 的「编号 → SQL」、评测从事件里重建一份。
三处各管各的，/save 查一处、export_csv 查另一处。现在只有这一份，由组装层（app.py）
建好，注入给用到它的工具，界面和评测从 Application 上拿。

── 为什么不放进对话历史 ──────────────────────────────────────────────
pi 把工具给界面的 details 存在工具结果消息里，跟着会话日志走。我们没这么做，因为语义对不上：
    · 历史是**事务**的：一轮失败，整轮回滚（Agent.run）。但那一轮查出来的 r5 已经在终端上
      展示过了，用户可能接着 /save r5 —— 用户见过的东西不能跟着回滚消失。
    · 编号也不能回收：回滚之后再查一次，如果又叫 r5，界面上就有两个不一样的 r5。
所以这里记的是「用户见过什么」，只增不减。pi 没这个矛盾：它失败的一轮也留在日志里，不回滚。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass

from ...db.connection import QueryResult

# 回答里的结果引用：模型写 {{r3}}，界面在这个位置展示整张表，模型就不用逐行抄写
REF = re.compile(r"\{\{\s*(r\d+)\s*\}\}")


@dataclass(frozen=True, slots=True)
class SqlResult:
    """一次 run_sql 的完整结果。也是 run_sql 给界面的 details。"""

    ref: str                 # 结果编号，r1、r2…
    sql: str
    result: QueryResult


class ResultStore:
    def __init__(self) -> None:
        self._results: dict[str, SqlResult] = {}
        self._numbers = itertools.count(1)

    def add(self, sql: str, result: QueryResult) -> SqlResult:
        """存一个结果，编上下一个号。"""
        table = SqlResult(f"r{next(self._numbers)}", sql, result)
        self._results[table.ref] = table
        return table

    def get(self, ref: str) -> SqlResult | None:
        return self._results.get(ref.strip())

    def refs(self) -> list[str]:
        """已有的编号，按先后排（dict 保持插入顺序），最后一个是最近的。"""
        return list(self._results)

    def expand(self, text: str, render: Callable[[SqlResult], str]) -> str:
        """把回答里的 {{r3}} 换成 render(结果)。怎么画由界面定：终端放前几行，评测展开整张。"""
        def one(m: re.Match) -> str:
            table = self.get(m.group(1))
            return render(table) if table else f"（找不到结果 {m.group(1)}）"
        return REF.sub(one, text)


def markdown_table(columns: list[str], rows: list[tuple]) -> str:
    """结果表 → Markdown 表格。模型看的预览、表结构里的样例、界面展示的完整表都用它。"""
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
