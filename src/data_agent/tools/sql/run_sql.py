"""工具 3：执行只读 SQL。

这是整个 Agent 唯一真正「产出结论」的工具，也是唯一有安全风险的工具。

安全设计是**三道独立的防线**，不指望任何单点：
    1. 这里：只允许单条 SELECT / WITH 语句，拒绝多语句和一切写操作关键字
    2. 连接层（db/connection.py）：只读事务 + 语句超时 + 强制行数上限
    3. 数据库（docker/initdb/03_readonly_role.sql）：agent_ro 账号物理上没有写权限

第 1 道是可以被绕过的（注释、大小写、奇怪语法…），所以第 3 道才是真正的底线。
**永远不要只靠关键字黑名单来做安全。**

结果分两份（见 core/tools.py 的 details）：
    模型    20 行以内原样给；更多只给前 10 行 + 行列数 —— 它要的是够推理的信息，
            要看别的行就改 SQL 再查（只读查询重跑拿到的是同一份数据，还能顺手筛选、聚合）
    界面    完整结果（最多 1 万行），放在 details 里；用户要文件时再导出 CSV（export_csv）
每个结果有编号（r1、r2…），模型和用户靠它指认同一份结果。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ...db.connection import Database, QueryResult
from ...core.tools import Tool, ToolOutput

# 语句必须以这些开头
ALLOWED_STARTS = ("select", "with")

# 一眼就该拒绝的写操作。这是「减少误伤」，不是安全边界。
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|"
    r"copy|vacuum|reindex|call|do|set|reset)\b",
    re.IGNORECASE,
)


# 界面拿到的完整结果最多这么多行。再多就不是「看」的量了，该在 SQL 里聚合
FETCH_ROWS = 10_000
# 模型看到的：不超过 FULL_ROWS 行原样给，超过就只给前 PREVIEW_ROWS 行
FULL_ROWS = 20
PREVIEW_ROWS = 10


@dataclass(frozen=True, slots=True)
class SqlResult:
    """run_sql 给界面的完整结果（ToolOutput.details）。"""

    ref: str                 # 结果编号，r1、r2…
    sql: str
    result: QueryResult


class RunSqlTool(Tool):
    name = "run_sql"
    rerunnable = True          # 只读查询：旧结果被清理后，再调一次就能拿回来
    description = (
        "在分析库上执行一条只读 SQL（只能是 SELECT 或 WITH 开头的单条语句），返回结果表。"
        "执行前请先用 describe_table 确认列名和外键。"
        f"每个结果有编号（r1、r2…）。超过 {FULL_ROWS} 行的结果你只能看到前 {PREVIEW_ROWS} 行，"
        "你要看别的行，就在 SQL 里筛选、排序或聚合后再查。"
        f"超过 {FULL_ROWS} 行的长清单，回答里单独一行写 {{{{r3}}}}，用户会在那个位置看到 r3 的完整结果表。"
    )

    class Args(BaseModel):
        sql: str = Field(description="要执行的 SQL，单条语句，不要加结尾分号以外的内容")
        purpose: str = Field(
            default="",
            description="一句话说明这条 SQL 想回答什么问题。会记进日志，方便你和用户回溯。",
        )

    def __init__(self, db: Database) -> None:
        self.db = db
        # 编号只增不减：/reset、回滚都不回收 —— 界面上已经展示过的 r3 不能换成别的结果
        self._refs = itertools.count(1)
        # 编号 → SQL。只记 SQL 不记结果：要导出时重跑（export_csv），工具层不用存大结果
        self.queries: dict[str, str] = {}

    def run(self, args: Args) -> ToolOutput:
        sql = _validate(args.sql)
        result = self.db.query(sql, max_rows=FETCH_ROWS)
        if not result.columns or not result.rows:
            return ToolOutput(_format_empty(result), _summarize(result))
        ref = f"r{next(self._refs)}"
        self.queries[ref] = sql
        return ToolOutput(_format(ref, result), _summarize(result, ref),
                          details=SqlResult(ref, sql, result))


# ---------------------------------------------------------------- 校验
def _validate(raw: str) -> str:
    sql = raw.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("SQL 为空。")

    # 去掉注释再判断，避免 `/* select */ delete ...` 这种绕过
    stripped = re.sub(r"--[^\n]*", " ", sql)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL).strip()

    if ";" in stripped:
        raise ValueError("只允许执行单条语句，不要用分号拼接多条 SQL。")

    lowered = stripped.lower()
    if not lowered.startswith(ALLOWED_STARTS):
        raise ValueError(
            f"只允许 SELECT / WITH 开头的只读查询，你这条是 '{lowered.split()[0]}'。"
        )

    hit = FORBIDDEN.search(stripped)
    if hit:
        raise ValueError(
            f"SQL 中包含被禁止的关键字 '{hit.group()}'。这个工具只能做只读查询。"
        )
    return sql


# ---------------------------------------------------------------- 格式化
def _summarize(result: QueryResult, ref: str = "") -> str:
    """结果被清理后留下的线索：几行几列、叫什么。

    只有一行几列的小结果（典型的 COUNT / SUM 聚合）直接把值写进来 ——
    这种结果重查一次也要一个来回，而线索本身就几乎等于原件。
    """
    if not result.columns:
        return "语句执行成功，没有结果集"
    head = f"结果 {ref}：" if ref else ""
    cols = result.columns
    shown = ", ".join(cols[:8]) + (" 等" if len(cols) > 8 else "")
    if result.row_count == 1 and len(cols) <= 4:
        values = ", ".join(
            f"{c}={'NULL' if v is None else v}" for c, v in zip(cols, result.rows[0])
        )
        return f"{head}1 行：{values}"
    text = f"{head}{result.row_count} 行 × {len(cols)} 列（{shown}）"
    if result.truncated:
        text += f"，超过 {FETCH_ROWS} 行被截断"
    return text


def _format_empty(result: QueryResult) -> str:
    if not result.columns:
        return "语句执行成功，但没有返回结果集。"
    return f"查询成功，但结果为空（0 行，耗时 {result.elapsed_ms}ms）。"


def _format(ref: str, result: QueryResult) -> str:
    """模型看到的那份。"""
    n, cols = result.row_count, len(result.columns)
    if n <= FULL_ROWS:
        return (f"结果 {ref}（{n} 行 × {cols} 列，耗时 {result.elapsed_ms}ms）：\n\n"
                + markdown_table(result.columns, result.rows))
    more = f"超过 {FETCH_ROWS} 行，只取了前 {n} 行" if result.truncated else f"共 {n} 行"
    return (
        f"结果 {ref}（{more} × {cols} 列，耗时 {result.elapsed_ms}ms）。"
        f"下面只给你看前 {PREVIEW_ROWS} 行。要给用户看整张表，在回答里单独一行写 {{{{{ref}}}}}：\n\n"
        + markdown_table(result.columns, result.rows[:PREVIEW_ROWS])
        + f"\n\n…还有 {n - PREVIEW_ROWS} 行没给你看。要用到其中的数，在 SQL 里筛选、排序或聚合后再查，不要猜。"
    )


# 回答里的结果引用：模型写 {{r3}}，界面在这个位置展示整张表，模型就不用逐行抄写
REF = re.compile(r"\{\{\s*(r\d+)\s*\}\}")


def expand_refs(text: str, tables: Mapping[str, SqlResult],
                render: Callable[[SqlResult], str]) -> str:
    """把回答里的 {{r3}} 换成 render(结果)。怎么画由界面定：终端放前几行，评测展开整张。"""
    def one(m: re.Match) -> str:
        table = tables.get(m.group(1))
        return render(table) if table else f"（找不到结果 {m.group(1)}）"
    return REF.sub(one, text)


def markdown_table(columns: list[str], rows: list[tuple]) -> str:
    """结果表 → Markdown 表格。模型看的预览、界面展示的完整表都用它。"""
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
