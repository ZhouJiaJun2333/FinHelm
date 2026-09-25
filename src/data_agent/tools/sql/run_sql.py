"""run_sql：执行只读 SQL。

安全是三道独立的防线：这里只放行单条 SELECT / WITH（能被绕过，只是减少误伤）；
连接层只读事务 + 超时 + 行数上限；数据库账号 agent_ro 物理上没有写权限（真正的底线）。

结果分两份：模型 20 行以内原样给，更多只给前 10 行（要别的行就改 SQL 再查）；
界面拿完整结果（最多 1 万行），存进结果仓库并编号。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ...core.tools import Tool, ToolOutput
from ...db.connection import Database, QueryResult
from .results import ResultStore, markdown_table

ALLOWED_STARTS = ("select", "with")

# 一眼就该拒绝的写操作。减少误伤，不是安全边界
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|"
    r"copy|vacuum|reindex|call|do|set|reset)\b",
    re.IGNORECASE,
)


# 界面拿到的完整结果最多这么多行，再多该在 SQL 里聚合
FETCH_ROWS = 10_000
# 模型看到的：不超过 FULL_ROWS 行原样给，超过只给前 PREVIEW_ROWS 行
FULL_ROWS = 20
PREVIEW_ROWS = 10


class RunSqlTool(Tool):
    name = "run_sql"
    rerunnable = True
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

    def __init__(self, db: Database, results: ResultStore | None = None) -> None:
        self.db = db
        # 和 export_csv、界面共用（app.py 注入）
        self.results = results if results is not None else ResultStore()

    def run(self, args: Args) -> ToolOutput:
        sql = _validate(args.sql)
        result = self.db.query(sql, max_rows=FETCH_ROWS)
        if not result.columns or not result.rows:
            return ToolOutput(_format_empty(result), _summarize(result))   # 空结果不占编号
        table = self.results.add(sql, result)
        return ToolOutput(_format(table.ref, result), _summarize(result, table.ref), details=table)


# ---------------------------------------------------------------- 校验
def _validate(raw: str) -> str:
    sql = raw.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("SQL 为空。")

    # 去掉注释再判断，防 `/* select */ delete ...`
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
    """结果被清理后留下的线索。一行几列的小结果（COUNT / SUM）直接把值写进来，线索就等于原件。"""
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

