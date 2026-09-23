"""工具 3：执行只读 SQL。

这是整个 Agent 唯一真正「产出结论」的工具，也是唯一有安全风险的工具。

安全设计是**三道独立的防线**，不指望任何单点：
    1. 这里：只允许单条 SELECT / WITH 语句，拒绝多语句和一切写操作关键字
    2. 连接层（db/connection.py）：只读事务 + 语句超时 + 强制行数上限
    3. 数据库（docker/initdb/03_readonly_role.sql）：agent_ro 账号物理上没有写权限

第 1 道是可以被绕过的（注释、大小写、奇怪语法…），所以第 3 道才是真正的底线。
**永远不要只靠关键字黑名单来做安全。**
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ...db.connection import Database, QueryResult
from ..base import Tool

# 语句必须以这些开头
ALLOWED_STARTS = ("select", "with")

# 一眼就该拒绝的写操作。这是「减少误伤」，不是安全边界。
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|"
    r"copy|vacuum|reindex|call|do|set|reset)\b",
    re.IGNORECASE,
)


class RunSqlTool(Tool):
    name = "run_sql"
    description = (
        "在分析库上执行一条只读 SQL（只能是 SELECT 或 WITH 开头的单条语句），返回结果表。"
        "执行前请先用 describe_table 确认列名和外键。"
        "结果行数有上限，做聚合分析时请在 SQL 里就把数据聚合好，不要 SELECT * 全量拉取。"
    )

    class Args(BaseModel):
        sql: str = Field(description="要执行的 SQL，单条语句，不要加结尾分号以外的内容")
        max_rows: int = Field(
            default=100, ge=1, le=500, description="最多返回多少行，默认 100"
        )
        purpose: str = Field(
            default="",
            description="一句话说明这条 SQL 想回答什么问题。会记进日志，方便你和用户回溯。",
        )

    def __init__(self, db: Database) -> None:
        self.db = db

    def run(self, args: Args) -> str:
        sql = _validate(args.sql)
        result = self.db.query(sql, max_rows=args.max_rows)
        return _format(result, args.max_rows)


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
def _format(result: QueryResult, max_rows: int) -> str:
    if not result.columns:
        return "语句执行成功，但没有返回结果集。"
    if result.row_count == 0:
        return f"查询成功，但结果为空（0 行，耗时 {result.elapsed_ms}ms）。"

    head = "| " + " | ".join(result.columns) + " |"
    sep = "|" + "|".join(["---"] * len(result.columns)) + "|"
    body = [
        "| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |"
        for row in result.rows
    ]
    table = "\n".join([head, sep, *body])

    footer = f"\n\n返回 {result.row_count} 行，耗时 {result.elapsed_ms}ms。"
    if result.truncated:
        footer += (
            f"\n⚠️ 结果被截断到 {max_rows} 行，还有更多数据。"
            "如果你需要的是汇总结论，请在 SQL 里用 GROUP BY / 聚合函数，而不是拉全量明细。"
        )
    return table + footer
