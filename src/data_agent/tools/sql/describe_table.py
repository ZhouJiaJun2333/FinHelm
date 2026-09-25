"""describe_table：看一张表的详细结构。和 list_tables 分开是渐进式披露：先看有哪些表，再看需要的那张。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.tools import Tool, ToolOutput
from ...db.connection import Database
from ...db.introspection import SchemaInspector
from .results import markdown_table


class DescribeTableTool(Tool):
    name = "describe_table"
    rerunnable = True
    description = (
        "查看一张表的详细结构：列名、类型、是否可空、列注释、主键外键约束，"
        "以及几行样例数据。**写任何 SQL 之前都应该先查这个**，不要凭空猜列名。"
        "外键信息会告诉你这张表和别的表该怎么 JOIN。"
    )

    class Args(BaseModel):
        table: str = Field(
            description="表名，可以带 schema 前缀，例如 'shop.orders' 或直接 'orders'"
        )
        sample_rows: int = Field(
            default=3, ge=0, le=20, description="附带几行样例数据，0 表示不要"
        )

    def __init__(self, db: Database, inspector: SchemaInspector,
                 default_schema: str = "shop") -> None:
        self.db = db
        self.inspector = inspector
        self.default_schema = default_schema

    def run(self, args: Args) -> ToolOutput:
        schema, table = self._split(args.table)

        columns = self.inspector.columns_of(schema, table)
        if not columns:
            known = ", ".join(t.qualified_name for t in self.inspector.list_tables())
            raise LookupError(f"表 {schema}.{table} 不存在。已知的表：{known}")

        lines = [f"### {schema}.{table}", "", "| 列名 | 类型 | 可空 | 说明 |", "|:--|:--|:--|:--|"]
        for c in columns:
            lines.append(
                f"| {c.name} | {c.data_type} | {'是' if c.nullable else '否'} | {c.comment or '—'} |"
            )

        constraints = self.inspector.constraints_of(schema, table)
        if constraints:
            lines += ["", "**约束**（外键说明了怎么 JOIN）："]
            lines += [f"- {c}" for c in constraints]

        if args.sample_rows > 0:
            lines += ["", f"**样例数据（{args.sample_rows} 行）**：", ""]
            # 表已经确认存在（columns_of 查到了），用标识符引用拼接
            result = self.db.query(
                f'SELECT * FROM "{schema}"."{table}" LIMIT {args.sample_rows}'
            )
            lines.append(markdown_table(result.columns, result.rows) if result.rows else "（无数据）")

        # 线索里放全部列名：清理之后模型照样能写 SQL
        summary = f"{schema}.{table} 的表结构，{len(columns)} 列：" + ", ".join(
            c.name for c in columns
        )
        return ToolOutput("\n".join(lines), summary)

    def _split(self, raw: str) -> tuple[str, str]:
        raw = raw.strip().strip('"')
        if "." in raw:
            schema, table = raw.split(".", 1)
            return schema.strip('"'), table.strip('"')
        return self.default_schema, raw

