"""工具 1：列出库里有哪些表。

没有这个工具，模型就是个瞎子 —— 你问「我们有什么数据」，它只能反问你。
**模型的世界 = 它的工具能感知到的 + 上下文里写着的**，仅此而已。
"""

from __future__ import annotations

from pydantic import BaseModel

from ...db.introspection import SchemaInspector
from ..base import Tool, ToolOutput


class ListTablesTool(Tool):
    name = "list_tables"
    description = (
        "列出数据库里所有可查询的表，返回表名、大致行数和表注释（业务含义）。"
        "当你不确定有哪些数据、或者用户问『我们有什么数据』时，先用这个。"
    )

    class Args(BaseModel):
        """不需要参数。"""

    def __init__(self, inspector: SchemaInspector) -> None:
        self.inspector = inspector

    def run(self, args: Args) -> "str | ToolOutput":
        tables = self.inspector.list_tables()
        if not tables:
            return "数据库里没有找到任何表。"

        lines = [f"共 {len(tables)} 张表：", ""]
        lines.append("| 表名 | 大致行数 | 说明 |")
        lines.append("|:--|--:|:--|")
        for t in tables:
            lines.append(f"| {t.qualified_name} | {t.estimated_rows} | {t.comment or '—'} |")
        lines.append("")
        lines.append("注：行数是 PG 的统计估算值，不是精确值；要精确数字请用 run_sql 查 count(*)。")
        lines.append("需要列名和类型，用 describe_table。")
        summary = f"共 {len(tables)} 张表：" + ", ".join(t.qualified_name for t in tables)
        return ToolOutput(True, "\n".join(lines), summary)
