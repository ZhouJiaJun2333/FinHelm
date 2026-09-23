"""Schema 自省：从 information_schema / pg_catalog 里读出库的结构。

这是数据分析 Agent 的「眼睛」。模型写 SQL 之前必须先看到表结构，
否则它只能瞎猜列名 —— 然后你会看到一堆 column "xxx" does not exist。

顺便把 COMMENT 也读出来：表注释和列注释是**最廉价的业务上下文**，
写在数据库里，Agent 自动就能读到，不用塞进提示词。
"""

from __future__ import annotations

from dataclasses import dataclass

from .connection import Database


@dataclass(frozen=True, slots=True)
class TableInfo:
    schema: str
    name: str
    comment: str | None
    estimated_rows: int

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    default: str | None
    comment: str | None


class SchemaInspector:
    def __init__(self, db: Database, schemas: tuple[str, ...] = ("shop",)) -> None:
        self.db = db
        self.schemas = schemas

    # ------------------------------------------------------------------
    def list_tables(self) -> list[TableInfo]:
        rows = self.db.query_dicts(
            """
            SELECT n.nspname                        AS schema,
                   c.relname                        AS name,
                   obj_description(c.oid, 'pg_class') AS comment,
                   GREATEST(c.reltuples, 0)::bigint AS estimated_rows
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r', 'p', 'v', 'm')
              AND n.nspname = ANY(%s)
            ORDER BY n.nspname, c.relname
            """,
            (list(self.schemas),),
        )
        return [TableInfo(**r) for r in rows]

    # ------------------------------------------------------------------
    def columns_of(self, schema: str, table: str) -> list[ColumnInfo]:
        rows = self.db.query_dicts(
            """
            SELECT a.attname                                       AS name,
                   format_type(a.atttypid, a.atttypmod)            AS data_type,
                   NOT a.attnotnull                                AS nullable,
                   pg_get_expr(d.adbin, d.adrelid)                 AS default,
                   col_description(a.attrelid, a.attnum)           AS comment
            FROM pg_attribute AS a
            JOIN pg_class     AS c ON c.oid = a.attrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            LEFT JOIN pg_attrdef AS d
                   ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE n.nspname = %s AND c.relname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (schema, table),
        )
        return [ColumnInfo(**r) for r in rows]

    # ------------------------------------------------------------------
    def constraints_of(self, schema: str, table: str) -> list[str]:
        """主键 / 外键 / 唯一约束的人类可读描述。

        外键尤其重要 —— 它告诉模型这些表**怎么 JOIN**。
        """
        rows = self.db.query_dicts(
            """
            SELECT pg_get_constraintdef(con.oid) AS definition,
                   con.contype                   AS kind
            FROM pg_constraint AS con
            JOIN pg_class      AS c ON c.oid = con.conrelid
            JOIN pg_namespace  AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            ORDER BY con.contype
            """,
            (schema, table),
        )
        label = {"p": "PRIMARY KEY", "f": "FOREIGN KEY", "u": "UNIQUE", "c": "CHECK"}
        return [f"{label.get(r['kind'], r['kind'])}: {r['definition']}" for r in rows]

    # ------------------------------------------------------------------
    def overview(self) -> str:
        """一段塞进系统提示词的库概览。

        只放表名、行数、注释 —— 详细列信息让模型用 describe_table 自己查。
        全部塞进提示词会又长又贵，而且模型反而抓不住重点。
        """
        tables = self.list_tables()
        if not tables:
            return "[数据库] 没有找到任何表。"
        lines = [f"[数据库概览] schema: {', '.join(self.schemas)}"]
        for t in tables:
            note = f" —— {t.comment}" if t.comment else ""
            lines.append(f"  · {t.qualified_name}（约 {t.estimated_rows} 行）{note}")
        lines.append("需要列名和类型时，用 describe_table 查，不要凭空猜。")
        return "\n".join(lines)
