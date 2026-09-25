"""数据访问层：所有真正碰数据库的代码都在这一层。

tools/sql/ 下的工具只调用这里的方法，不自己写连接管理。
好处是以后换 MySQL / ClickHouse，只要实现一个同样接口的类，工具代码不用动。

⚠️ 安全设计（两道防线，不依赖任何单点）：
    第一道 —— 这里：只读事务 + 语句超时 + 强制 LIMIT
    第二道 —— 数据库：agent_ro 账号物理上就没有写权限（见 docker/initdb/03_readonly_role.sql）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool          # 是否因为行数上限被截断
    elapsed_ms: int

    @property
    def row_count(self) -> int:
        return len(self.rows)


class Database:
    """一个很薄的 Postgres 封装。

    为什么不用 SQLAlchemy？这个项目的重点是搞懂 Agent，不是 ORM。
    直连 psycopg 代码更少、更透明，出了问题一眼能看到 SQL。
    """

    def __init__(self, dsn: str, *, statement_timeout_ms: int = 30_000) -> None:
        self._dsn = dsn
        self._statement_timeout_ms = statement_timeout_ms

    # ------------------------------------------------------------------
    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, autocommit=True)
        with conn.cursor() as cur:
            # 双保险：即使连的是有写权限的账号，这个连接也只能读
            cur.execute("SET default_transaction_read_only = on")
            cur.execute(f"SET statement_timeout = {self._statement_timeout_ms}")
        return conn

    def ping(self) -> str:
        """连通性检查，返回服务端版本。启动时跑一次，比跑到一半才报错友好。"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def query(self, sql: str, params: tuple[Any, ...] | None = None,
              *, max_rows: int = 200) -> QueryResult:
        """执行一条只读查询。

        max_rows 是硬上限：多取一行用来判断「是不是还有更多」，
        但只返回 max_rows 行。不能让一条 SELECT * 把整个上下文撑爆。
        """
        started = time.perf_counter()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:          # 不返回结果集的语句
                return QueryResult([], [], False, 0)
            columns = [d.name for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)

        truncated = len(rows) > max_rows
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return QueryResult(columns, rows[:max_rows], truncated, elapsed_ms)

    def query_dicts(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict]:
        """内部用（schema 自省等），返回字典列表，不做行数限制。"""
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else []
