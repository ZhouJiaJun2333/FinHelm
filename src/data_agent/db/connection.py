"""数据访问层：只读事务 + 语句超时 + 行数上限。账号本身也没有写权限（docker/initdb/03_readonly_role.sql）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql as pgsql
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
    """很薄的 psycopg 封装，出了问题一眼能看到 SQL。"""

    def __init__(self, dsn: str, *, statement_timeout_ms: int = 30_000,
                 search_path: str | None = None) -> None:
        self._dsn = dsn
        self._statement_timeout_ms = statement_timeout_ms
        # SQL 不写 schema 前缀时去哪找表（BIRD 的标准 SQL 都不带前缀）
        self._search_path = search_path

    # ------------------------------------------------------------------
    def _connect(self) -> psycopg.Connection:
        # 连不上就报错，别一直挂着：localhost 可能先解析成 ::1，端口只绑了 127.0.0.1 时会卡住
        conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=10)
        with conn.cursor() as cur:
            # 连的是有写权限的账号，这个连接也只能读
            cur.execute("SET default_transaction_read_only = on")
            cur.execute(f"SET statement_timeout = {self._statement_timeout_ms}")
            if self._search_path:
                cur.execute(pgsql.SQL("SET search_path = {}").format(pgsql.Identifier(self._search_path)))
        return conn

    def ping(self) -> str:
        """返回服务端版本。启动时跑一次。"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def query(self, sql: str, params: tuple[Any, ...] | None = None,
              *, max_rows: int = 200) -> QueryResult:
        """执行一条只读查询。多取一行判断是不是还有更多，只返回 max_rows 行。"""
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
