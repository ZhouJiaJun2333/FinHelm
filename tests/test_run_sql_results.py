"""run_sql 的结果分两份：模型看预览，界面拿完整结果。不需要数据库。"""

from __future__ import annotations

from data_agent.db.connection import QueryResult
from data_agent.tools.sql.run_sql import FETCH_ROWS, FULL_ROWS, PREVIEW_ROWS, RunSqlTool, SqlResult


class FakeDb:
    def __init__(self, n_rows: int, *, truncated: bool = False) -> None:
        self.n_rows = n_rows
        self.truncated = truncated
        self.max_rows_seen: list[int] = []

    def query(self, sql, params=None, *, max_rows=200):
        self.max_rows_seen.append(max_rows)
        rows = [(f"客户{i}", i * 10) for i in range(self.n_rows)]
        return QueryResult(["name", "amount"], rows, self.truncated, elapsed_ms=3)


def run(db, sql="SELECT name, amount FROM t"):
    return RunSqlTool(db).execute({"sql": sql})


def test_小结果原样给模型_带编号():
    out = run(FakeDb(FULL_ROWS))
    assert out.content.startswith(f"结果 r1（{FULL_ROWS} 行 × 2 列")
    assert f"客户{FULL_ROWS - 1}" in out.content
    assert isinstance(out.details, SqlResult) and out.details.ref == "r1"


def test_大结果模型只看前几行_界面拿全部():
    out = run(FakeDb(347))
    assert f"客户{PREVIEW_ROWS - 1}" in out.content
    assert f"客户{PREVIEW_ROWS} " not in out.content
    assert f"还有 {347 - PREVIEW_ROWS} 行没给你看" in out.content
    assert out.details.result.row_count == 347
    assert out.summary == "结果 r1：347 行 × 2 列（name, amount）"


def test_一次取够界面要的行数():
    db = FakeDb(5)
    run(db)
    assert db.max_rows_seen == [FETCH_ROWS]


def test_编号只增不减_空结果和出错不占号():
    tool = RunSqlTool(FakeDb(3))
    assert tool.execute({"sql": "SELECT 1"}).details.ref == "r1"
    assert tool.execute({"sql": "DELETE FROM t"}).ok is False
    tool.db = FakeDb(0)
    assert tool.execute({"sql": "SELECT 1"}).details is None
    tool.db = FakeDb(3)
    assert tool.execute({"sql": "SELECT 1"}).details.ref == "r2"


def test_超过取数上限时告诉模型():
    out = run(FakeDb(50, truncated=True))
    assert f"超过 {FETCH_ROWS} 行" in out.content
