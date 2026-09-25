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


def test_回答里的引用展开成结果_找不到的编号照实说():
    from data_agent.tools.sql.run_sql import expand_refs

    tables = {"r1": run(FakeDb(3)).details}
    text = expand_refs("见下表：\n{{r1}}\n还有 {{ r9 }}", tables, lambda t: f"<{t.ref} 共 {t.result.row_count} 行>")
    assert text == "见下表：\n<r1 共 3 行>\n还有 （找不到结果 r9）"


def test_摘要要求里的引用写法没被format吃掉():
    from data_agent.core.context.compaction import DIRECT_OUTPUT, SUMMARY_PROMPT

    assert "{{r3}}" in SUMMARY_PROMPT.format(output_format=DIRECT_OUTPUT)


# ================================================================ 导出
def test_导出按编号重跑当时的SQL_不认识的编号报错并列出已有的(tmp_path):
    from data_agent.tools.sql.export_csv import ExportCsvTool

    db = FakeDb(347)
    sql_tool = RunSqlTool(db)
    sql_tool.execute({"sql": "SELECT name, amount FROM t"})
    export = ExportCsvTool(db, sql_tool.queries, tmp_path)

    out = export.execute({"ref": "r1"})
    assert out.ok and "347 行 × 2 列" in out.content
    assert db.max_rows_seen[-1] > FETCH_ROWS           # 导出可以比界面多拿
    text = (tmp_path / "r1.csv").read_text(encoding="utf-8-sig")
    assert text.splitlines()[:2] == ["name,amount", "客户0,0"]
    assert (tmp_path / "r1.csv").read_bytes().startswith(b"\xef\xbb\xbf")   # 带 BOM，Excel 不乱码

    bad = export.execute({"ref": "r9"})
    assert not bad.ok and "r1" in bad.content


def test_导出文件名不能带路径_重名不覆盖(tmp_path):
    from data_agent.tools.sql.export_csv import export_query

    first = export_query(FakeDb(2), "SELECT", tmp_path, "../../外面/客户清单")
    second = export_query(FakeDb(2), "SELECT", tmp_path, "客户清单.csv")
    assert first.path == tmp_path / "客户清单.csv"
    assert second.path == tmp_path / "客户清单-1.csv"
