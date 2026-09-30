"""export_csv：把某个编号的结果导出成 CSV。只在用户要的时候写文件。

run_sql 的结果按当时的 SQL 重新跑（可以比仓库多导一些行）；前提是数据没变，分析库是离线灌的。
沙箱 save_result() 存的表没有 SQL 可重跑，直接写存下的行。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ...core.tools import Tool
from ...db.connection import Database, QueryResult
from .results import ResultStore, StoredResult

EXPORT_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class Exported:
    path: Path
    rows: int
    columns: int
    truncated: bool          # 超过行数上限，只导了前面这些
    requeried: bool = True   # 按 SQL 重新查的（沙箱存的表是直接写存下的行）

    def describe(self, ref: str) -> str:
        # 写绝对路径：用户未必知道相对的是哪个目录
        how = "，按导出时的数据重新查询" if self.requeried else ""
        text = f"已把 {ref} 导出到 {self.path.resolve()}（{self.rows} 行 × {self.columns} 列{how}）"
        if self.truncated:
            text += f"。⚠️ 结果超过 {self.rows} 行，只导出了前 {self.rows} 行"
        return text


def export_result(db: Database | None, table: StoredResult, out_dir: Path, filename: str) -> Exported:
    """导出一个编号的结果：有 SQL 就重跑，沙箱的表写存下的行。"""
    if table.sql:
        if db is None:
            raise ValueError(f"{table.ref} 是 SQL 查询结果，但现在没有连数据库，没法重新查询。")
        return export_query(db, table.sql, out_dir, filename)
    return _write(table.result, out_dir, filename, requeried=False)


def export_query(db: Database, sql: str, out_dir: Path, filename: str) -> Exported:
    """重跑 sql，结果写成 out_dir/filename。同名文件已存在就加 -1、-2… 不覆盖。"""
    return _write(db.query(sql, max_rows=EXPORT_ROWS), out_dir, filename)


def _write(result: QueryResult, out_dir: Path, filename: str, *, requeried: bool = True) -> Exported:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _free_path(out_dir, filename)
    # utf-8-sig：带 BOM，Excel 打开中文不乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result.columns)
        writer.writerows(result.rows)
    return Exported(path, result.row_count, len(result.columns), result.truncated, requeried)


def _free_path(out_dir: Path, filename: str) -> Path:
    # 只取文件名部分：模型给的名字不能带着路径跑到别的目录去
    name = Path(filename.strip()).name or "result.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    path, stem, n = out_dir / name, Path(name).stem, 1
    while path.exists():
        path, n = out_dir / f"{stem}-{n}.csv", n + 1
    return path


class ExportCsvTool(Tool):
    name = "export_csv"
    # 不是 rerunnable：重跑一次会再写一个文件
    description = (
        "把某个编号的结果（run_sql 的，或 save_result 存的，比如 r3）导出成 CSV 文件，交给用户。"
        "只在用户要导出、下载、保存成文件时用；只是想让用户看到整张表，在回答里写 {{r3}} 就行。"
        f"SQL 的结果会按当时的 SQL 重新查一遍，最多 {EXPORT_ROWS} 行。"
    )

    class Args(BaseModel):
        ref: str = Field(description="结果编号，比如 r3")
        filename: str = Field(default="", description="文件名，不填就用编号，比如 r3.csv")

    def __init__(self, db: Database, results: ResultStore, out_dir: Path) -> None:
        self.db = db
        self.results = results
        self.out_dir = out_dir

    def cancel(self) -> None:
        self.db.cancel()

    def run(self, args: Args) -> str:
        table = self.results.get(args.ref)
        if table is None:
            known = "、".join(self.results.refs()) or "还没有"
            raise ValueError(f"没有编号为 {args.ref} 的结果。本次对话里的编号：{known}")
        return export_result(self.db, table, self.out_dir,
                             args.filename or f"{table.ref}.csv").describe(table.ref)
