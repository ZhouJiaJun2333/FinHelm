"""工具 4：把 run_sql 的某个结果导出成 CSV，交给用户。

只在用户要的时候写文件（说「导出」「下载」「存成表格」，或者在终端里敲 /save r3）。
查出来的大结果不自动落盘：界面本来就拿着完整数据，模型要看更多就改 SQL 重查，
落盘只剩「交给用户」这一个用途 —— 那就按需做。

导出不用界面手里那份，而是**按编号找到当时的 SQL，重新跑一遍**：
    · 工具层不用保存大结果，只记「编号 → SQL」
    · 可以比界面多导一些行（界面最多 1 万行）
    · 前提是两次之间数据没变。分析库是离线灌的，没问题；接实时库时导出的是「导出那一刻」的数据，
      回复里会写明
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ...db.connection import Database
from ...core.tools import Tool

EXPORT_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class Exported:
    path: Path
    rows: int
    columns: int
    truncated: bool          # 超过 EXPORT_ROWS，只导了前面这些

    def describe(self, ref: str) -> str:
        # 写绝对路径：相对路径是相对「启动程序时所在的目录」，用户未必知道是哪
        text = f"已把 {ref} 导出到 {self.path.resolve()}（{self.rows} 行 × {self.columns} 列，按导出时的数据重新查询）"
        if self.truncated:
            text += f"。⚠️ 结果超过 {EXPORT_ROWS} 行，只导出了前 {EXPORT_ROWS} 行"
        return text


def export_query(db: Database, sql: str, out_dir: Path, filename: str) -> Exported:
    """重跑 sql，结果写成 out_dir/filename。同名文件已存在就加 -1、-2… 不覆盖。"""
    result = db.query(sql, max_rows=EXPORT_ROWS)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _free_path(out_dir, filename)
    # utf-8-sig：带 BOM，Excel 打开中文不乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result.columns)
        writer.writerows(result.rows)
    return Exported(path, result.row_count, len(result.columns), result.truncated)


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
    # rerunnable 保持默认 False：重跑一次会再写一个文件，它的结果不能清理成「重新调用即可」
    description = (
        "把 run_sql 的某个结果（按编号，比如 r3）导出成 CSV 文件，交给用户。"
        "只在用户要导出、下载、保存成文件时用；只是想让用户看到整张表，在回答里写 {{r3}} 就行。"
        f"会按当时的 SQL 重新查一遍，最多 {EXPORT_ROWS} 行。"
    )

    class Args(BaseModel):
        ref: str = Field(description="结果编号，比如 r3")
        filename: str = Field(default="", description="文件名，不填就用编号，比如 r3.csv")

    def __init__(self, db: Database, queries: Mapping[str, str], out_dir: Path) -> None:
        self.db = db
        self.queries = queries        # run_sql 记的「编号 → SQL」，同一个 dict，随查随有
        self.out_dir = out_dir

    def run(self, args: Args) -> str:
        sql = self.queries.get(args.ref.strip())
        if sql is None:
            known = "、".join(self.queries) or "还没有"
            raise ValueError(f"没有编号为 {args.ref} 的结果。本次对话里的编号：{known}")
        return export_query(self.db, sql, self.out_dir, args.filename or f"{args.ref}.csv").describe(args.ref)
