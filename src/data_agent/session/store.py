"""会话目录：一次对话在磁盘上的家。

    sessions/<id>/
        session.jsonl    对话历史（消息 + 标记），每轮成功之后追加
        results.jsonl    查询结果 r1、r2…（tools/sql/results.py 写）
        exports/         /save 和 export_csv 写的 CSV
        work/            run_python 沙箱的工作目录（图表在 work/figures/）

只记提交了的历史：Agent.run 是事务，失败的一轮会回滚，来一条写一条会在日志里留下半截回合
（pi 不回滚，所以它来一条写一条）。/reset 在日志里记一行 reset。第一次真要写的时候才建目录。
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.context import Entry
from .codec import decode, encode

VERSION = 1


class Session:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.id = root.name
        self._written = 0            # 历史里已经写进日志的条数（从上一次 reset 算起）

    # ------------------------------------------------------------ 路径
    @property
    def log_path(self) -> Path:
        return self.root / "session.jsonl"

    @property
    def results_path(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    # ------------------------------------------------------------ 新建 / 打开
    @classmethod
    def create(cls, base: Path) -> "Session":
        """只定下目录名，还不建目录。id 按时间排序，后缀防同一秒撞名。"""
        return cls(base / f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}")

    @classmethod
    def open(cls, base: Path, session_id: str | None = None) -> "Session":
        """给了 id 就是那个，没给就是最近的一个。接着写之前先 load()，sync() 才知道从哪接着写。"""
        if session_id:
            root = base / session_id
            if not (root / "session.jsonl").exists():
                raise FileNotFoundError(f"没有这个会话：{root}")
            return cls(root)
        logs = sorted(base.glob("*/session.jsonl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            raise FileNotFoundError(f"{base} 下还没有会话")
        return cls(logs[-1].parent)

    # ------------------------------------------------------------ 读
    def load(self) -> list[Entry]:
        """读回最后一次 reset 之后的条目。最后一行是崩溃时写了一半的就跳过；中间的行坏了照常报错。"""
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        if header.get("type") != "session" or header.get("version") != VERSION:
            raise ValueError(f"不认识的会话日志格式：{self.log_path}（header：{header}）")
        entries: list[Entry] = []
        for n, line in enumerate(lines[1:], start=2):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if n == len(lines):
                    break
                raise ValueError(f"{self.log_path} 第 {n} 行坏了") from None
            if data["type"] == "reset":
                entries = []
            else:
                entries.append(decode(data))
        self._written = len(entries)
        return entries

    # ------------------------------------------------------------ 写
    def sync(self, history: list[Entry]) -> None:
        """把还没写过的条目追加进日志。history 只会往后追加，或者被清空（/reset）。"""
        if len(history) < self._written:
            if history:
                raise ValueError("历史变短了但没有清空 —— 只追加的历史不该出现这种情况")
            self._append([{"type": "reset"}])
            self._written = 0
            return
        new = history[self._written:]
        if new:
            self._append([encode(e) for e in new])
            self._written = len(history)

    def _append(self, records: list[dict[str, Any]]) -> None:
        if not self.log_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            records = [{"type": "session", "version": VERSION, "id": self.id,
                        "created": datetime.now().isoformat(timespec="seconds")}, *records]
        with open(self.log_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
