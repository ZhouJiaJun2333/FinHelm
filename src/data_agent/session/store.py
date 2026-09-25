"""会话目录：一次对话在磁盘上的家。

    sessions/<id>/
        session.jsonl    对话历史（消息 + 标记），每轮**成功之后**追加
        results.jsonl    查询结果 r1、r2…（用户见过的），查一条追加一条（tools/sql/results.py 写）
        exports/         /save 和 export_csv 写的 CSV
        tool-results/    （以后）结果不能重拿的工具，大输出落盘在这里

── 参考 ───────────────────────────────────────────────────────────────
    Claude Code  ~/.claude/projects/<项目>/<id>.jsonl，旁边一个同名目录放 tool-results/ 等附属文件
    pi           ~/.pi/agent/sessions/<项目>/<时间>_<id>.jsonl，第一行是带版本号的 header
我们把日志和附属文件放进同一个目录：找、删、打包都只看一个地方。

── 什么时候写：一轮成功之后，而不是来一条写一条 ─────────────────────────
pi 订阅 message_end 事件，一条消息结束就追加一行；失败的一轮也留在日志里，
它的会话本来就不回滚。我们的 Agent.run 是事务：一轮失败，历史整轮回滚。
来一条写一条的话，日志里会留下内存里已经不存在的半截回合，恢复出来就是坏的
（tool_call 没有结果 → 400）。所以日志只记**提交了的**历史：界面每处理完一次输入，
调一次 sync()，把历史里新增的条目追加进去。程序中途崩了，丢的最多是正在跑的那一轮。

/reset 清空了历史：日志里记一行 reset，读回来时只要最后一次 reset 之后的部分。
编号（r1、r2…）不跟着 reset 回收，和以前一样。

── 什么时候建目录：第一次真要写的时候 ────────────────────────────────
和 pi 一样：打开程序看一眼就退出，不留下空会话。
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

    # ------------------------------------------------------------ 新建 / 打开
    @classmethod
    def create(cls, base: Path) -> "Session":
        """新会话。只定下目录名，还不建目录。id 按时间排序，后缀防同一秒撞名。"""
        return cls(base / f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}")

    @classmethod
    def open(cls, base: Path, session_id: str | None = None) -> "Session":
        """打开已有的会话：给了 id 就是那个，没给就是最近的一个（按日志最后修改时间）。
        接着往里写之前先 load() —— 它记下日志里已经有几条，sync() 才知道从哪接着写。"""
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
        """读回历史：最后一次 reset 之后的条目。

        最后一行可能是崩溃时写了一半的，读不出来就跳过 —— 丢一行总比整个会话打不开强。
        中间的行坏了说明文件被改过，照常报错。
        """
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
        """把 history 里还没写过的条目追加进日志。每处理完一次输入调一次。

        history 只会有两种变化：往后追加（成功的一轮、/compact 加的标记），或者被清空（/reset）。
        失败的一轮已经回滚，长度没变，这里什么都不写。
        """
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
