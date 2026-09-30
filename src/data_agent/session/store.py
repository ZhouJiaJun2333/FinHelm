"""会话目录：一次对话在磁盘上的家。

    sessions/<id>/
        session.jsonl    对话历史（消息 + 标记），每轮成功之后追加
        checkpoint.json  没跑完的那一轮的进度（每走一步整个重写），这一轮提交了就删
        meta.json        用户改过的标题（Web 界面的「重命名」）
        results.jsonl    查询结果 r1、r2…（tools/sql/results.py 写）
        subagents/       子 Agent 的过程，一个任务一个 t1.jsonl（delegate 写，界面重新打开时展开看）
        exports/         /save 和 export_csv 写的 CSV
        work/            run_python 沙箱的工作目录（图表在 work/figures/）

删除 = 整个目录挪进 <base>/.trash/，恢复就是挪回来。

只记提交了的历史：Agent.run 是事务，失败的一轮会回滚，来一条写一条会在日志里留下半截回合
（pi 不回滚，所以它来一条写一条）。/reset 在日志里记一行 reset。第一次真要写的时候才建目录。
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.agent import InterruptedTurn, PendingQuestion
from ..core.context import Entry
from .codec import decode, encode

VERSION = 1
TRASH = ".trash"


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
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def results_path(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def subagents_dir(self) -> Path:
        return self.root / "subagents"

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

    # ------------------------------------------------------------ 标题、删除
    def title(self) -> str:
        """用户改过的标题，没改过是空的。"""
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8")).get("title", "")
        except (OSError, ValueError):
            return ""

    def rename(self, title: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps({"title": title}, ensure_ascii=False), encoding="utf-8")

    def trash(self) -> None:
        """挪进 <base>/.trash/，不真删。"""
        if self.root.exists():
            bin_ = self.root.parent / TRASH
            bin_.mkdir(exist_ok=True)
            shutil.move(str(self.root), str(bin_ / self.id))

    @classmethod
    def trashed(cls, base: Path) -> list["Session"]:
        return [cls(p) for p in (base / TRASH).glob("*") if (p / "session.jsonl").exists()]

    @classmethod
    def restore(cls, base: Path, session_id: str) -> "Session":
        src = base / TRASH / session_id
        if not (src / "session.jsonl").exists():
            raise FileNotFoundError(f"回收站里没有这个会话：{session_id}")
        if (base / session_id).exists():
            raise FileExistsError(f"已经有同名的会话：{session_id}")
        shutil.move(str(src), str(base / session_id))
        return cls(base / session_id)

    def checkpoint_question(self) -> str:
        """没跑完的那一轮问的是什么。只读：不校验、不删（load_checkpoint 会删过期的，别的线程可能正在写）。"""
        try:
            first = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))["entries"][0]
            return decode(first).content
        except (OSError, ValueError, KeyError, IndexError, AttributeError):
            return ""

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

    # ------------------------------------------------------------ 检查点
    def save_checkpoint(self, turn: InterruptedTurn | None) -> None:
        """存没跑完的那一轮（None = 删掉）。和 session.jsonl 分开：正式历史只放完整的回合。

        base 记下它接在第几条正式历史后面：这一轮提交之后、删检查点之前崩了，读的时候对不上就知道它过期了。
        """
        if turn is None:
            self.checkpoint_path.unlink(missing_ok=True)
            return
        if not self.log_path.exists():
            self._append([])          # 第一轮就断了：先把会话建起来，--resume 才找得到
        data = {"version": VERSION, "base": self._written, "steps": turn.steps, "reason": turn.reason,
                "entries": [encode(e) for e in turn.entries]}
        if (q := turn.pending) is not None:        # 停在提问上：重启之后还要接着问
            data["pending"] = {"call_id": q.call_id, "name": q.name, "question": q.question,
                               "options": list(q.options)}
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self.checkpoint_path)     # 写到一半崩了也不会留下半个文件

    def load_checkpoint(self) -> InterruptedTurn | None:
        """load() 之后调。接不上现在的正式历史（过期、格式不认识）就删掉，返回 None。"""
        if not self.checkpoint_path.exists():
            return None
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if data.get("version") == VERSION and data["base"] == self._written:
                q = data.get("pending")
                pending = PendingQuestion(q["call_id"], q["name"], q["question"], tuple(q["options"])) if q else None
                return InterruptedTurn(tuple(decode(e) for e in data["entries"]), data["steps"], data["reason"],
                                       pending)
        except (ValueError, KeyError):
            pass
        self.checkpoint_path.unlink()
        return None

    def _append(self, records: list[dict[str, Any]]) -> None:
        if not self.log_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            records = [{"type": "session", "version": VERSION, "id": self.id,
                        "created": datetime.now().isoformat(timespec="seconds")}, *records]
        with open(self.log_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------- 子 Agent 的过程
def save_transcript(path: Path, header: dict[str, Any], entries: list[Entry] | tuple[Entry, ...] = ()) -> None:
    """一个子任务的过程：第一行 header（任务号、类型、标题、任务说明），后面是它的历史条目。整个重写。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "subagent", "version": VERSION, **header}, *(encode(e) for e in entries)]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in lines), encoding="utf-8")
    os.replace(tmp, path)


def load_transcript(path: Path) -> tuple[dict[str, Any], list[Entry]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    if header.get("type") != "subagent" or header.get("version") != VERSION:
        raise ValueError(f"不认识的子 Agent 过程格式：{path}")
    return header, [decode(json.loads(line)) for line in lines[1:] if line.strip()]
