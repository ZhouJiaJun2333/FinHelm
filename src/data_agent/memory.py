"""长期记忆：跨会话记住用户定过的口径、偏好、对做法的纠正（学 Claude Code 的 auto memory）。

一条记忆一个 Markdown 文件（frontmatter 写 name / description / updated，后面是正文），分两层：
    用户级  <MEMORY_DIR>/memory/                       个人偏好，所有项目都用
    项目级  <MEMORY_DIR>/projects/<项目路径>/memory/     这个项目的口径、事实
不放进项目目录：AGENTS.md 是大家共享、进版本库的项目说明，记忆是替这个人记的笔记（Claude Code、Codex 也放家目录）。

文件是唯一的数据，人可以直接改、删；MEMORY.md 由程序生成，给人翻看。
检索不用向量：索引（每条一行摘要）进系统提示词，模型看摘要决定要不要读全文。几十条的量级这就够了。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .frontmatter import render, split

SCOPES = {"project": "项目", "user": "用户"}
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
INDEX_LIMIT = 100          # 索引最多列这么多条（大约 4k token），多了提示用户整理


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    scope: str
    name: str
    description: str
    updated: date
    path: Path


def age(updated: date, today: date) -> str:
    """写成「3 天前」而不是日期：模型对「47 天前」会想到可能过期，对原始日期不会（Claude Code 的评测结论）。"""
    days = (today - updated).days
    return "今天" if days <= 0 else "昨天" if days == 1 else f"{days} 天前"


class MemoryStore:
    """一个作用域的目录。"""

    def __init__(self, scope: str, root: Path) -> None:
        self.scope, self.root = scope, root

    def entries(self) -> list[MemoryEntry]:
        """按更新时间从新到旧。写坏的文件跳过（人手改坏了不该让程序起不来）。"""
        out = []
        for path in sorted(self.root.glob("*.md")) if self.root.is_dir() else []:
            if path.name == "MEMORY.md":
                continue
            try:
                fields, _ = split(path.read_text(encoding="utf-8-sig"))
                updated = (date.fromisoformat(fields["updated"]) if "updated" in fields
                           else datetime.fromtimestamp(path.stat().st_mtime).date())
            except (ValueError, OSError):
                continue
            out.append(MemoryEntry(self.scope, path.stem, fields.get("description", ""), updated, path))
        return sorted(out, key=lambda e: e.updated, reverse=True)

    def read(self, name: str) -> str:
        path = self._path(name)
        if not path.is_file():
            known = ", ".join(e.name for e in self.entries()) or "（没有）"
            raise LookupError(f"{self.scope} 里没有叫 {name} 的记忆。有这些：{known}")
        return path.read_text(encoding="utf-8-sig")

    def save(self, name: str, description: str, content: str, today: date) -> bool:
        """写入（同名覆盖）。返回是不是新建的。先写临时文件再改名：写到一半断了不会留下半个文件。"""
        path = self._path(name)
        created = not path.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(render({"name": name, "description": description, "updated": today.isoformat()}, content),
                       encoding="utf-8")
        os.replace(tmp, path)
        self._write_index()
        return created

    def delete(self, name: str) -> None:
        self.read(name)          # 不存在就报错，并列出有哪些
        self._path(name).unlink()
        self._write_index()

    def _path(self, name: str) -> Path:
        if not NAME.fullmatch(name):
            raise ValueError(f"name 只用小写英文字母、数字、连字符，比如 big-customer（现在是 {name!r}）")
        return self.root / f"{name}.md"

    def _write_index(self) -> None:
        lines = [f"- [{e.name}]({e.name}.md)（{e.updated}）：{e.description}" for e in self.entries()]
        (self.root / "MEMORY.md").write_text(
            "<!-- 程序生成的索引，改记忆请改对应的文件 -->\n" + "\n".join(lines) + "\n", encoding="utf-8")


def project_slug(project_dir: Path) -> str:
    """项目目录的绝对路径 → 一个目录名（学 Claude Code 的 ~/.claude/projects/<slug>/）。"""
    return re.sub(r"[^\w]+", "-", str(project_dir.resolve())).strip("-")


class Memory:
    """两层记忆。"""

    def __init__(self, user: MemoryStore, project: MemoryStore) -> None:
        self.stores = {"project": project, "user": user}

    @classmethod
    def open(cls, root: Path, project_dir: Path) -> "Memory":
        return cls(MemoryStore("user", root / "memory"),
                   MemoryStore("project", root / "projects" / project_slug(project_dir) / "memory"))

    def store(self, scope: str) -> MemoryStore:
        return self.stores[scope]

    def entries(self) -> list[MemoryEntry]:
        """项目的在前：项目口径优先于个人偏好。"""
        return [e for s in self.stores.values() for e in s.entries()]

    def index(self, today: date) -> str:
        """拼进系统提示词的目录：会话开始时算一次，整个会话不变。"""
        entries = self.entries()
        if not entries:
            return "[长期记忆] 还没有。"
        lines = [f"- [{SCOPES[e.scope]}] {e.name}（{age(e.updated, today)}）：{e.description}"
                 for e in entries[:INDEX_LIMIT]]
        if len(entries) > INDEX_LIMIT:
            lines.append(f"……还有 {len(entries) - INDEX_LIMIT} 条没列出。记忆太多了，提醒用户用 /memory 整理。")
        return "[长期记忆] 以前的对话里记下的，每行一条的摘要（要细节用 read_memory 读全文）：\n" + "\n".join(lines)
