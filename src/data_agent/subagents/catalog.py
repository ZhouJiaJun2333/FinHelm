"""子 Agent 的类型：一个 Markdown 文件一种（学 pi 的 agents/*.md、Claude Code 的 .claude/agents/）。

开头 --- 包起来的字段：name（和文件名一样）、description（主 Agent 靠它选）、tools（能用哪些工具，* 是全部）、
max_steps（步数上限）。正文是这种子 Agent 的角色说明，拼进它的系统提示词。
项目的（.agents/agents/）盖过内置的；要的工具一个都不在就不列。
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..frontmatter import split
from ..skills.catalog import NAME

BUILTIN = Path(__file__).parent / "builtin"
DEFAULT_MAX_STEPS = 20

# 只有主 Agent 有：只分派一层；子 Agent 不能直接问用户，也不写长期记忆
PARENT_ONLY = frozenset({"delegate", "ask_user", "remember"})


@dataclass(frozen=True, slots=True)
class Definition:
    name: str
    description: str
    path: Path
    tools: tuple[str, ...]          # ("*",) 是主 Agent 有的全部
    max_steps: int = DEFAULT_MAX_STEPS

    def role(self) -> str:
        """正文。每次现读：改了文件不用重启。"""
        return split(self.path.read_text(encoding="utf-8-sig"))[1].strip()

    def allowed(self, available: Iterable[str]) -> list[str]:
        """在主 Agent 有的工具里，这种子 Agent 能用哪些（保持主 Agent 那边的顺序）。"""
        return [t for t in available if t not in PARENT_ONLY and ("*" in self.tools or t in self.tools)]


def _parse(path: Path) -> Definition:
    fields, _ = split(path.read_text(encoding="utf-8-sig"))
    name, description = fields.get("name", ""), fields.get("description", "")
    if name != path.stem or not NAME.fullmatch(name):
        raise ValueError(f"name 要和文件名一样，只用小写字母、数字、连字符（现在是 {name!r}）")
    if not description:
        raise ValueError("缺 description：主 Agent 靠它选派给谁")
    tools = tuple(t.strip() for t in fields.get("tools", "").strip("[]").split(",") if t.strip())
    if not tools:
        raise ValueError("缺 tools：写能用的工具名，或者 * 表示全部")
    try:
        max_steps = int(fields.get("max_steps", DEFAULT_MAX_STEPS))
    except ValueError:
        raise ValueError(f"max_steps 要是整数（现在是 {fields['max_steps']!r}）") from None
    return Definition(name, description, path, tools, max_steps)


def load_definitions(dirs: Iterable[Path]) -> tuple[list[Definition], list[str]]:
    """按顺序扫描，同名的前面优先。返回 (类型, 跳过了的和原因)。写坏的跳过，不让程序起不来。"""
    found: dict[str, Definition] = {}
    problems = []
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.md")):
            try:
                definition = _parse(path)
            except (ValueError, OSError) as exc:
                problems.append(f"{path}：{exc}")
                continue
            found.setdefault(definition.name, definition)
    return list(found.values()), problems


def usable(definitions: Iterable[Definition], tools: Collection[str]) -> list[Definition]:
    """至少有一个工具能用的类型。"""
    return [d for d in definitions if d.allowed(tools)]
