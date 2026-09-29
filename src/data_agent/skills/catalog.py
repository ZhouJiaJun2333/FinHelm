"""技能目录：启动时只读每个 SKILL.md 的 name / description，正文等模型要用时再读（Agent Skills 规范，学 pi）。

一个技能是一个目录，里面有 SKILL.md：开头 --- 包起来的几行 key: value，后面是正文。
自定义字段 tools：这个技能要用到的工具，环境里缺了就不列出来（比如没开 R 沙箱，meta 分析就不出现）。
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..frontmatter import split

BUILTIN = Path(__file__).parent / "builtin"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")     # 规范：小写字母、数字、连字符，和目录名一致


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path                      # SKILL.md
    tools: tuple[str, ...] = ()

    def body(self) -> str:
        """正文（去掉开头的 frontmatter）。每次现读：改了 SKILL.md 不用重启。"""
        return split(self.path.read_text(encoding="utf-8-sig"))[1].strip()


def _parse(path: Path) -> Skill:
    fields, _ = split(path.read_text(encoding="utf-8-sig"))
    name, description = fields.get("name", ""), fields.get("description", "")
    if name != path.parent.name or not NAME.fullmatch(name):
        raise ValueError(f"name 要和目录名一样，只用小写字母、数字、连字符（现在是 {name!r}）")
    if not description:
        raise ValueError("缺 description：模型靠它判断什么时候用这个技能")
    tools = tuple(t.strip() for t in fields.get("tools", "").strip("[]").split(",") if t.strip())
    return Skill(name, description, path, tools)


def load_skills(dirs: Iterable[Path]) -> tuple[list[Skill], list[str]]:
    """按顺序扫描，同名的前面优先（项目的盖过内置的）。返回 (技能, 跳过了的和原因)。

    写坏的技能不让程序起不来，跳过并说明原因，由界面告诉用户。
    """
    skills: dict[str, Skill] = {}
    problems = []
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*/SKILL.md")):
            try:
                skill = _parse(path)
            except (ValueError, OSError) as exc:
                problems.append(f"{path}：{exc}")
                continue
            skills.setdefault(skill.name, skill)
    return list(skills.values()), problems


def usable(skills: Iterable[Skill], tools: Collection[str]) -> list[Skill]:
    """要用的工具都注册了的技能。"""
    return [s for s in skills if set(s.tools) <= set(tools)]
