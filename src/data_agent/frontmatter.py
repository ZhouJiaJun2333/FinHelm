"""Markdown 开头 --- 包起来的几行 key: value。技能（SKILL.md）和长期记忆共用。

只认单行的值，够用了，不引入 YAML 依赖。
"""

from __future__ import annotations


def split(text: str) -> tuple[dict[str, str], str]:
    """(字段, 正文)。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("开头要有 --- 包起来的 name、description")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        raise ValueError("frontmatter 没有结尾的 ---") from None
    fields = {}
    for line in lines[1:end]:
        if line.strip() and not line.lstrip().startswith("#"):
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"看不懂这一行：{line}")
            fields[key.strip()] = value.strip().strip("'\"")
    return fields, "\n".join(lines[end + 1:])


def render(fields: dict[str, str], body: str) -> str:
    """split 的反操作。值里的换行压成空格：只认单行的值。"""
    head = "\n".join(f"{k}: {' '.join(str(v).split())}" for k, v in fields.items())
    return f"---\n{head}\n---\n{body.strip()}\n"
