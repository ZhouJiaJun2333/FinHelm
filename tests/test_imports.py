"""每个模块单独 import 都不能炸。

循环导入只在「某个特定的导入顺序」下才暴露：测试和 CLI 恰好先导入了 core，
就一直发现不了，直到有人先 import settings 或 llm。这里每个模块都在一个
**全新的进程**里单独导入一次，顺序问题无处可藏。
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
MODULES = sorted(
    ".".join(p.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")
    for p in (SRC / "data_agent").rglob("*.py")
)


@pytest.mark.parametrize("module", MODULES)
def test_单独导入不会循环(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=SRC, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]


def _imported(path: pathlib.Path) -> list[str]:
    """一个文件 import 了项目里的哪些模块（相对导入换算成绝对名）。"""
    package = ".".join(path.relative_to(SRC).parent.parts)
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                out.append(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                out.append(node.module)
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names]
    return [m for m in out if m.startswith("data_agent.")]


def test_core_不依赖包外的模块():
    """分层规则：llm/、tools/、app、cli 依赖 core，反过来不行。

    一旦 core 开始 import 具体实现，「换厂商 / 加工具不用动主循环」就不成立了，
    而且循环导入会回来（以前 core ↔ llm 就是这么绕成环的）。
    """
    bad = [
        f"{p.relative_to(SRC)} → {m}"
        for p in (SRC / "data_agent" / "core").rglob("*.py")
        for m in _imported(p)
        if not m.startswith("data_agent.core")
    ]
    assert not bad, bad
