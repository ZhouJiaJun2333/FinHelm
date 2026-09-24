"""每个模块单独 import 都不能炸。

循环导入只在「某个特定的导入顺序」下才暴露：测试和 CLI 恰好先导入了 core，
就一直发现不了，直到有人先 import settings 或 llm。这里每个模块都在一个
**全新的进程**里单独导入一次，顺序问题无处可藏。
"""

from __future__ import annotations

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
