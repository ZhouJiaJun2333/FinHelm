"""所有测试共用的隔离：不碰真实的长期记忆（~/.finhelm）。

环境变量优先于 .env，所以本机 .env 里开没开记忆都不影响测试。要测记忆的用例显式传 memory_enabled=True。
"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "finhelm-home"))
