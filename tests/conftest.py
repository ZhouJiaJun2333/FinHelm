"""所有测试共用的隔离：不碰真实的长期记忆和知识库索引（~/.finhelm），默认不注册 ask_user、不挂知识库。

环境变量优先于 .env，所以本机 .env 里开没开都不影响测试。要测的用例显式传 memory_enabled=True / ask_user=True。
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "finhelm-home"))
    monkeypatch.setenv("ASK_USER", "false")
    monkeypatch.setenv("DOCS_DIRS", "")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("RAG_DIR", str(tmp_path / "finhelm-rag"))
