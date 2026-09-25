"""医学科研场景包：不连数据库，工具是 run_python + run_r，文件靠 /attach 上传。不需要 Docker（容器第一次调用才启动）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.cli import parse_paths
from data_agent.domains import get_domain
from data_agent.prompts import build_system_prompt
from data_agent.settings import Settings

from fakes import ScriptedProvider


def research_app(tmp_path: Path, **settings):
    return build_application(Settings(domain="research", **settings), llm=ScriptedProvider(),
                             work_dir=tmp_path / "work")


def test_不连数据库_只有沙箱工具(tmp_path):
    app = research_app(tmp_path)
    try:
        assert app.db is None and app.inspector is None
        assert [t.name for t in app.tools] == ["run_python", "run_r", "read_file"]
        assert set(app.sandboxes) == {"python", "r"}
        assert not any(s.running for s in app.sandboxes.values()), "第一次调用才启动容器"
        assert app.agent.session_context is None
    finally:
        app.close()


def test_沙箱关掉的就不注册(tmp_path):
    app = research_app(tmp_path, r_sandbox=False)
    assert [t.name for t in app.tools] == ["run_python", "read_file"]
    assert "run_r" not in app.agent.system_prompt


def test_提示词_讲文件和模板_不讲SQL(tmp_path):
    prompt = build_system_prompt(get_domain("research"), ["run_python", "run_r"])
    assert "医学统计分析师" in prompt and "inputs/" in prompt and "fh_help()" in prompt
    assert "RevMan 5" in prompt and "不要根据 I² 自动切换" in prompt
    for sql_only in ("run_sql", "list_tables", "SQL 里做完"):
        assert sql_only not in prompt
    steps = [line[:2] for line in prompt.splitlines() if line[:1].isdigit()]
    assert steps == ["1.", "2.", "3.", "4."]


def test_金融场景的提示词没有被R影响():
    assert "run_r" not in build_system_prompt(get_domain("shop"), ["list_tables", "describe_table", "run_sql",
                                                                   "run_python"])


def test_上传_复制进inputs_下一条消息带上文件清单(tmp_path):
    app = research_app(tmp_path)
    original = tmp_path / "纳入研究.xlsx"
    original.write_bytes(b"x" * 3000)
    copied = app.attach([original])
    assert copied == [tmp_path / "work" / "inputs" / "纳入研究.xlsx"] and copied[0].read_bytes() == original.read_bytes()

    first = app.with_uploads("画森林图")
    assert first.startswith("[用户上传了文件，在 inputs/ 下：纳入研究.xlsx（3 KB）]") and first.endswith("画森林图")
    assert app.with_uploads("再来一次") == "再来一次", "只提一次"


def test_上传_有一个找不到就都不复制(tmp_path):
    app = research_app(tmp_path)
    real = tmp_path / "a.csv"
    real.write_text("x")
    with pytest.raises(FileNotFoundError, match="nope.xlsx"):
        app.attach([real, tmp_path / "nope.xlsx"])
    assert not (tmp_path / "work" / "inputs").exists() and app.pending_uploads == []


def test_attach参数_引号里的空格_Windows反斜杠都保留():
    assert parse_paths(r'D:\data\a.xlsx "C:\my files\b c.xlsx"') == [Path(r"D:\data\a.xlsx"), Path(r"C:\my files\b c.xlsx")]
    assert parse_paths("") == []
