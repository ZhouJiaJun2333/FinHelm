"""read_file 和路径解析：/data、/work、相对路径、宿主机绝对路径；分页、长行、二进制。不需要 Docker。"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent.tools.paths import SandboxPaths
from data_agent.tools.read_file import MAX_CHARS, ReadFileTool

MANUAL = Path("data/dabstep/context/manual.md")


@pytest.fixture
def dirs(tmp_path):
    work, data = tmp_path / "work", tmp_path / "data"
    (work / "inputs").mkdir(parents=True)
    data.mkdir()
    (data / "manual.md").write_text("\n".join(f"第 {i} 行" for i in range(1, 11)), encoding="utf-8")
    (work / "inputs" / "说明.txt").write_text("你好\n世界", encoding="utf-8")
    return work, data


def tool(work, data=None) -> ReadFileTool:
    return ReadFileTool(SandboxPaths(work, data))


# ================================================================ 路径
def test_几种写法都认(dirs):
    work, data = dirs
    paths = SandboxPaths(work, data)
    assert paths.resolve("/data/manual.md") == (data / "manual.md").resolve()
    assert paths.resolve("inputs/说明.txt") == paths.resolve("/work/inputs/说明.txt") == paths.resolve(str(work / "inputs" / "说明.txt"))
    assert paths.display(paths.resolve("/data/manual.md")) == "/data/manual.md"
    assert paths.display(paths.resolve("/work/inputs/说明.txt")) == "inputs/说明.txt"


def test_目录外的不给读(dirs, tmp_path):
    work, data = dirs
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    paths = SandboxPaths(work, data)
    for raw in ("../secret.txt", "/data/../secret.txt", str(tmp_path / "secret.txt")):
        with pytest.raises(ValueError, match="只能读"):
            paths.resolve(raw)
    with pytest.raises(ValueError, match="只能读工作目录（inputs/、figures/）下"):
        SandboxPaths(work).resolve("/data/manual.md")          # 没有数据目录时 /data 不存在


# ================================================================ 读
def test_整个读完_带行号(dirs):
    work, data = dirs
    out = tool(work, data).execute({"path": "/data/manual.md"})
    assert not out.is_error
    lines = out.content.splitlines()
    assert lines[0] == "/data/manual.md：共 10 行"
    assert lines[1] == "     1\t第 1 行" and lines[-1] == "    10\t第 10 行"
    assert out.summary == "读了 /data/manual.md 第 1–10 行"


def test_分页_告诉下次从哪行接着读(dirs):
    work, data = dirs
    out = tool(work, data).execute({"path": "/data/manual.md", "offset": 3, "limit": 4})
    assert out.content.splitlines()[0] == "/data/manual.md：共 10 行，这是第 3–6 行"
    assert out.content.endswith("（后面还有 4 行，接着读用 offset=7）")
    assert "offset=20 超出" in tool(work, data).execute({"path": "/data/manual.md", "offset": 20}).content


def test_超过字符上限按整行停_不从中间截(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join("x" * 900 for _ in range(100)), encoding="utf-8")
    out = tool(tmp_path).execute({"path": "big.txt"})
    assert not out.is_error and "输出过长" not in out.content
    shown = [line for line in out.content.splitlines() if "\t" in line]
    assert all(line.endswith("x" * 900) for line in shown) and len(out.content) <= MAX_CHARS + 200
    assert f"接着读用 offset={len(shown) + 1}" in out.content


def test_长行只给开头_二进制不读(tmp_path):
    (tmp_path / "one.json").write_text("[" + "1," * 3000 + "1]", encoding="utf-8")
    assert "这一行共 6003 个字符" in tool(tmp_path).execute({"path": "one.json"}).content
    (tmp_path / "a.xlsx").write_bytes(b"PK\x03\x04\x00\x00binary")
    out = tool(tmp_path).execute({"path": "a.xlsx"})
    assert out.is_error and "run_python" in out.content


@pytest.mark.skipif(not MANUAL.exists(), reason="没下载 DABstep 数据（python -m evals.dabstep.prepare）")
def test_DABstep手册一次读完():
    """加这个工具的起因：2.2 万字符的手册以前要 print 四五段。"""
    out = tool(Path("outputs/work"), MANUAL.parent).execute({"path": "/data/manual.md"})
    assert not out.is_error and "接着读" not in out.content
    assert "Fraud is defined as the ratio of fraudulent volume over total volume" in out.content


def test_能重读_旧结果可以被清理():
    assert ReadFileTool.rerunnable and ReadFileTool.max_output_chars > MAX_CHARS
