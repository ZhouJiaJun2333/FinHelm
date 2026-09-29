"""Skills：目录只进名字和描述，正文 load_skill 按需读；项目的盖过内置的；要的工具不在就不列。不需要数据库和 Docker。"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.prompts import build_system_prompt
from data_agent.settings import Settings
from data_agent.skills import BUILTIN, load_skills, usable
from data_agent.tools.load_skill import LoadSkillTool

from fakes import ScriptedProvider


def _write(root: Path, name: str, front: str, body: str = "正文") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{front}\n---\n{body}\n", encoding="utf-8")
    return d


def test_内置的meta分析技能_要R():
    skills, problems = load_skills([BUILTIN])
    meta = next(s for s in skills if s.name == "meta-analysis")
    assert problems == [] and meta.tools == ("run_r",)
    assert "fh_help()" in meta.body() and "---" not in meta.body().splitlines()[0]


def test_项目的盖过内置的_同名只留一个(tmp_path):
    _write(tmp_path, "meta-analysis", "name: meta-analysis\ndescription: 我们课题组的做法", "用 OR")
    skills, _ = load_skills([tmp_path, BUILTIN])
    [meta] = [s for s in skills if s.name == "meta-analysis"]
    assert meta.description == "我们课题组的做法" and meta.body() == "用 OR"


@pytest.mark.parametrize("front, why", [
    ("name: other\ndescription: x", "目录名"),
    ("name: Bad_Name\ndescription: x", "目录名"),
    ("name: broken", "description"),
])
def test_写坏的技能跳过_说明原因_不影响别的(tmp_path, front, why):
    name = "bad_name" if "Bad" in front else "broken"
    _write(tmp_path, name, front)
    _write(tmp_path, "good", "name: good\ndescription: 好的")
    skills, problems = load_skills([tmp_path])
    assert [s.name for s in skills] == ["good"]
    assert len(problems) == 1 and why in problems[0]


def test_没有frontmatter也算写坏(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "SKILL.md").write_text("# 只有正文\n", encoding="utf-8")
    assert load_skills([tmp_path])[0] == []


def test_要的工具不在就不列(tmp_path):
    _write(tmp_path, "needs-r", "name: needs-r\ndescription: x\ntools: [run_r, run_python]")
    _write(tmp_path, "any", "name: any\ndescription: y")
    skills, _ = load_skills([tmp_path])
    assert [s.name for s in usable(skills, ["run_python"])] == ["any"]
    assert [s.name for s in usable(skills, ["run_python", "run_r"])] == ["any", "needs-r"]


def test_提示词只列名字和描述_没有技能就没有那一节(tmp_path):
    _write(tmp_path, "report", "name: report\ndescription: 写月报时用", "很长的正文很长的正文")
    skills, _ = load_skills([tmp_path])
    prompt = build_system_prompt(["run_python"], skills=skills)
    assert "## 技能" in prompt and "- `report`：写月报时用" in prompt and "很长的正文" not in prompt
    assert "## 技能" not in build_system_prompt(["run_python"])


def test_load_skill_读正文_名字不对报错列出有哪些(tmp_path):
    _write(tmp_path, "report", "name: report\ndescription: x", "第一步……")
    tool = LoadSkillTool(load_skills([tmp_path])[0])
    out = tool.run(tool.Args(name="report"))
    assert out.content == "[技能 report]\n第一步……" and "report" in out.summary
    with pytest.raises(LookupError, match="report"):
        tool.run(tool.Args(name="没有"))


def test_组装_项目技能从AGENTS_md旁边的目录读_没有能用的就不注册工具(tmp_path):
    _write(tmp_path / ".agents" / "skills", "report", "name: report\ndescription: 写月报时用")
    base = dict(project_dir=str(tmp_path), database_url="", r_sandbox=False)
    app = build_application(Settings(**base, python_sandbox=True), llm=ScriptedProvider())
    assert [s.name for s in app.skills] == ["report"], "meta-analysis 要 R，没开就不列"
    assert "load_skill" in app.tools and "`report`" in app.agent.system_prompt

    (tmp_path / ".agents" / "skills" / "report" / "SKILL.md").unlink()
    app = build_application(Settings(**base, python_sandbox=False), llm=ScriptedProvider())
    assert app.skills == [] and "load_skill" not in app.tools and "## 技能" not in app.agent.system_prompt


def test_CLI_skill命令把技能全文拼在消息前面(monkeypatch, tmp_path, capsys):
    from data_agent import cli

    _write(tmp_path / ".agents" / "skills", "report", "name: report\ndescription: x", "先列指标")
    app = build_application(Settings(project_dir=str(tmp_path), database_url="", r_sandbox=False,
                                     python_sandbox=False), llm=ScriptedProvider())
    sent = []
    monkeypatch.setattr(app.agent, "run", lambda text: sent.append(text) or "好")
    assert cli.handle_command("/skill:report 做九月的", app)
    assert sent == ["[用户指定按技能 report 来做，下面是技能全文]\n先列指标\n\n做九月的"]

    assert cli.handle_command("/skill:report", app) and cli.handle_command("/skill:没有 x", app)
    assert len(sent) == 1, "没写要做的事、技能不存在：只提示，不发给模型"
    cli.handle_command("/skills", app)
    assert "report" in capsys.readouterr().out
