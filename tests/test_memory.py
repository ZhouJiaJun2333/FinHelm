"""长期记忆：一条一个文件、两层目录、索引进系统提示词、remember / read_memory。不需要数据库和 key。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.memory import INDEX_LIMIT, Memory, MemoryStore, age, project_slug
from data_agent.settings import Settings
from data_agent.tools.memory import ReadMemoryTool, RememberTool
from evals.cases import Case
from evals.runner import Trial, check_memory, memory_files

from fakes import ScriptedProvider

TODAY = date(2026, 9, 29)


def _memory(tmp_path: Path) -> Memory:
    return Memory.open(tmp_path / "home", tmp_path / "project")


def test_一条一个文件_同名覆盖_索引由程序生成(tmp_path):
    store = MemoryStore("project", tmp_path)
    assert store.save("active-customer", "活跃客户 = 近 90 天下过单", "用户 9 月定的", TODAY) is True
    assert store.save("active-customer", "活跃客户 = 近 60 天下过单", "改过一次", TODAY) is False
    [entry] = store.entries()
    assert entry.description == "活跃客户 = 近 60 天下过单" and entry.updated == TODAY
    text = store.read("active-customer")
    assert text.startswith("---\nname: active-customer\n") and "改过一次" in text
    assert "active-customer" in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp")), "临时文件改名之后不留"


def test_删除_不存在的报错并列出有哪些(tmp_path):
    store = MemoryStore("user", tmp_path)
    store.save("style", "先给结论", "x", TODAY)
    with pytest.raises(LookupError, match="style"):
        store.delete("nope")
    store.delete("style")
    assert store.entries() == [] and "style" not in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["大客户", "Big", "a b", "../x", ""])
def test_名字只能是小写英文连字符_不能跑出目录(tmp_path, name):
    with pytest.raises(ValueError, match="name"):
        MemoryStore("user", tmp_path).save(name, "d", "c", TODAY)


def test_手改坏的文件跳过_没写updated用修改时间(tmp_path):
    (tmp_path / "broken.md").write_text("没有 frontmatter", encoding="utf-8")
    (tmp_path / "hand.md").write_text("---\nname: hand\ndescription: 手写的\n---\n正文", encoding="utf-8")
    [entry] = MemoryStore("user", tmp_path).entries()
    assert entry.name == "hand" and entry.updated == date.today()


def test_两层目录_项目按路径分开(tmp_path):
    a = Memory.open(tmp_path / "home", tmp_path / "proj-a")
    b = Memory.open(tmp_path / "home", tmp_path / "proj-b")
    a.store("project").save("target", "毛利率目标 35%", "x", TODAY)
    a.store("user").save("style", "先给结论", "x", TODAY)
    assert [e.name for e in a.entries()] == ["target", "style"], "项目的在前"
    assert [e.name for e in b.entries()] == ["style"], "另一个项目看不到 a 的项目记忆，用户记忆共用"
    assert a.store("project").root == tmp_path / "home" / "projects" / project_slug(tmp_path / "proj-a") / "memory"


def test_索引写成几天前_没有记忆也说一声_太多就截断(tmp_path):
    assert age(TODAY, TODAY) == "今天" and age(date(2026, 9, 28), TODAY) == "昨天"
    assert age(date(2026, 8, 13), TODAY) == "47 天前"
    memory = _memory(tmp_path)
    assert memory.index(TODAY) == "[长期记忆] 还没有。"
    memory.store("project").save("target", "毛利率目标 35%", "x", date(2026, 9, 26))
    assert "- [项目] target（3 天前）：毛利率目标 35%" in memory.index(TODAY)
    for i in range(INDEX_LIMIT + 2):
        memory.store("user").save(f"m{i}", "x", "x", TODAY)
    assert "还有 3 条没列出" in memory.index(TODAY)


def test_remember和read_memory(tmp_path):
    memory = _memory(tmp_path)
    remember, read = RememberTool(memory), ReadMemoryTool(memory)
    out = remember.run(remember.Args(action="save", scope="project", name="target", description="毛利率目标 35%"))
    assert "新建" in out.content and "35%" in read.run(read.Args(scope="project", name="target")).content
    with pytest.raises(ValueError, match="description"):
        remember.run(remember.Args(action="save", scope="user", name="x"))
    remember.run(remember.Args(action="delete", scope="project", name="target"))
    assert memory.entries() == []
    assert read.rerunnable and not remember.rerunnable, "写入有副作用，不能被当成可以清掉的结果"
    with pytest.raises(ValueError):
        remember.Args(action="save", scope="team", name="x")


def test_组装_开了记忆才有工具和规则_目录附在系统提示词最后(tmp_path):
    base = dict(project_dir=str(tmp_path), database_url="", python_sandbox=False, r_sandbox=False,
                memory_dir=str(tmp_path / "home"))
    off = build_application(Settings(**base, memory_enabled=False), llm=ScriptedProvider())
    assert off.memory is None and "remember" not in off.tools and "长期记忆" not in off.agent._render_system_prompt()

    app = build_application(Settings(**base, memory_enabled=True), llm=ScriptedProvider())
    assert "remember" in app.tools and "read_memory" in app.tools
    assert "## 长期记忆" in app.agent.system_prompt
    assert "[长期记忆] 还没有。" in app.agent._render_system_prompt()     # 会话开始
    app.memory.store("project").save("target", "毛利率目标 35%", "x", date.today())
    assert "[长期记忆] 还没有。" in app.agent._render_system_prompt(), "会话中途不变：变了缓存就废了"
    app.agent.reset()
    assert "target（今天）：毛利率目标 35%" in app.agent._render_system_prompt(), "新会话重新读"


def test_评测_记忆文件按作用域拼起来检查(tmp_path):
    memory = Memory.open(tmp_path, tmp_path / "project")
    memory.store("user").save("unit", "金额用万元", "x", TODAY)
    memory.store("project").save("target", "毛利率目标 35%", "x", TODAY)
    files = memory_files(tmp_path)
    assert set(files) == {"user/unit", "project/target"}

    def check(**kw):
        t = Trial("m/1", 1)
        check_memory(t, Case("m/1", "q", (), **kw), files)
        return t.memory_ok, t.memory_problems

    assert check() == (None, "")
    assert check(memory_has=(r"(?s)=== user/[^=]*万元",), memory_lacks=("华东",)) == (True, "")
    ok, why = check(memory_has=(r"(?s)=== project/[^=]*万元",), memory_lacks=(r"35\s*%",))
    assert not ok and "万元" in why and "35" in why
