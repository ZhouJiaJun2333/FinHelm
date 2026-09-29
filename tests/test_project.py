"""只有一个 Agent：项目的约定来自 AGENTS.md，有哪些工具只看环境里配了什么。不需要数据库。"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.db.introspection import SchemaInspector, TableInfo
from data_agent.prompts import build_system_prompt
from data_agent.settings import Settings
from data_agent.tools.sql.describe_table import DescribeTableTool
from evals.run import run_schema

from fakes import ScriptedProvider, eval_settings

SQL = ("list_tables", "describe_table", "run_sql")
# 连不上的库：组装时要是碰了数据库，测试就会卡 10 秒再报错
NO_DB = "postgresql://nobody:x@127.0.0.1:1/none"


def _rules(name: str) -> str:
    return Path(f"evals/projects/{name}/AGENTS.md").read_text(encoding="utf-8")


def test_约定跟着项目走_其余部分一字不差():
    shop, bank = (build_system_prompt(SQL, rules=_rules(n)) for n in ("shop", "financial"))
    assert "只算 `completed`" in shop and "completed" not in bank
    assert "DISPONENT" in bank and "DISPONENT" not in shop
    assert shop.split("## 项目约定")[0] == bank.split("## 项目约定")[0]
    assert shop.split("## 展示结果")[1] == bank.split("## 展示结果")[1]


def test_AGENTS_md原样进提示词_没有就不出约定那一节(tmp_path):
    (tmp_path / "AGENTS.md").write_text("- 金额单位是万元。\n", encoding="utf-8")
    with_md = build_application(Settings(project_dir=str(tmp_path), database_url="", python_sandbox=False),
                                llm=ScriptedProvider())
    assert "## 项目约定（很重要）\n- 金额单位是万元。" in with_md.agent.system_prompt

    (tmp_path / "AGENTS.md").unlink()
    without = build_application(Settings(project_dir=str(tmp_path), database_url="", python_sandbox=False),
                                llm=ScriptedProvider())
    assert "## 项目约定" not in without.agent.system_prompt


def test_项目目录不存在_组装时就报错(tmp_path):
    with pytest.raises(FileNotFoundError, match="项目目录"):
        build_application(Settings(project_dir=str(tmp_path / "没有")), llm=ScriptedProvider())


def test_没配数据库就没有SQL工具():
    app = build_application(Settings(database_url="", python_sandbox=False, r_sandbox=False),
                            llm=ScriptedProvider())
    assert app.db is None and len(app.tools) == 0
    assert "run_sql" not in app.agent.system_prompt


def test_题库指定schema_只看它():
    app = build_application(eval_settings("bird_financial", database_url=NO_DB), llm=ScriptedProvider())
    assert app.inspector.schemas == ("financial",)
    assert "DISPONENT" in app.agent.system_prompt and "捷克银行" in app.agent.system_prompt


def test_没指定schema_组装时不碰数据库_第一次用到才查(monkeypatch):
    calls = []
    monkeypatch.setattr(SchemaInspector, "_user_schemas", lambda self: calls.append(1) or ("financial", "shop"))
    app = build_application(Settings(database_url=NO_DB, db_schema="", python_sandbox=False, r_sandbox=False),
                            llm=ScriptedProvider())
    assert calls == []
    assert app.inspector.schemas == ("financial", "shop") and app.inspector.schemas == ("financial", "shop")
    assert calls == [1], "查一次就记住"


class _Inspector:
    def __init__(self, schemas):
        self.schemas = schemas

    def list_tables(self):
        return [TableInfo("financial", "account", None, 1), TableInfo("shop", "orders", None, 1)]


def test_describe_table_不带前缀_看得到几个schema时找表在哪():
    split = DescribeTableTool(None, _Inspector(("financial", "shop")))._split
    assert split("orders") == ("shop", "orders")
    assert split("account") == ("financial", "account")
    assert split("没有这张表") == ("financial", "没有这张表"), "找不到按第一个报错，报错里会列出已知的表"
    assert split("shop.account") == ("shop", "account"), "写了前缀就照前缀"
    assert DescribeTableTool(None, _Inspector(("shop",)))._split("account") == ("shop", "account")


def test_旧运行只记了场景包名_照样找到schema():
    assert run_schema({"domain": "financial"}, Settings()) == "financial"
    assert run_schema({"db_schema": "shop", "domain": "financial"}, Settings()) == "shop"
    assert run_schema({"domain": "research"}, Settings(db_schema="")) == ""
