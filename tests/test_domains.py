"""场景包：同一个内核，换一个场景包就换一个业务库和一套业务约定。不需要数据库。"""

from __future__ import annotations

import pytest

from data_agent.app import build_application
from data_agent.domains import get_domain
from data_agent.prompts import build_system_prompt
from data_agent.settings import Settings

from fakes import ScriptedProvider


def test_业务约定跟着场景包走_通用部分不变():
    shop = build_system_prompt(get_domain("shop"))
    bank = build_system_prompt(get_domain("financial"))
    assert "只算 `completed`" in shop and "completed" not in bank
    assert "DISPONENT" in bank and "DISPONENT" not in shop
    # 通用部分两边都有：工作流程、结果引用的写法（format 之后还是两层花括号）
    for prompt in (shop, bank):
        assert "`describe_table`" in prompt and "`{{r3}}`" in prompt


def test_组装时按配置选场景包():
    app = build_application(Settings(domain="financial"), llm=ScriptedProvider())
    assert app.inspector.schemas == ("financial",)
    assert "捷克银行" in app.agent.system_prompt


def test_不认识的场景包直接报错():
    with pytest.raises(ValueError, match="可选"):
        get_domain("insurance")
