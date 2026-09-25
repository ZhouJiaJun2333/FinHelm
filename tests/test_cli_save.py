"""终端 /save 的参数解析：编号可以省，省了就存最近一个结果。"""

from __future__ import annotations

from data_agent.cli import parse_save

TABLES = {"r1": object(), "r2": object(), "r3": object()}


def test_不写编号就存最近一个():
    assert parse_save("", TABLES) == ("r3", "")


def test_写了编号就存那个_后面是文件名():
    assert parse_save("r1", TABLES) == ("r1", "")
    assert parse_save("r1 华东订单", TABLES) == ("r1", "华东订单")


def test_第一个词不像编号就当文件名():
    assert parse_save("华东订单", TABLES) == ("r3", "华东订单")


def test_编号不存在或还没有结果():
    assert parse_save("r9", TABLES) == (None, "")
    assert parse_save("", {}) == (None, "")
