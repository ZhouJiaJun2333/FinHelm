"""run_sql 那道关键字防线的测试。

注意：这些测试保证的是「常见写操作会被挡下」，
**不保证这道防线不可绕过**。真正的底线是数据库里的 agent_ro 只读账号。
安全永远不要只靠一层。
"""

from __future__ import annotations

import pytest

from data_agent.tools.sql.run_sql import _validate


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select * from orders limit 10",
    "  SELECT count(*) FROM shop.orders;  ",
    "WITH t AS (SELECT 1 AS a) SELECT * FROM t",
    "-- 统计订单\nSELECT count(*) FROM orders",
])
def test_合法的只读查询能通过(sql):
    assert _validate(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "drop table orders",
    "UPDATE orders SET status = 'x'",
    "INSERT INTO orders VALUES (1)",
    "TRUNCATE orders",
    "CREATE TABLE hack (i int)",
    "GRANT ALL ON orders TO public",
])
def test_写操作被拒绝(sql):
    with pytest.raises(ValueError):
        _validate(sql)


def test_多条语句被拒绝():
    with pytest.raises(ValueError, match="单条语句"):
        _validate("SELECT 1; DROP TABLE orders")


def test_注释里藏写操作也会被发现():
    """先去注释再判断，避免 `/* select */ delete` 这种绕过。"""
    with pytest.raises(ValueError):
        _validate("/* select */ DELETE FROM orders")


def test_select里夹带写操作会被发现():
    with pytest.raises(ValueError, match="(?i)delete"):
        _validate("SELECT * FROM orders WHERE 1=1 -- x\nDELETE FROM orders")


def test_空语句():
    with pytest.raises(ValueError, match="为空"):
        _validate("   ")
