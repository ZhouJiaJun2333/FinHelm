"""FinanceBench 数值题判分：只看最后的 Final answer 行，容差 max(1%, 标准答案末位的一半)。"""

from evals.financebench.grade import final_number, gold_value, grade


def test_标准答案的写法():
    assert gold_value("$1577.00") == (1577.0, 0.005)
    assert gold_value("65.4%") == (65.4, 0.05)
    assert gold_value("-0.02") == (-0.02, 0.005)
    assert gold_value("0.8") == (0.8, 0.05)


def test_只看最后一行_中间列过的数不算():
    reply = "2018: 1,577; 2017: 1,373.\n\nFinal answer: 1,373"
    assert not grade("$1577.00", reply).ok, "中间说到了 1,577 也不算"
    assert grade("$1577.00", "Final answer: $1,577 million").ok
    assert grade("$1577.00", "**Final answer:** 1577").ok, "加粗也认"
    assert final_number("Final answer: 1\nFinal answer: 2") == 2.0, "写了两次按最后一次"


def test_容差_按标准答案的精度和1percent():
    assert grade("0.01", "Final answer: 0.0147").ok, "标准答案保留两位：0.005–0.015 都对"
    assert not grade("0.01", "Final answer: 0.016").ok
    assert grade("$1577.00", "Final answer: 1590").ok, "1% 以内"
    assert not grade("$1577.00", "Final answer: 1600").ok


def test_正负号_百分数_单位换算():
    assert grade("-3.7", "Final answer: −3.7").ok and grade("$1577.00", "Final answer: (1,577)").ok
    assert grade("1.9%", "Final answer: 0.019").ok and grade("1.9%", "Final answer: 1.9%").ok
    assert grade("$0.40", "Final answer: 400").ok, "问的是十亿，答成百万"


def test_没写最终答案():
    g = grade("24.26", "The ratio is 24.26.")
    assert not g.ok and g.got is None and g.note == "没写最终答案"
    assert grade("24.26", "Final answer: N/A").note == "最终答案里没有数"
