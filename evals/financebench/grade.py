"""FinanceBench 数值题判分：只看回答最后的「Final answer: <数>」那一行，按容差比。

为什么只看最后一行：回答里常把几年的数、中间结果都列出来，「回答里说到了标准答案」太容易碰上。
四种做法（Agentic RAG、传统 RAG、给证据页、整份文档）都要求写这一行，判法一样。

容差 = max(相对 1%，标准答案末位的一半)：标准答案是按题目要求舍入过的（「0.01」是 ROA 保留两位小数），
回答写 0.0147 也算对；「$1577.00」这种末位很细的，按 1% 算。
不看正负号（「(1,577)」「capex of -1,577」说的都是 1,577）；单位差 100 / 1000 倍的也认
（1.9% 写成 0.019；「$0.40 billion」写成 400 million）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

REL_TOL = 0.01
SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001)
FINAL = re.compile(r"(?:final answer|最终答案)\s*[:：]\s*(.+)", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")


@dataclass(frozen=True, slots=True)
class Grade:
    ok: bool
    got: float | None            # 从最后一行解析出来的数；None = 没写最终答案、或者那行没有数
    gold: float
    note: str = ""


def gold_value(answer: str) -> tuple[float, float]:
    """「$1577.00」「65.4%」「-0.02」→ (数, 末位的一半)。"""
    text = answer.replace("−", "-").replace("$", "").replace("%", "").replace(",", "").strip()
    d = Decimal(text)
    exp = d.as_tuple().exponent
    return float(d), 0.5 * 10.0 ** exp if isinstance(exp, int) else 0.0


def final_number(reply: str) -> float | None:
    """最后一个「Final answer: …」后面的第一个数（不带正负号）。"""
    lines = FINAL.findall(reply.replace("**", ""))
    if not lines:
        return None
    m = _NUMBER.search(lines[-1])
    return float(m.group().replace(",", "")) if m else None


def grade(gold_answer: str, reply: str) -> Grade:
    gold, half_unit = gold_value(gold_answer)
    got = final_number(reply)
    if got is None:
        return Grade(False, None, gold, "没写最终答案" if not FINAL.search(reply.replace("**", "")) else "最终答案里没有数")
    tol = max(REL_TOL * abs(gold), half_unit)
    for scale in SCALES:
        if abs(got * scale - abs(gold)) <= tol:
            return Grade(True, got, gold, "" if scale == 1 else f"单位差 {scale:g} 倍")
    return Grade(False, got, gold)
