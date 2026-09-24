"""判分：拿 Agent 的结果和标准答案比。全是纯函数，tests/test_eval_graders.py 里有单元测试。

判分器自己有 bug，所有分数就都没意义了 —— 所以这里宁可啰嗦，每条规则都写清楚为什么。

两个判分项：
    compare_results  执行准确率：一条 SQL 的结果 vs 标准 SQL 的结果。
                     运行器拿 Agent 跑过的**每一条**成功的 SQL 来比，有一条对上就算对。
                     BIRD 只看最后一条 —— 可 Agent 常常先查出答案，再多查一条明细给你对比
                     口径，只看最后一条会把答对的判成错（冒烟测试里真的发生了）。
    check_answer     最终回答：标准结果里的关键数字，在回答里能不能找到
                     SQL 对了不等于回答对了 —— 模型可能抄错数、四舍五入过头、单位换错

比 BIRD 宽的地方（都有测试钉住）：多给几列、比例乘了 100、NULL 显示成「未填写」这种标签。
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Sequence

# 结果怎么比：
#   set       行的集合相同，不管顺序（默认）
#   ordered   行的顺序也要一样（题目要求「从高到低」这种）
#   top       标准答案是排名前几名；Agent 的结果前几行和它一样就行，
#             后面多列几名不算错（问「哪个最高」，Agent 常常把整个排名都查出来）
#   contains  标准答案的行都能在 Agent 的结果里找到，多出的行不算错。
#             标准答案只有一个数的题用它：Agent 顺手按状态列了三行对比，只要其中有这个数就行
#   empty     应该查不到东西（问 2030 年的销售额）。运行器按回答判：说了「没有」就算对 ——
#             Agent 常常是先查一下数据覆盖哪几年来证明没有，这比返回一个空结果更好
#   answer    只看回答（多轮会话里的回忆题：「第一个问题里华东是多少？」）。回答里说到
#             answer_sql 的数就算对，靠记忆答、重新查一遍都行
MatchMode = Literal["set", "ordered", "top", "contains", "empty", "answer"]

# 数值比较的容差：**按数字写出来的精度算**，不用一个固定的绝对误差。
#   ROUND(x, 2) 得到 0.33，舍入误差最多 0.005，那就容 0.005；写成 0.0561 就只容 0.00005；
#   回答里写「810 万」容 0.5 万，写「5.61%」容 0.005%。
# 以前是固定 0.005：对 0.3 量级的比例刚好，对 0.05 量级的退货率等于容了 10% 的误差 ——
# 按 2024 年算的 5.61% 被当成全部年份的 5.57% 判对（2026-09-25 多轮评测里发现）。
# REL_TOL 只兜浮点计算的末位噪声（同一个数，SUM/COUNT 和 AVG 算出来末几位可能不同）。
REL_TOL = 1e-6
# 比例的两种写法：0.3318 和 33.18（乘了 100）都算对
SCALES = (1.0, 100.0)
# 标准结果超过这么多行就不核对回答了 —— 没人会在回答里把 50 行全念一遍
ANSWER_CHECK_MAX_ROWS = 20


# ============================================================== 值的归一化
@dataclass(frozen=True, slots=True)
class Num:
    """一个数，连同它写出来的精度带来的舍入误差（半个末位）。"""

    value: float
    tol: float = 0.0

    @classmethod
    def of(cls, v: int | float | Decimal | str) -> "Num":
        """int 是精确的（COUNT）；Decimal 看小数位数（ROUND(x, 2) → Decimal('0.33')）；
        float 看它最短的写法 —— 0.33 显示成 0.33，全精度算出来的数显示十几位。"""
        if isinstance(v, int):
            return cls(float(v))
        d = Decimal(repr(v)) if isinstance(v, float) else Decimal(v)
        exp = d.as_tuple().exponent
        return cls(float(d), 0.5 * 10.0 ** exp if isinstance(exp, int) else 0.0)

    def scaled(self, factor: float) -> "Num":
        return Num(self.value * factor, self.tol * abs(factor))


def normalize(v: Any) -> Any:
    """数据库返回的值 → 能比较的值。数字变成 Num（带精度），日期统一成字符串。"""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        return Num.of(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return str(v).strip()


def _close(a: Any, b: Any) -> bool:
    """两个数的差不超过双方的舍入误差之和，就算同一个数。"""
    if isinstance(a, Num) and isinstance(b, Num):
        return abs(a.value - b.value) <= a.tol + b.tol + REL_TOL * abs(b.value)
    return a == b


def _cell_match(gold: Any, pred: Any) -> bool:
    # 标准答案是 NULL、Agent 给了个文字标签（COALESCE(region, '未填写')）：算对上。
    # 把 NULL 单列成「未填写」一行是很正常的展示方式。只对文字放宽，数字的 NULL 不能随便对。
    if gold is None and isinstance(pred, str):
        return True
    return _close(pred, gold)


def _rows_equal(gold: Sequence[Any], pred: Sequence[Any]) -> bool:
    """注意参数顺序：先标准答案，后 Agent 的 —— NULL 的放宽只朝一个方向。"""
    return len(gold) == len(pred) and all(_cell_match(g, p) for g, p in zip(gold, pred))


def _pair_up(gold: list[tuple], pred: list[tuple]) -> bool:
    """标准答案的每一行，都能在 Agent 的结果里找到一行对上（一行只能用一次）。

    带 NULL 的行最后配：NULL 能对上任何文字标签，先配它会抢走别的行的位置。
    """
    remaining = list(pred)
    for g in sorted(gold, key=lambda r: sum(v is None for v in r)):
        hit = next((i for i, p in enumerate(remaining) if _rows_equal(g, p)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


# ============================================================== 结果比对
@dataclass(frozen=True, slots=True)
class ResultMatch:
    strict: bool      # 列数也一样
    lenient: bool     # 允许多出几列（Agent 顺手多给了订单数之类），比例允许乘了 100
    reason: str = ""  # 没对上时说一句为什么，给失败分析用


def compare_results(
    gold: Sequence[Sequence[Any]],
    pred: Sequence[Sequence[Any]],
    mode: MatchMode = "set",
) -> ResultMatch:
    """比较两份结果（行的列表）。不看列名，只看值。

    列的对应关系是自动找的：Agent 的列顺序、列名和标准答案不一样很正常，
    所以给标准答案的每一列在 Agent 的结果里找一列「值对得上」的，找到一种
    对应方式能让整行都对上，就算匹配。
    """
    gold = [tuple(normalize(v) for v in r) for r in gold]
    pred = [tuple(normalize(v) for v in r) for r in pred]

    if mode == "empty":
        ok = _looks_empty(pred)
        return ResultMatch(ok, ok, "" if ok else f"应该查不到数据，但结果有 {len(pred)} 行")

    if not gold:
        ok = not pred
        return ResultMatch(ok, ok, "" if ok else "标准答案是空结果，Agent 查出了数据")
    if not pred:
        return ResultMatch(False, False, "Agent 的结果是空的")

    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
        return ResultMatch(False, False, f"列不够：标准答案 {n_gold} 列，Agent 只有 {n_pred} 列")
    if not _row_count_ok(len(gold), len(pred), mode):
        return ResultMatch(False, False, f"行数不对：标准答案 {len(gold)} 行，Agent {len(pred)} 行")

    # 每一列的候选：Agent 的哪些列（乘哪个倍数）单独看是对得上的
    candidates = []
    for j in range(n_gold):
        gold_col = [(r[j],) for r in gold]
        options = [
            (k, scale)
            for k in range(n_pred)
            for scale in SCALES
            if _match_rows(gold_col, [(_scale(r[k], scale),) for r in pred], mode)
        ]
        if not options:
            return ResultMatch(False, False, f"标准答案第 {j + 1} 列在 Agent 的结果里找不到对应的列")
        candidates.append(options)

    # 再看整行：找一种列的对应方式（不同的标准列对到不同的 Agent 列）
    for combo in itertools.product(*candidates):
        cols = [k for k, _ in combo]
        if len(set(cols)) < len(cols):
            continue
        projected = [tuple(_scale(r[k], s) for k, s in combo) for r in pred]
        if _match_rows(gold, projected, mode):
            return ResultMatch(strict=(n_pred == n_gold), lenient=True)
    return ResultMatch(False, False, "每一列单独都对得上，但整行对不上（行和行串了）")


def _scale(v: Any, scale: float) -> Any:
    return v.scaled(1 / scale) if isinstance(v, Num) and scale != 1.0 else v


def _row_count_ok(n_gold: int, n_pred: int, mode: MatchMode) -> bool:
    if mode in ("set", "ordered"):
        return n_gold == n_pred
    return n_pred >= n_gold          # top / contains：Agent 可以多给


def _match_rows(gold: list[tuple], pred: list[tuple], mode: MatchMode) -> bool:
    if mode == "ordered":
        return len(gold) == len(pred) and all(_rows_equal(g, p) for g, p in zip(gold, pred))
    if mode == "top":
        return len(pred) >= len(gold) and all(_rows_equal(g, p) for g, p in zip(gold, pred))
    if mode == "contains":
        return _pair_up(gold, pred)
    # set：行数一样，而且一一对得上
    return len(gold) == len(pred) and _pair_up(gold, pred)


def _looks_empty(rows: list[tuple]) -> bool:
    """空结果、或者只有一行而且全是 NULL / 0（SUM 在没有数据时返回 NULL，COUNT 返回 0）。"""
    if not rows:
        return True
    return len(rows) == 1 and all(v is None or (isinstance(v, Num) and v.value == 0) for v in rows[0])


# ============================================================== 回答核对
@dataclass(frozen=True, slots=True)
class AnswerCheck:
    ok: bool | None                      # None = 不适用（结果太多行，不核对）
    missing: list[float] = field(default_factory=list)
    checked: int = 0


_NUMBER = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(%|万|亿)?")


_UNITS = {"%": 0.01, "万": 1e4, "亿": 1e8}


def answer_numbers(text: str) -> list[Num]:
    """回答里出现的所有数字，带着写出来的精度。

    「810.05 万」同时记成 810.05（容 0.005）和 8100500（容 50）；
    「33.18%」记成 33.18 和 0.3318（容 0.00005）。
    """
    values: list[Num] = []
    for m in _NUMBER.finditer(text):
        n = Num.of(m.group(1).replace(",", ""))
        values.append(n)
        if m.group(2):
            values.append(n.scaled(_UNITS[m.group(2)]))
    return values


def check_answer(gold: Sequence[Sequence[Any]], answer: str) -> AnswerCheck:
    """标准结果里的每个数字，回答里都要说到（允许常见的写法差异）。

    只核对数字：文字（区域名、商品名）的写法太多样，交给结果比对去管。
    """
    if len(gold) > ANSWER_CHECK_MAX_ROWS:
        return AnswerCheck(None)
    targets = [v for r in gold for v in (normalize(x) for x in r) if isinstance(v, Num)]
    if not targets:
        return AnswerCheck(None)
    said = answer_numbers(answer)
    missing = [t.value for t in targets if not any(_said(t, s) for s in said)]
    return AnswerCheck(not missing, missing, len(targets))


def _said(target: Num, said: Num) -> bool:
    return any(_close(said, target.scaled(scale)) for scale in SCALES)
