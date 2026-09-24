"""题库：evals/cases/<名字>.jsonl，一行一道题。

    {"id": "shop-003",
     "question": "2024 年的销售额是多少？",
     "gold_sql": "SELECT ...",          或者一个列表：几种做法都算对（比如 NULL 剔除或单列）
     "answer_sql": "SELECT ...",        可选：回答里必须说到的数。不写就用对上的那条 gold_sql。
                                        取数和结论不是一回事：问「线上比线下多多少」，Agent 查出
                                        两个渠道各多少、在回答里自己减，取数就是对的 —— 但回答里
                                        必须有那个差值
     "match": "set",                    怎么比，见 graders.MatchMode；不写就是 set
     "tags": ["口径:completed"],        报告里按它分类统计
     "note": "为什么这么算"}            给人看的：出题人的口径说明

答案存 SQL 不存结果：判分时现跑，数据改了不用改题（BIRD 也是这么做的）。
题目里**不写口径**（只算 completed、扣折扣）：系统提示词里约定过，考的就是 Agent 守不守约定。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .graders import MatchMode

CASES_DIR = Path(__file__).parent / "cases"


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    gold_sql: tuple[str, ...]          # 至少一条；多条 = 几种做法都算对
    match: MatchMode = "set"
    answer_sql: str | None = None      # 回答里必须说到的数；None = 用对上的那条 gold_sql
    tags: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class CaseSet:
    name: str
    cases: list[Case] = field(default_factory=list)
    sha1: str = ""                     # 题库文件的指纹，写进运行记录：题改过，分数就不能直接比


def load_cases(name: str, only: set[str] | None = None) -> CaseSet:
    path = CASES_DIR / f"{name}.jsonl"
    raw = path.read_bytes()
    cases = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {n} 行不是合法 JSON：{exc}") from exc
        gold = d["gold_sql"]
        case = Case(
            id=d["id"],
            question=d["question"],
            gold_sql=tuple(gold) if isinstance(gold, list) else (gold,),
            match=d.get("match", "set"),
            answer_sql=d.get("answer_sql"),
            tags=tuple(d.get("tags", ())),
            note=d.get("note", ""),
        )
        if only is None or case.id in only:
            cases.append(case)

    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} 里有重复的 id")
    return CaseSet(name, cases, hashlib.sha1(raw).hexdigest()[:12])
