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

多轮会话：一行一段会话，同一个 Agent 按顺序回答每一轮，考的是上下文管理 ——
清理、压缩之后还答不答得对。

    {"id": "multi-001",
     "settings": {"context_clear_trigger_tokens": 15000},   覆盖这段会话的配置（调低门槛）
     "turns": [
        {"question": "...", "gold_sql": "..."},             和单题一样判
        {"question": "...", "filler": true},                填充：不判分，只为塞进大结果、撑大上下文
        {"question": "...", "match": "answer",              回忆：只看回答里的数对不对，
         "answer_sql": "..."}                                 靠记忆答、重查一遍都算对
     ]}

每一轮的 id 是「会话 id/轮次」，比如 multi-001/3。答错了会话照样往下问 —— 真实用户也会接着问。

上传文件的题（research 场景，不连数据库）：先把文件传上去再提问，按回答和做了什么判分。

    {"id": "rs-01",
     "files": ["research/files/xx.xlsx"],        相对 evals/cases/，每个 trial 复制进自己的工作目录
     "question": "帮我做个 meta 分析",
     "gold_values": [0.4896, 0.3448, 0.6952],     回答里必须说到的数（不看正负号：「少住 14 天」也算说到了 -14）
     "expect_code": ["fh_meta_gen"],       沙箱代码里必须出现（正则）：该用的模板用了没有
     "expect_text": ["Heard"],                     回答里必须出现（正则）：该指出的问题指出了没有
     "expect_figure": true}                        至少画出一张图

题库级配置：没有 id 的一行，整个题库都用它。
    settings   覆盖配置（优先级：.env < 这里 < 命令行 --set）。BIRD 的题库靠它选场景包
    submit     提交轮：每题答完之后追问这句话，收一条 SQL 按 BIRD 官方规则判（见 runner.py）

    {"settings": {"domain": "financial"}, "submit": "请交一条……"}
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
    gold_sql: tuple[str, ...]          # 多条 = 几种做法都算对；只有填充和回忆轮可以没有
    match: MatchMode = "set"
    answer_sql: str | None = None      # 回答里必须说到的数；None = 用对上的那条 gold_sql
    tags: tuple[str, ...] = ()
    note: str = ""
    filler: bool = False               # 多轮会话里的填充轮：不判分
    # 上传文件的题（见开头）
    files: tuple[str, ...] = ()
    gold_values: tuple[float, ...] = ()
    expect_code: tuple[str, ...] = ()
    expect_text: tuple[str, ...] = ()
    expect_figure: bool = False

    @property
    def graded(self) -> bool:
        return not self.filler

    @property
    def uses_files(self) -> bool:
        """上传文件的题：没有 SQL 可比，按回答、代码、图判。"""
        return bool(self.files)


@dataclass(frozen=True, slots=True)
class Session:
    """一段多轮会话。"""

    id: str
    turns: list[Case]
    settings: dict[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class CaseSet:
    name: str
    cases: list[Case] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)   # 多轮题库；和 cases 二选一
    settings: dict[str, object] = field(default_factory=dict)  # 题库级配置（比如 domain）
    submit: str = ""                   # 提交轮追问的话；空 = 不要提交轮
    sha1: str = ""                     # 题库文件的指纹，写进运行记录：题改过，分数就不能直接比

    @property
    def graded_cases(self) -> list[Case]:
        """所有要判分的题：单题库就是 cases，多轮题库是各段会话里不是填充的那些轮。"""
        if self.sessions:
            return [t for s in self.sessions for t in s.turns if t.graded]
        return self.cases


def load_cases(name: str, only: set[str] | None = None) -> CaseSet:
    path = CASES_DIR / f"{name}.jsonl"
    raw = path.read_bytes()
    cases: list[Case] = []
    sessions: list[Session] = []
    settings: dict[str, object] = {}
    submit = ""
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            d = json.loads(line)
            if "id" not in d:
                if unknown := set(d) - {"settings", "submit"}:
                    raise ValueError(f"题库级配置里有不认识的键：{sorted(unknown)}")
                settings.update(d.get("settings", {}))
                submit = d.get("submit", submit)
            elif "turns" in d:
                s = Session(
                    id=d["id"],
                    turns=[_case(t, f"{d['id']}/{i}") for i, t in enumerate(d["turns"], 1)],
                    settings=d.get("settings", {}),
                    tags=tuple(d.get("tags", ())),
                    note=d.get("note", ""),
                )
                if only is None or s.id in only:
                    sessions.append(s)
            else:
                case = _case(d, d["id"])
                if only is None or case.id in only:
                    cases.append(case)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(f"{path.name} 第 {n} 行：{exc}") from exc

    if cases and sessions:
        raise ValueError(f"{path.name} 里单题和多轮会话混在一起了，分成两个文件")
    ids = [c.id for c in cases] + [s.id for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} 里有重复的 id")
    if submit and sessions:
        raise ValueError(f"{path.name}：提交轮只给单题库用，多轮会话里每轮都追问会打乱会话")
    return CaseSet(name, cases, sessions, settings, submit, hashlib.sha1(raw).hexdigest()[:12])


def _case(d: dict, case_id: str) -> Case:
    gold = d.get("gold_sql", [])
    case = Case(
        id=case_id,
        question=d["question"],
        gold_sql=tuple(gold) if isinstance(gold, list) else (gold,),
        match=d.get("match", "set"),
        answer_sql=d.get("answer_sql"),
        tags=tuple(d.get("tags", ())),
        note=d.get("note", ""),
        filler=d.get("filler", False),
        files=tuple(d.get("files", ())),
        gold_values=tuple(float(v) for v in d.get("gold_values", ())),
        expect_code=tuple(d.get("expect_code", ())),
        expect_text=tuple(d.get("expect_text", ())),
        expect_figure=d.get("expect_figure", False),
    )
    if case.uses_files:
        if missing := [f for f in case.files if not (CASES_DIR / f).is_file()]:
            raise ValueError(f"{case_id}：找不到文件 {missing}")
        if not (case.gold_values or case.expect_code or case.expect_text or case.expect_figure):
            raise ValueError(f"{case_id}：上传文件的题至少要有一项判分依据（gold_values / expect_*）")
        return case
    if case.match == "answer" and not case.answer_sql:
        raise ValueError(f"{case_id}：match=answer 的题要写 answer_sql")
    if case.graded and case.match != "answer" and not case.gold_sql:
        raise ValueError(f"{case_id}：要判分的题至少一条 gold_sql")
    return case
