"""把 BIRD Mini-Dev 的 financial 库和题目准备好。

    python -m evals.bird.prepare            导库 + 生成题库（两步都做）
    python -m evals.bird.prepare --cases    只重新生成题库

要先下载（见 README「BIRD」一节），放在 data/bird/ 下：
    minidev/MINIDEV_postgresql/BIRD_dev.sql                   11 个库的 PostgreSQL 导出（约 1 GB）
    minidev/MINIDEV/dev_databases/financial/database_description/*.csv   每列的说明
    mini_dev_pg.json                                          题目（Hugging Face 上 2025-07 修订版）

导库做的事：
    1. 从 BIRD_dev.sql 里只挑 financial 的 8 张表（建表、数据、主外键），public → financial schema
    2. BIRD 给的列说明写成 COMMENT ON COLUMN —— describe_table 会读出来给模型看。
       这是 BIRD 随库发布的元数据，不是我们额外加的知识；不加的话 A2~A16 这种列名模型只能猜
    3. 只读账号 agent_ro 能读这个 schema，ANALYZE 一下（list_tables 的行数估算要用统计信息）
重复跑没关系：先 DROP SCHEMA financial 再建。

用超级用户连（docker-compose 里的 postgres/postgres），可以用 BIRD_ADMIN_URL 覆盖。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import psycopg
from psycopg import sql as pgsql

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "bird"
DUMP = DATA / "minidev" / "MINIDEV_postgresql" / "BIRD_dev.sql"
DESCRIPTIONS = DATA / "minidev" / "MINIDEV" / "dev_databases" / "financial" / "database_description"
QUESTIONS = DATA / "mini_dev_pg.json"
CASES = ROOT / "evals" / "cases" / "bird_financial.jsonl"

ADMIN_URL = os.environ.get("BIRD_ADMIN_URL", "postgresql://postgres:postgres@localhost:5433/analytics")
DB_ID = "financial"
SCHEMA = "financial"
TABLES = ("account", "card", "client", "disp", "district", "loan", "order", "trans")

# 导出里表名的写法：public.account、public."order"（order 是关键字，带引号）
_NAME = "|".join(f'"{t}"' if t == "order" else t for t in TABLES)
CREATE = re.compile(rf"^CREATE TABLE public\.({_NAME}) \(")
COPY = re.compile(rf"^COPY public\.({_NAME}) ")
ALTER = re.compile(rf"^ALTER TABLE ONLY public\.({_NAME})$")


# ================================================================ 导库
def load_database() -> None:
    for path in (DUMP, DESCRIPTIONS):
        if not path.exists():
            sys.exit(f"找不到 {path}，先按 README「BIRD」一节下载解压")
    with psycopg.connect(ADMIN_URL) as conn:
        conn.execute(pgsql.SQL("DROP SCHEMA IF EXISTS {s} CASCADE").format(s=pgsql.Identifier(SCHEMA)))
        conn.execute(pgsql.SQL("CREATE SCHEMA {s}").format(s=pgsql.Identifier(SCHEMA)))
        constraints = _load_tables(conn)
        for stmt in constraints:            # 数据灌完再加约束：先加外键的话每行都要检查一遍，慢
            conn.execute(stmt)
        n = _add_comments(conn)
        conn.execute(pgsql.SQL("GRANT USAGE ON SCHEMA {s} TO agent_ro").format(s=pgsql.Identifier(SCHEMA)))
        conn.execute(pgsql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {s} TO agent_ro")
                     .format(s=pgsql.Identifier(SCHEMA)))
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        for t in TABLES:
            conn.execute(pgsql.SQL("ANALYZE {s}.{t}").format(s=pgsql.Identifier(SCHEMA), t=pgsql.Identifier(t)))
        counts = {t: conn.execute(pgsql.SQL("SELECT count(*) FROM {s}.{t}").format(
            s=pgsql.Identifier(SCHEMA), t=pgsql.Identifier(t))).fetchone()[0] for t in TABLES}
    print(f"已导入 {SCHEMA} schema：" + "，".join(f"{t} {c}" for t, c in counts.items()))
    print(f"写了 {n} 条列说明，agent_ro 已授权")


def _load_tables(conn: psycopg.Connection) -> list[str]:
    """顺着导出文件读一遍（约 1 GB，逐行流式），建表、灌数据；返回要最后执行的约束语句。"""
    constraints: list[str] = []
    with open(DUMP, encoding="utf-8") as f:
        for line in f:
            if CREATE.match(line):
                ddl = [line]
                for more in f:
                    ddl.append(more)
                    if more.startswith(");"):
                        break
                conn.execute(_to_schema("".join(ddl)))
            elif m := COPY.match(line):
                print(f"  灌数据 {m.group(1)} …", flush=True)
                with conn.cursor().copy(_to_schema(line.rstrip().removesuffix(";"))) as copy:
                    for row in f:
                        if row.startswith("\\."):
                            break
                        copy.write(row)
            elif ALTER.match(line):
                constraints.append(_to_schema(line.rstrip() + " " + next(f).strip()))
    return constraints


def _to_schema(text: str) -> str:
    return text.replace("public.", f"{SCHEMA}.")


def _add_comments(conn: psycopg.Connection) -> int:
    n = 0
    for t in TABLES:
        with open(DESCRIPTIONS / f"{t}.csv", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                column = (row.get("original_column_name") or "").strip().lower()
                if not column:
                    continue
                conn.execute(pgsql.SQL("COMMENT ON COLUMN {s}.{t}.{c} IS {text}").format(
                    s=pgsql.Identifier(SCHEMA), t=pgsql.Identifier(t), c=pgsql.Identifier(column),
                    text=pgsql.Literal(_describe(row))))
                n += 1
    return n


def _describe(row: dict[str, str]) -> str:
    """一列的说明：名字、含义、取值说明（照 BIRD 原文，去重去空）。"""
    parts: list[str] = []
    for key in ("column_name", "column_description"):
        text = " ".join((row.get(key) or "").split())
        if text and text.lower() not in (p.lower() for p in parts):
            parts.append(text)
    values = " ".join((row.get("value_description") or "").split())
    if values:
        parts.append(f"取值：{values}")
    return "；".join(parts)


# ================================================================ 题库
# 提交轮（见 evals/runner.py）：BIRD 官方一题只收一条 SQL、结果要和标准答案完全一样。
# 这是评测的输出格式，不是银行业务知识，所以写在这里，不写进 financial 场景包
SUBMIT = """评测要求（BIRD 官方判分）：请把刚才这道题的最终答案写成**一条** SQL，用 run_sql 执行一次作为提交。
- 只返回题目问到的列，按题目里提到的顺序；不要加名称、ID、中间量这类辅助列
- 数值保持原始精度，不要 ROUND 或格式化
- 题目要「最高 / 最多的那个」，就只返回那一行
- 就算前面已经有一条符合要求的 SQL，也再执行一次
执行完直接结束，不用解释。"""

# 标准答案确实错了的题：补上我们认为对的写法，排在 BIRD 原版后面 —— 主分数哪种都认，
# 官方判分（evals/bird/official.py）和报告里的 BIRD 判法只认原版。每条都在库里跑过、核对过。
# 只收「标准答案错了」的；题意有歧义的（账户所有人还是全部客户）不收，那是题目本身的难度。
DISPUTED: dict[int, tuple[str, list[str]]] = {
    115: ("A4（居民数）是文本列，原版按文本排序，选中的是 Jindrichuv Hradec（93931 人），"
          "人口最多的是 Ceske Budejovice（177686 人）", [
        "SELECT CAST(SUM(CASE WHEN T1.gender = 'M' THEN 1 ELSE 0 END) AS REAL) * 100 / NULLIF(COUNT(T1.client_id), 0) "
        "FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id "
        "WHERE T2.A3 = 'south Bohemia' GROUP BY T2.A4 ORDER BY CAST(T2.A4 AS INTEGER) DESC LIMIT 1",
    ]),
    129: ("原版按区名字母序取前十，和取现多少无关。另认两种读法：按区的取现总额排；取金额最大的十笔、列出所在区", [
        "SELECT T1.A2 FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id "
        "INNER JOIN trans AS T3 ON T2.account_id = T3.account_id "
        "WHERE T3.type = 'VYDAJ' AND CAST(T3.date AS TEXT) LIKE '1996-01%' GROUP BY T1.A2 ORDER BY SUM(T3.amount) DESC LIMIT 10",
        "SELECT T1.A2 FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id "
        "INNER JOIN trans AS T3 ON T2.account_id = T3.account_id "
        "WHERE T3.type = 'VYDAJ' AND CAST(T3.date AS TEXT) LIKE '1996-01%' ORDER BY T3.amount DESC LIMIT 10",
    ]),
    152: ("原版把区连上账户表再求平均，一个区有几个账户就被算几次（布拉格一个区就占了一大截）。每个区只算一次", [
        "SELECT AVG(A15) FROM district WHERE A15 > 4000 AND district_id IN "
        "(SELECT district_id FROM account WHERE TO_CHAR(CAST(date AS TIMESTAMP), 'YYYY') >= '1997')",
    ]),
    186: ("原版多连了一次 district，要求客户所在区 = 账户所在区，题目和提示里都没有这个条件", [
        "SELECT CAST(SUM(CASE WHEN T1.gender = 'M' THEN 1 ELSE 0 END) AS REAL) * 100 / NULLIF(COUNT(T1.client_id), 0) "
        "FROM client AS T1 INNER JOIN disp AS T4 ON T1.client_id = T4.client_id "
        "INNER JOIN account AS T2 ON T2.account_id = T4.account_id WHERE T2.frequency = 'POPLATEK TYDNE'",
    ]),
    194: ("年龄用年份相减，生日还没过的人多算一岁（而且每年结果都变）。另认周岁", [
        "SELECT T1.client_id, EXTRACT(YEAR FROM AGE(CURRENT_DATE, T3.birth_date)) AS age "
        "FROM disp AS T1 INNER JOIN card AS T2 ON T2.disp_id = T1.disp_id "
        "INNER JOIN client AS T3 ON T1.client_id = T3.client_id WHERE T2.type = 'gold' AND T1.type = 'OWNER'",
    ]),
}


def write_cases() -> None:
    """BIRD 题目 → evals/cases/bird_financial.jsonl。

    evidence（专家标注的外部知识）拼在问题后面 —— BIRD 的标准设定就是「问题 + evidence」一起给模型。
    match 用 distinct：BIRD 比的是去重后的行集合（set(pred) == set(gold)），不管顺序和重复。
    标准答案有错的题（DISPUTED）：gold_sql 写成列表，原版在第一条，打「标注存疑」标签，理由写进 note。
    """
    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["db_id"] == DB_ID]
    lines = [json.dumps({"settings": {"domain": "financial"}, "submit": SUBMIT}, ensure_ascii=False)]
    for q in sorted(questions, key=lambda q: q["question_id"]):
        question = q["question"].strip()
        if q.get("evidence", "").strip():
            question += f"\n\n提示（外部知识）：{q['evidence'].strip()}"
        case = {
            "id": f"bird-fin-{q['question_id']:04d}",
            "question": question,
            "gold_sql": q["SQL"],
            "match": "distinct",
            "tags": [f"难度:{q['difficulty']}"],
            "note": f"BIRD Mini-Dev question_id={q['question_id']}（{QUESTIONS.name}）",
        }
        if q["question_id"] in DISPUTED:
            reason, alternatives = DISPUTED[q["question_id"]]
            case["gold_sql"] = [q["SQL"], *alternatives]
            case["tags"].append("标注存疑")
            case["note"] += f"。标注存疑：{reason}"
        lines.append(json.dumps(case, ensure_ascii=False))
    if missing := set(DISPUTED) - {q["question_id"] for q in questions}:
        sys.exit(f"DISPUTED 里的题不在题目里：{sorted(missing)}")
    CASES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写了 {len(questions)} 道题（{len(DISPUTED)} 道标注存疑）→ {CASES.relative_to(ROOT)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="准备 BIRD financial 库和题库")
    ap.add_argument("--cases", action="store_true", help="只重新生成题库，不导库")
    args = ap.parse_args()
    if not args.cases:
        load_database()
    write_cases()


if __name__ == "__main__":
    main()
