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
def write_cases() -> None:
    """BIRD 题目 → evals/cases/bird_financial.jsonl。

    evidence（专家标注的外部知识）拼在问题后面 —— BIRD 的标准设定就是「问题 + evidence」一起给模型。
    match 用 distinct：BIRD 比的是去重后的行集合（set(pred) == set(gold)），不管顺序和重复。
    """
    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["db_id"] == DB_ID]
    lines = [json.dumps({"settings": {"domain": "financial"}}, ensure_ascii=False)]
    for q in sorted(questions, key=lambda q: q["question_id"]):
        question = q["question"].strip()
        if q.get("evidence", "").strip():
            question += f"\n\n提示（外部知识）：{q['evidence'].strip()}"
        lines.append(json.dumps({
            "id": f"bird-fin-{q['question_id']:04d}",
            "question": question,
            "gold_sql": q["SQL"],
            "match": "distinct",
            "tags": [f"难度:{q['difficulty']}"],
            "note": f"BIRD Mini-Dev question_id={q['question_id']}（{QUESTIONS.name}）",
        }, ensure_ascii=False))
    CASES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写了 {len(questions)} 道题 → {CASES.relative_to(ROOT)}")


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
