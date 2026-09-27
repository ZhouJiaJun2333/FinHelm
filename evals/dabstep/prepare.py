"""下载 DABstep 数据、生成题库。在项目根目录跑：

    python -m evals.dabstep.prepare

数据（CC BY 4.0，Adyen）从 Hugging Face 的 adyen/DABstep 下载到 data/dabstep/（不进 git，约 24 MB）：
    context/   7 个文件：payments.csv（13.8 万笔交易）、fees.json（1000 条手续费规则）、manual.md …
               payments 场景包把它只读挂进沙箱的 /data/
    tasks/     all.jsonl（450 题，答案不公开）、dev.jsonl（10 题，有答案）

生成两个题库：
    evals/cases/dabstep_dev.jsonl   10 题，本地按官方规则判分，平时回归用
    evals/cases/dabstep.jsonl       450 题，答案不公开：跑完用 python -m evals.dabstep.submission 导出提交文件，
                                    去排行榜（https://huggingface.co/spaces/adyen/DABstep）提交才有分

题目原文（英文）照发，后面附官方的答案格式要求，再要求最后一行写「最终答案：」，评测只看那一行。
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dabstep"
CASES = ROOT / "evals" / "cases"
BASE = "https://huggingface.co/datasets/adyen/DABstep/resolve/main/data"
CONTEXT = ["acquirer_countries.csv", "fees.json", "manual.md", "merchant_category_codes.csv",
           "merchant_data.json", "payments-readme.md", "payments.csv"]

# 步数用完时尽量交一个答案：排行榜上弃权和答错一样算错
SETTINGS = {"domain": "payments", "max_steps": 25, "wrap_up": "best_guess"}

ASK = """{question}

答案格式要求（评测按这个判）：{guidelines}

回答的最后单独一行写：最终答案：<答案>
<答案> 严格按上面的格式要求写，不加粗、不加引号。"""


def download() -> None:
    for sub, names in (("context", CONTEXT), ("tasks", ["all.jsonl", "dev.jsonl"])):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            target = DATA / sub / name
            if target.exists():
                continue
            print(f"下载 {sub}/{name} …")
            urllib.request.urlretrieve(f"{BASE}/{sub}/{name}", target)


def case(task: dict, *, hidden: bool) -> dict:
    return {
        "id": f"dab-{task['task_id']}",
        "question": ASK.format(question=task["question"].strip(), guidelines=task["guidelines"].strip()),
        **({"answer_hidden": True} if hidden else {"official_answer": str(task["answer"])}),
        "tags": [f"难度:{task['level']}"],
        "note": f"DABstep task_id={task['task_id']}",
    }


def write(name: str, tasks: list[dict], *, hidden: bool) -> None:
    lines = [json.dumps({"settings": SETTINGS}, ensure_ascii=False)]
    lines += [json.dumps(case(t, hidden=hidden), ensure_ascii=False) for t in tasks]
    (CASES / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}.jsonl：{len(tasks)} 题")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / "tasks" / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    download()
    write("dabstep_dev", load("dev.jsonl"), hidden=False)
    write("dabstep", load("all.jsonl"), hidden=True)


if __name__ == "__main__":
    main()
