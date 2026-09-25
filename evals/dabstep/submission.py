"""把一次 DABstep 运行导出成排行榜要的提交文件：

    python -m evals.dabstep.submission evals/runs/<一次 dabstep 的运行>

写到运行目录下的 submission.jsonl，一行一题：{"task_id", "agent_answer", "reasoning_trace"}。
每题跑了几次就取第 1 次（排行榜一题只收一个答案）；没写出「最终答案」的交空串（算错，但不能缺题）。
提交要去 https://huggingface.co/spaces/adyen/DABstep 的表单手动交，这里只生成文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 推理过程只放回答正文，截一段：排行榜只是存档，不参与判分
TRACE_CHARS = 4000


def export(run_dir: Path) -> Path:
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    first: dict[str, dict] = {}
    for t in sorted(trials, key=lambda t: t["trial"]):
        first.setdefault(t["case_id"], t)
    out = run_dir / "submission.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for case_id, t in sorted(first.items(), key=lambda kv: int(kv[0].removeprefix("dab-"))):
            f.write(json.dumps({"task_id": case_id.removeprefix("dab-"), "agent_answer": t.get("final_answer", ""),
                                "reasoning_trace": t["answer"][:TRACE_CHARS]}, ensure_ascii=False) + "\n")
    missing = sum(not t.get("final_answer") for t in first.values())
    print(f"{len(first)} 题 → {out}" + (f"（{missing} 题没写出最终答案，交的是空串）" if missing else ""))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        sys.exit("用法：python -m evals.dabstep.submission <运行目录>")
    export(Path(sys.argv[1]))
