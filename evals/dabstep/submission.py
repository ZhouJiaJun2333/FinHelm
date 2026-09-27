"""把 DABstep 运行导出成排行榜要的提交文件：

    python -m evals.dabstep.submission evals/runs/<一次 dabstep 的运行> [补跑的运行 …]

写到第一个运行目录下的 submission.jsonl，一行一题：{"task_id", "agent_answer", "reasoning_trace"}。
每题取第一个写出了「最终答案」的 trial：先看第一个运行目录，没写出来（步数耗尽、API 报错）才用后面补跑的。
补跑只看「有没有写出答案」，不看答案对不对（答案不公开，也不该看）。都没写出的交空串（算错，但不能缺题）。
最终答案按现在的规则从回答原文重新抽（抽取规则改过，旧运行记录里的 final_answer 可能是按旧规则抽的）。
提交要去 https://huggingface.co/spaces/adyen/DABstep 的表单手动交，这里只生成文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..runner import extract_final

# 推理过程只放回答正文，截一段：排行榜只是存档，不参与判分
TRACE_CHARS = 4000


def export(run_dir: Path, *retries: Path) -> Path:
    chosen: dict[str, tuple[dict, str | None, int]] = {}      # 题 → (trial, 最终答案, 来自第几个运行)
    for source, d in enumerate((run_dir, *retries)):
        trials = [json.loads(line) for line in (d / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for t in sorted(trials, key=lambda t: t["trial"]):
            final = extract_final(t["answer"])
            if t["case_id"] not in chosen or (chosen[t["case_id"]][1] is None and final is not None):
                chosen[t["case_id"]] = (t, final, source)
    out = run_dir / "submission.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for case_id, (t, final, _) in sorted(chosen.items(), key=lambda kv: int(kv[0].removeprefix("dab-"))):
            f.write(json.dumps({"task_id": case_id.removeprefix("dab-"), "agent_answer": final or "",
                                "reasoning_trace": t["answer"][:TRACE_CHARS]}, ensure_ascii=False) + "\n")
    missing = sum(final is None for _, final, _ in chosen.values())
    empty = sum(final == "" for _, final, _ in chosen.values())
    patched = sum(source > 0 for _, _, source in chosen.values())
    print(f"{len(chosen)} 题 → {out}"
          + (f"；{patched} 题用的是补跑的结果" if patched else "")
          + (f"；{empty} 题的最终答案本身是空（空列表）" if empty else "")
          + (f"；{missing} 题没写出最终答案，交的是空串" if missing else ""))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit("用法：python -m evals.dabstep.submission <运行目录> [补跑的运行目录 …]")
    export(*(Path(a) for a in sys.argv[1:]))
