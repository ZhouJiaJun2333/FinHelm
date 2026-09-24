"""报告：把一批 trial 汇总成数字，并和上一次运行逐题对比。

三个准确率，从宽到严：
    结果对     Agent 跑过的某条 SQL 查出了标准答案（允许多几列）
    严格       而且列数也一样（BIRD 的判法）
    回答对     结果对，而且最终回答里的数字对得上 —— 用户真正看到的是这个

pass@1 = 跑一次答对的概率（所有 trial 的平均）
pass^k = 连跑 k 次全对的比例（τ-bench 的指标）：时对时错的 Agent 比稳定答不上来的更坑人
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .cases import Case
from .runner import Trial


def summarize(cases: list[Case], trials: list[Trial]) -> dict[str, Any]:
    by_case: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_case[t.case_id].append(t)

    def rates(ts: list[Trial]) -> dict[str, float]:
        n = len(ts) or 1
        return {
            "结果对": sum(t.result_ok for t in ts) / n,
            "严格": sum(bool(t.result and t.result.strict) for t in ts) / n,
            "回答对": sum(t.answer_ok for t in ts) / n,
        }

    per_case = {}
    for c in cases:
        ts = by_case.get(c.id, [])
        per_case[c.id] = {
            "question": c.question,
            "tags": list(c.tags),
            "trials": len(ts),
            "结果对": sum(t.result_ok for t in ts),
            "回答对": sum(t.answer_ok for t in ts),
            "全对": bool(ts) and all(t.answer_ok for t in ts),
            "失败": dict(Counter(t.failure for t in ts if t.failure)),
        }

    by_tag: dict[str, list[Trial]] = defaultdict(list)
    for c in cases:
        for tag in c.tags:
            by_tag[tag].extend(by_case.get(c.id, []))

    n = len(trials) or 1
    return {
        "trials": len(trials),
        "cases": len(cases),
        "pass@1": rates(trials),
        "pass^k": sum(pc["全对"] for pc in per_case.values()) / (len(cases) or 1),
        "平均步数": round(sum(t.steps for t in trials) / n, 2),
        "平均输入token": round(sum(t.usage.prompt_tokens for t in trials) / n),
        "平均输出token": round(sum(t.usage.output for t in trials) / n),
        "平均耗时s": round(sum(t.elapsed_s for t in trials) / n, 1),
        "步数耗尽": sum(t.step_limit for t in trials),
        "运行出错": sum(bool(t.error) for t in trials),
        "失败分类": dict(Counter(t.failure for t in trials if t.failure)),
        "按标签": {tag: {**rates(ts), "trials": len(ts)} for tag, ts in sorted(by_tag.items())},
        "逐题": per_case,
    }


def compare(prev: dict[str, Any], cur: dict[str, Any]) -> list[str]:
    """逐题对比两次运行：哪些题变好了、哪些变坏了。找回归问题最直接的办法。"""
    lines = []
    for cid, c in cur["逐题"].items():
        p = prev["逐题"].get(cid)
        if p is None or not p["trials"] or not c["trials"]:
            continue
        before, after = p["回答对"] / p["trials"], c["回答对"] / c["trials"]
        if abs(after - before) > 1e-9:
            arrow = "↑" if after > before else "↓"
            lines.append(f"{arrow} {cid} {c['question']}：{p['回答对']}/{p['trials']} → {c['回答对']}/{c['trials']}")
    return lines


def render(meta: dict[str, Any], s: dict[str, Any], diff: list[str] | None = None,
           trials: list[Trial] | None = None) -> str:
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    p1 = s["pass@1"]
    out = [
        f"# 评测报告：{meta['cases']}（{meta['model']}）",
        "",
        f"- 时间：{meta['started']}　版本：{meta['git']}{'（有未提交的改动）' if meta['dirty'] else ''}",
        f"- 题库指纹：{meta['cases_sha1']}　提示词指纹：{meta['prompt_sha1']}　每题 {meta['trials']} 次",
        "",
        "| 指标 | 值 |",
        "|:--|--:|",
        f"| pass@1 结果对 | {pct(p1['结果对'])} |",
        f"| pass@1 严格（列数也一样） | {pct(p1['严格'])} |",
        f"| **pass@1 回答对** | **{pct(p1['回答对'])}** |",
        f"| pass^{meta['trials']}（每次都回答对的题） | {pct(s['pass^k'])} |",
        f"| 平均步数 | {s['平均步数']} |",
        f"| 平均 token（输入 / 输出） | {s['平均输入token']:,} / {s['平均输出token']:,} |",
        f"| 平均耗时 | {s['平均耗时s']}s |",
        f"| 步数耗尽 / 运行出错 | {s['步数耗尽']} / {s['运行出错']} |",
        "",
    ]
    if s["失败分类"]:
        out += ["## 失败分类", ""]
        out += [f"- {k}：{v} 次" for k, v in sorted(s["失败分类"].items(), key=lambda kv: -kv[1])]
        out.append("")

    out += ["## 按标签", "", "| 标签 | trial 数 | 结果对 | 回答对 |", "|:--|--:|--:|--:|"]
    for tag, r in sorted(s["按标签"].items(), key=lambda kv: kv[1]["回答对"]):
        out.append(f"| {tag} | {r['trials']} | {pct(r['结果对'])} | {pct(r['回答对'])} |")

    out += ["", "## 逐题", "", "| 题 | 问题 | 回答对 | 失败 |", "|:--|:--|--:|:--|"]
    for cid, c in s["逐题"].items():
        fails = "，".join(f"{k}×{v}" for k, v in c["失败"].items())
        out.append(f"| {cid} | {c['question']} | {c['回答对']}/{c['trials']} | {fails} |")

    if diff is not None:
        out += ["", f"## 和上一次比（{meta.get('compared_with', '')}）", ""]
        out += diff or ["没有变化。"]

    if trials:
        bad = [t for t in trials if t.failure]
        if bad:
            out += ["", "## 失败详情", ""]
            for t in bad:
                reason = (t.error or t.grade_error
                          or (t.result.reason if t.result and not t.result.lenient else "")
                          or (f"回答里没找到：{t.answer_check.missing[:5]}" if t.answer_check else ""))
                out.append(f"- **{t.case_id} #{t.trial}** {t.failure}：{reason}")
                if t.final_sql:
                    out.append(f"  - 最后一条 SQL：`{' '.join(t.final_sql.split())[:300]}`")
    return "\n".join(out) + "\n"
