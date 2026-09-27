"""报告：把一批 trial 汇总成数字，并和上一次运行逐题对比。

几个准确率：
    结果对     Agent 跑过的某条 SQL 查出了标准答案（允许多几列）
    严格       而且列数也一样
    回答对     结果对，而且最终回答里的数字对得上 —— 用户真正看到的是这个。
               另外，SQL 没对上、但标准答案是单个数且回答里算对了，也算回答对（报告里单独列出来）
    最后一条   只看最后执行的那条 SQL、列数也一样、只认 BIRD 原版标准答案 —— BIRD 的规则
    提交       有提交轮的题库（BIRD）：提交轮交的那条 SQL，同样按 BIRD 的规则
               这两个都是「按值比」。官方脚本连 Python 类型都比：标准 SQL 里 CAST(... AS REAL)
               出来是 float，Agent 写 100.0 * ... 出来是 Decimal，前 15 位一样也算错。
               对外说的分数以官方脚本为准（evals/bird/official.py），会比这两个低

pass@1 = 跑一次答对的概率（所有 trial 的平均）
pass^k = 连跑 k 次全对的比例（τ-bench 的指标）：时对时错的 Agent 比稳定答不上来的更坑人

缓存命中率按 token 加权（Σ命中 / Σ输入），不是每题命中率的平均 —— 钱是按 token 算的。
分成两段看，因为它们靠的是不同的东西：
    首次调用   只有系统提示词 + 工具定义可能命中，靠的是「别的 trial 刚发过同样的开头」，
               跟并发、跑题顺序有关，不是我们的上下文策略管得了的
    后续调用   命中的是本题自己的历史前缀 —— 第 5 步要优化的就是这一段
    清理后调用 后续调用里紧跟在清理之后的那几次：历史中间被改了，缓存断在第一处改动
    压缩后调用 紧跟在压缩之后：开头就换成了摘要，除了系统提示词基本全断
    写摘要     压缩时写摘要的那次调用。原样发对话、末尾追加要求，应该几乎全中
               这两项只有多轮会话里才会有
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from data_agent.core.messages import Usage

from .cases import Case, CaseSet
from .runner import SessionTrial, Trial


def hit_rate(usages: list[Usage]) -> float | None:
    """Σ命中 / Σ输入。没有输入（比如一题只调了一次，没有后续调用）返回 None。"""
    total = sum(u.prompt_tokens for u in usages)
    return sum(u.cache_read for u in usages) / total if total else None


def cache_stats(trials: list[Trial] | list[SessionTrial]) -> dict[str, Any]:
    """trials 可以是单题的 Trial，也可以是整段会话（SessionTrial 有同名的 calls / usage / after_edit）。"""
    first = [t.calls[0] for t in trials if t.calls]
    later = [u for t in trials for u in t.calls[1:]]
    compacted = [(t, set(t.after_compact)) for t in trials]
    cleared = [t.calls[i] for t, c in compacted for i in t.after_edit if i > 0 and i not in c]
    after_compact = [t.calls[i] for t, c in compacted for i in c if i > 0]
    n = len(trials) or 1
    return {
        "命中率": hit_rate([t.usage for t in trials]),
        "首次调用命中率": hit_rate(first),
        "后续调用命中率": hit_rate(later),
        "清理后调用命中率": hit_rate(cleared),
        "清理后调用次数": len(cleared),
        "压缩后调用命中率": hit_rate(after_compact),
        "压缩后调用次数": len(after_compact),
        "写摘要命中率": hit_rate([u for t in trials for u in t.summary_calls]),
        "写摘要次数": sum(len(t.summary_calls) for t in trials),
        "平均命中token": round(sum(t.usage.cache_read for t in trials) / n),
        "平均写入token": round(sum(t.usage.cache_write for t in trials) / n),
    }


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
            "最后一条": sum(t.final_strict for t in ts) / n,
            "提交": sum(bool(t.submission and t.submission.strict) for t in ts) / n,
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
            "缓存命中率": hit_rate([t.usage for t in ts]),
            "失败": dict(Counter(t.failure for t in ts if t.failure)),
        }

    by_tag: dict[str, list[Trial]] = defaultdict(list)
    for c in cases:
        for tag in c.tags:
            by_tag[tag].extend(by_case.get(c.id, []))

    n = len(trials) or 1
    submitted = [t.submission for t in trials if t.submission is not None]
    return {
        "trials": len(trials),
        "cases": len(cases),
        "pass@1": rates(trials),
        "pass^k": sum(pc["全对"] for pc in per_case.values()) / (len(cases) or 1),
        "平均步数": round(sum(t.steps for t in trials) / n, 2),
        "平均输入token": round(sum(t.usage.prompt_tokens for t in trials) / n),
        "平均输出token": round(sum(t.usage.output for t in trials) / n),
        "缓存": cache_stats(trials),
        "平均耗时s": round(sum(t.elapsed_s for t in trials) / n, 1),
        "步数耗尽": sum(t.step_limit and not t.wrapped_up for t in trials),
        "收尾": sum(t.wrapped_up for t in trials),
        "收尾后答对": sum(t.wrapped_up and t.answer_ok for t in trials),
        "重复调用提醒": sum(t.repeat_warnings for t in trials),
        "被提醒的trial": sum(bool(t.repeat_warnings) for t in trials),
        "不判分": bool(trials) and not any(t.graded for t in trials),
        "运行出错": sum(bool(t.error) for t in trials),
        "失败分类": dict(Counter(t.failure for t in trials if t.failure)),
        "回答里算对": [f"{t.case_id} #{t.trial}" for t in trials if t.text_ok],
        "提交轮": {
            "次数": len(submitted),
            "没交SQL": sum(not s.sql for s in submitted),
            "平均步数": round(sum(s.steps for s in submitted) / (len(submitted) or 1), 2),
            "平均输入token": round(sum(s.usage.prompt_tokens for s in submitted) / (len(submitted) or 1)),
        },
        "按标签": {tag: {**rates(ts), "trials": len(ts)} for tag, ts in sorted(by_tag.items())},
        "逐题": per_case,
        **({"上传文件": file_stats(files)} if (files := [t for t in trials if t.no_sql]) else {}),
        "只看最终答案": bool(trials) and all(t.official for t in trials),
    }


def file_stats(trials: list[Trial]) -> dict[str, Any]:
    """上传文件的题（research）：自己写代码画了图的 trial 里，交付前看了图的占多少。"""
    plotted = [t for t in trials if t.custom_plots]
    return {
        "自己画图的trial": len(plotted),
        "画完看了图的trial": sum(t.viewed_after > 0 for t in plotted),
        "自己画图次数": sum(t.custom_plots for t in plotted),
        "画完看图次数": sum(t.viewed_after for t in plotted),
    }


def summarize_sessions(case_set: CaseSet, sessions: list[SessionTrial]) -> dict[str, Any]:
    """多轮题库：每一轮当一道题，套用 summarize；缓存和整理次数按整段会话算。"""
    turns = [t for st in sessions for t in st.turns if t.graded]
    s = summarize(case_set.graded_cases, turns)
    s["缓存"] = cache_stats(sessions)
    recall_ids = {c.id for c in case_set.graded_cases if c.match == "answer"}
    recalls = [t for t in turns if t.case_id in recall_ids]
    n = len(sessions) or 1
    fillers = [t for st in sessions for t in st.turns if not t.graded]
    s["会话"] = {
        "会话数": len(sessions),
        "平均清理次数": round(sum(st.edit_count("ToolResultsCleared") for st in sessions) / n, 2),
        "平均压缩次数": round(sum(st.edit_count("HistoryCompacted") for st in sessions) / n, 2),
        "回忆轮": len(recalls),
        "回忆轮答对": sum(t.answer_ok for t in recalls),
        "回忆轮重查": sum(bool(t.sql_calls) for t in recalls),
        # 一轮里整理了两次以上：清完还贴着门槛、又过线。门槛缩小的评测里会放大，真实门槛下该很少
        "一轮多次整理": sum(t.context_edits >= 2 for st in sessions for t in st.turns),
        "压缩失败": sum(t.compaction_failures for st in sessions for t in st.turns),
        "总轮数": sum(len(st.turns) for st in sessions),
        # 填充轮不判分，出错不进「运行出错」—— 但它会让这一轮回滚、上下文没按预期变大，得单独看
        "填充轮出错": dict(Counter(t.error.split(":")[0] for st in sessions for t in st.turns
                                    if not t.graded and t.error)),
        # 填充轮就是「列清单」：用引用让界面展示整张表，模型就不用把几十行抄一遍
        "填充轮": len(fillers),
        "填充轮用引用": sum(t.refs > 0 for t in fillers),
        "填充轮平均输出token": round(sum(t.usage.output for t in fillers) / (len(fillers) or 1)),
        "平均会话输入token": round(sum(st.usage.prompt_tokens for st in sessions) / n),
        "平均会话耗时s": round(sum(st.elapsed_s for st in sessions) / n, 1),
        "逐段": {
            st_id: [{"trial": st.trial, "整理": [f"第{e['turn']}轮{_KIND.get(e['kind'], e['kind'])}"
                                                  f" {e['before']:,}→{e['after']:,}" for e in st.edits]}
                    for st in sessions if st.session_id == st_id]
            for st_id in dict.fromkeys(st.session_id for st in sessions)
        },
    }
    return s


_KIND = {"ToolResultsCleared": "清理", "HistoryCompacted": "压缩"}


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
    pct = lambda x: "—" if x is None else f"{x:.0%}"  # noqa: E731
    p1 = s["pass@1"]
    cache = s["缓存"]
    files = s.get("上传文件")
    hidden = s.get("不判分")          # 答案不公开：对错的数一律不显示，免得 0% 看着像全错
    out = [
        f"# 评测报告：{meta['cases']}（{meta['model']}）",
        "",
        f"- 时间：{meta['started']}　版本：{meta['git']}{'（有未提交的改动）' if meta['dirty'] else ''}",
        f"- 题库指纹：{meta['cases_sha1']}　提示词指纹：{meta['prompt_sha1']}　每题 {meta['trials']} 次",
        *([f"- 标签：{meta['label']}"] if meta.get("label") else []),
        *([f"- 按新规则重判过：{meta['regraded']}"] if meta.get("regraded") else []),
        *([f"- 临时配置（--set）：{meta['overrides']}"] if meta.get("overrides") else []),
        *([f"- 端点：{meta['base_url']}"] if meta.get("base_url") else []),
        "",
        "| 指标 | 值 |",
        "|:--|--:|",
        *([] if s.get("只看最终答案") else
          [f"| pass@1 要求的步骤都做了（模板、图、指出数据问题） | {pct(p1['结果对'])} |"] if files else [
            f"| pass@1 结果对 | {pct(p1['结果对'])} |",
            f"| pass@1 严格（列数也一样） | {pct(p1['严格'])} |"]),
        f"| **pass@1 回答对** | **{pct(p1['回答对'])}** |" if not hidden else
        "| **pass@1 回答对** | 答案不公开，没有判分 |",
        *([f"| 　其中 SQL 没对上、回答里把数算对了 | {len(s['回答里算对'])} 次 |"] if s.get("回答里算对") else []),
        *([] if files else [f"| pass@1 只看最后一条 SQL（BIRD 的规则，按值比） | {pct(p1['最后一条'])} |"]),
        *([f"| **pass@1 提交轮交的 SQL（BIRD 的规则，按值比）** | **{pct(p1['提交'])}** |"]
          if s.get("提交轮", {}).get("次数") else []),
        *([] if hidden else [f"| pass^{meta['trials']}（每次都回答对的题） | {pct(s['pass^k'])} |"]),
        f"| 平均步数 | {s['平均步数']} |",
        f"| 平均 token（输入 / 输出） | {s['平均输入token']:,} / {s['平均输出token']:,} |",
        f"| 缓存命中率（全部 / 首次调用 / 后续调用） | {pct(cache['命中率'])} / "
        f"{pct(cache['首次调用命中率'])} / {pct(cache['后续调用命中率'])} |",
        f"| 平均缓存 token（命中 / 写入） | {cache['平均命中token']:,} / {cache['平均写入token']:,} |",
        *[f"| {k}后调用命中率（{cache[k + '后调用次数']} 次） | {pct(cache[k + '后调用命中率'])} |"
          for k in ("清理", "压缩") if cache.get(k + "后调用次数")],
        *([f"| 写摘要命中率（{cache['写摘要次数']} 次） | {pct(cache['写摘要命中率'])} |"]
          if cache.get("写摘要次数") else []),
        f"| 平均耗时 | {s['平均耗时s']}s |",
        f"| 步数耗尽 / 运行出错 | {s['步数耗尽']} / {s['运行出错']} |",
        *([f"| 步数用完后收尾（其中答对） | {s['收尾']}（{s['收尾后答对']}） |"] if s.get("收尾") else []),
        *([f"| 重复调用提醒（次 / trial 数） | {s['重复调用提醒']} / {s['被提醒的trial']} |"]
          if s.get("重复调用提醒") else []),
        "",
    ]
    if "会话" in s:
        ss = s["会话"]
        out += [
            "## 会话", "",
            f"- {ss['会话数']} 段会话，平均每段 {ss['平均会话输入token']:,} 输入 token、{ss['平均会话耗时s']}s",
            f"- 平均每段清理 {ss['平均清理次数']} 次、压缩 {ss['平均压缩次数']} 次",
            f"- 回忆轮 {ss['回忆轮']} 次：答对 {ss['回忆轮答对']}，其中重新查了 {ss['回忆轮重查']} 次"
            "（标签「回忆:查不到」的只能靠上下文 / 摘要，见按标签一节）",
            f"- 一轮里整理 ≥ 2 次：{ss.get('一轮多次整理', 0)} / {ss.get('总轮数', 0)} 轮",
            f"- 填充轮（列清单）：回答里引用结果 {ss.get('填充轮用引用', 0)} / {ss.get('填充轮', 0)} 轮，"
            f"平均输出 {ss.get('填充轮平均输出token', 0):,} token",
            *([f"- ⚠️ 填充轮出错（这一轮回滚，没撑大上下文）："
               + "，".join(f"{k} × {v}" for k, v in ss["填充轮出错"].items())] if ss.get("填充轮出错") else []),
            "",
        ]
        for sid, runs in ss["逐段"].items():
            for r in runs:
                out.append(f"- {sid} #{r['trial']}：{'，'.join(r['整理']) or '没有整理'}")
        out.append("")

    if sub := s.get("提交轮", {}):
        if sub.get("次数"):
            out += ["## 提交轮", "",
                    f"- {sub['次数']} 次，平均 {sub['平均步数']} 步、{sub['平均输入token']:,} 输入 token"
                    "（不算进上面的步数和 token）",
                    f"- 没交出 SQL（按最后一条算）：{sub['没交SQL']} 次",
                    "- 对外的分数用官方脚本算（python -m evals.bird.official <这个目录>）：官方连类型都比，"
                    "float 和 Decimal 值一样也算错，会比上面的低", ""]
    if files and files["自己画图的trial"]:
        out += ["## 看图", "",
                f"- 自己写代码画了图（没用 fh_ 模板）的 trial：{files['自己画图的trial']} 个，"
                f"其中交付前用 view_image 看了图的 {files['画完看了图的trial']} 个",
                f"- 按次数：自己画图 {files['自己画图次数']} 次，画完看了的 {files['画完看图次数']} 次", ""]
    if s.get("回答里算对"):
        out += ["## SQL 没对上、回答里算对了", "",
                "标准答案是单个算出来的数（比例、平均数），回答里说到了：",
                "", "，".join(s["回答里算对"]), ""]

    if s["失败分类"]:
        out += ["## 失败分类", ""]
        out += [f"- {k}：{v} 次" for k, v in sorted(s["失败分类"].items(), key=lambda kv: -kv[1])]
        out.append("")

    out += ["## 按标签", "", "| 标签 | trial 数 | 结果对 | 回答对 |", "|:--|--:|--:|--:|"]
    for tag, r in sorted(s["按标签"].items(), key=lambda kv: kv[1]["回答对"]):
        out.append(f"| {tag} | {r['trials']} | " + ("— | — |" if hidden else f"{pct(r['结果对'])} | {pct(r['回答对'])} |"))

    out += ["", "## 逐题", "", "| 题 | 问题 | 回答对 | 缓存命中 | 失败 |", "|:--|:--|--:|--:|:--|"]
    for cid, c in s["逐题"].items():
        fails = "，".join(f"{k}×{v}" for k, v in c["失败"].items())
        out.append(f"| {cid} | {c['question'].splitlines()[0]} | {'—' if hidden else f"{c['回答对']}/{c['trials']}"} | {pct(c['缓存命中率'])} | {fails} |")

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
