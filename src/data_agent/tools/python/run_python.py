"""run_python：在沙箱里跑 Python，做 SQL 不方便的计算，画图。

有状态：变量在调用之间保留。load_result("r3") 直接拿到 run_sql 的完整结果，
数字不经过模型的手，不会抄错。不是 rerunnable：重跑一次会改变内核里的状态。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from ...core.tools import Tool, ToolOutput
from ...db.connection import QueryResult
from ..sql.results import ResultStore
from .sandbox import Execution, Resolve, Sandbox

RESTART_HINT = "内核重启过，之前定义的变量都没了。需要的话重新运行定义它们的代码，数据用 load_result 重新取。"
# NameError 不一定是重启造成的，只提一句可能
MAYBE_RESTARTED = "如果这个变量是之前的调用里定义的：内核可能重启过（超时、恢复会话），重新运行定义它的代码。"


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "在沙箱里执行 Python 代码，用来做 SQL 不方便的计算和画图：收益率、同比环比、累计、波动率、"
        "回归、分布、图表等。内核有状态，变量在调用之间保留（像 Jupyter）。"
        "已经导入 pandas as pd、numpy as np、matplotlib.pyplot as plt；还能用 scipy、statsmodels。"
        '用 load_result("r3") 取 run_sql 结果 r3 的完整数据（DataFrame），不要把数字手抄进代码。'
        "最后一行是表达式就显示它的值；用 plt 画的图执行完自动保存，会告诉你文件路径。中文字体已经配好，图里直接写中文，不用改字体设置。"
        "沙箱不能联网、不能连数据库（取数用 run_sql）、不能装包。"
        "内核可能重启（超时、恢复会话），变量没了就重新运行定义它们的代码。"
    )

    class Args(BaseModel):
        code: str = Field(description="要执行的 Python 代码")

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run(self, args: Args) -> ToolOutput:
        ex = self.sandbox.run(args.code)
        return ToolOutput(_format(ex), _summarize(ex), details=ex, is_error=ex.error is not None)


def _format(ex: Execution) -> str:
    parts = []
    if ex.output.strip():
        parts.append(ex.output.rstrip())
    if ex.value is not None:
        parts.append(ex.value)
    if ex.figures:
        parts.append("图表已保存（用户能看到这些文件）：\n" + "\n".join(f"- {p}" for p in ex.figures))
    if ex.error:
        parts.append(f"出错了：\n{ex.error.rstrip()}")
    if ex.restarted:
        parts.append(RESTART_HINT)
    elif ex.error and "NameError" in ex.error:
        parts.append(MAYBE_RESTARTED)
    return "\n\n".join(parts) or "执行成功，没有输出。要看结果就 print，或者把表达式放在最后一行。"


def _summarize(ex: Execution) -> str:
    if ex.error:
        return "报错：" + ex.error.strip().splitlines()[-1]
    text = f"输出 {(ex.output + (ex.value or '')).count(chr(10)) + 1} 行"
    return text + (f"，{len(ex.figures)} 张图" if ex.figures else "")


# ---------------------------------------------------------------- SQL 结果 → 内核
def result_resolver(results: ResultStore) -> Resolve:
    def resolve(ref: str) -> dict[str, Any]:
        table = results.get(ref)
        if table is None:
            known = "、".join(results.refs()) or "还没有"
            return {"error": f"没有编号为 {ref} 的结果。本次对话里的编号：{known}"}
        return encode_result(table.result)
    return resolve


def encode_result(result: QueryResult) -> dict[str, Any]:
    """转成能过 JSON 的样子。Decimal 转 float（DataFrame 里本来也是 float）；日期列记下来，内核那头再转回去。"""
    rows = [[_plain(v) for v in row] for row in result.rows]
    dates = [
        col for i, col in enumerate(result.columns)
        if any(isinstance(row[i], dt.date) for row in result.rows)
        and all(row[i] is None or isinstance(row[i], dt.date) for row in result.rows)
    ]
    return {"columns": result.columns, "rows": rows, "dates": dates, "truncated": result.truncated}


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return str(v)
