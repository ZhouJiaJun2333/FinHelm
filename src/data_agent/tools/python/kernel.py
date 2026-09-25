"""沙箱里的 Python 内核：一个常驻进程，变量在调用之间保留（像 Jupyter）。

在容器里跑，不 import 项目里的任何东西。和宿主机按行说 JSON：
    宿主 → 内核（stdin）   {"op": "exec", "code": ..., "timeout": 秒}
                           {"op": "data", "ref": "r3", ...}          回应 need
    内核 → 宿主（stdout）  {"op": "ready"}  {"op": "need", "ref": "r3"}  {"op": "done", ...}
用户代码的 print 收进 StringIO；C 扩展直接写 fd 1 的内容转去 stderr，搅不乱协议。
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import linecache
import os
import signal
import sys
import traceback

PROTO_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
PROTO_IN = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")   # Windows 默认是 GBK
os.dup2(2, 1)
sys.stdin = io.StringIO()            # 用户代码里的 input() 读不到协议

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    CJK_FONT = "WenQuanYi Micro Hei"

    def _keep_cjk(validate):
        # 模型常习惯性地把字体设成 DejaVu Sans（别处没中文字体）：不管设成什么，末尾都补上中文字体。
        # 挂在 font.family 上才会逐字回退；sans-serif 列表只会取第一个能用的字体
        def wrapped(value):
            fonts = validate(value)
            return fonts if CJK_FONT in fonts else [*fonts, CJK_FONT]
        return wrapped

    plt.rcParams.validate["font.family"] = _keep_cjk(plt.rcParams.validate["font.family"])
    plt.rcParams["font.family"] = "sans-serif"
except ImportError:                  # 本地测试环境可以不装
    plt = None

pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

MAX_OUTPUT_CHARS = 20_000
CELL = "<cell>"


class Timeout(BaseException):
    """继承 BaseException：用户代码里的 except Exception 拦不住它。"""


def send(msg: dict) -> None:
    PROTO_OUT.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")


def receive() -> dict:
    line = PROTO_IN.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


# ---------------------------------------------------------------- SQL 结果
_results: dict[str, pd.DataFrame] = {}


def load_result(ref: str) -> pd.DataFrame:
    """run_sql 的某个结果（比如 "r3"）的完整数据。每次给一份副本，改了不影响下次取。"""
    ref = str(ref).strip()
    if ref not in _results:
        send({"op": "need", "ref": ref})
        reply = receive()
        if reply.get("error"):
            raise LookupError(reply["error"])
        frame = pd.DataFrame(reply["rows"], columns=reply["columns"])
        for col in reply.get("dates", []):
            frame[col] = pd.to_datetime(frame[col])
        if reply.get("truncated"):
            print(f"注意：{ref} 超过 {len(frame)} 行被截断了，这里只有前 {len(frame)} 行。")
        _results[ref] = frame
    return _results[ref].copy()


def _blocked_input(*_args, **_kwargs):
    raise RuntimeError("沙箱里没有人能回答 input()。")


NAMESPACE: dict = {
    "__name__": "__main__",
    "pd": pd, "np": np, "plt": plt,
    "load_result": load_result, "input": _blocked_input,
}


# ---------------------------------------------------------------- 执行
def run(code: str, timeout: float) -> dict:
    out = io.StringIO()
    value = error = None
    linecache.cache[CELL] = (len(code), None, code.splitlines(True), CELL)
    try:
        tree = ast.parse(code, CELL, "exec")
    except SyntaxError as exc:
        return {"output": "", "value": None, "error": _format_error(exc), "figures": []}
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    before = _figure_files()

    _arm(timeout)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(tree, CELL, "exec"), NAMESPACE)
            if last is not None:
                result = eval(compile(ast.Expression(last.value), CELL, "eval"), NAMESPACE)
                if result is not None and not _is_plot(result):
                    # numpy 2 的标量 repr 是 np.float64(1001.25)，模型要的是 1001.25
                    value = _cap(str(result) if isinstance(result, np.generic) else repr(result))
    except Timeout:
        error = f"执行超过 {timeout:g} 秒，被中断了。已经算完的变量还在。"
    except BaseException as exc:  # noqa: BLE001 —— 连 SystemExit 也拦下，内核不能退
        error = _format_error(exc)
    finally:
        _arm(0)
    written = [f for f, mtime in _figure_files().items() if before.get(f) != mtime]
    return {"output": _cap(out.getvalue()), "value": value, "error": error,
            "figures": written + _save_figures()}


def _figure_files() -> dict[str, float]:
    """figures/ 下的文件和修改时间：用户代码自己 savefig 的 PDF 也要报给宿主。"""
    return {e.path: e.stat().st_mtime for e in os.scandir("figures") if e.is_file()}


def _arm(seconds: float) -> None:
    if hasattr(signal, "setitimer"):         # Windows 上没有，靠宿主机的硬超时
        signal.setitimer(signal.ITIMER_REAL, seconds)


def _on_alarm(_signum, _frame):
    raise Timeout()


def _is_plot(value) -> bool:
    """plt.title(...) 这类返回值没人想看（Jupyter 里要加分号压掉）。"""
    first = value[0] if isinstance(value, list) and value else value
    return type(first).__module__.startswith("matplotlib")


def _format_error(exc: BaseException) -> str:
    """只留用户代码里的帧：库内部几十层调用栈对改代码没用。"""
    te = traceback.TracebackException.from_exception(exc)
    seen, cur = set(), te
    while cur is not None and id(cur) not in seen:        # 连带着 raise ... from 的那几层
        seen.add(id(cur))
        cur.stack = traceback.StackSummary.from_list([f for f in cur.stack if f.filename == CELL])
        cur = cur.__cause__ or cur.__context__
    return "".join(te.format()).replace(f'File "{CELL}", ', "")


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n…（输出太长，只保留了前 {MAX_OUTPUT_CHARS} 个字符）"


def _save_figures() -> list[str]:
    if plt is None or not plt.get_fignums():
        return []
    saved, n = [], 1
    for num in plt.get_fignums():
        while os.path.exists(path := f"figures/fig-{n}.png"):
            n += 1
        plt.figure(num).savefig(path, dpi=120, bbox_inches="tight")
        saved.append(path)
    plt.close("all")
    return saved


def main() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _on_alarm)
    os.makedirs("figures", exist_ok=True)         # 模型会自己往里存 PDF / SVG
    send({"op": "ready", "python": sys.version.split()[0], "pandas": pd.__version__})
    while True:
        msg = receive()
        if msg.get("op") == "exec":
            send({"op": "done", **run(msg["code"], float(msg.get("timeout", 60)))})


if __name__ == "__main__":
    main()
