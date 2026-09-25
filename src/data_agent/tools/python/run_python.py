"""run_python：在沙箱里跑 Python，做 SQL 不方便的计算，读用户给的文件，画图。

有状态：变量在调用之间保留。load_result("r3") 直接拿到 run_sql 的完整结果，数字不经过模型的手。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..sandbox import KernelSpec, SandboxTool

PYTHON_KERNEL = KernelSpec(
    files=(Path(__file__).with_name("kernel.py"),),
    command=("python", "-u"),
    env=(("MPLCONFIGDIR", "/tmp"),),
)


class RunPythonTool(SandboxTool):
    name = "run_python"
    language = "Python"
    missing_name = re.compile(r"NameError")
    description = (
        "在沙箱里执行 Python 代码：SQL 不方便的计算（收益率、同比环比、累计、波动率、回归、分布）、"
        "读用户上传的文件（inputs/ 下的 Excel、CSV）、画图。内核有状态，变量在调用之间保留（像 Jupyter）。"
        "已经导入 pandas as pd、numpy as np、matplotlib.pyplot as plt；还能用 scipy、statsmodels、openpyxl。"
        '用 load_result("r3") 取 run_sql 结果 r3 的完整数据（DataFrame），不要把数字手抄进代码。'
        '要给用户看的长表用 save_result(df, "标题") 存下来，会得到编号，回答里写 {{r5}} 引用。'
        "最后一行是表达式就显示它的值；用 plt 画的图执行完自动保存，会告诉你文件路径。"
        "中文字体已经配好，图里直接写中文，不用改字体设置。要 PDF / SVG 等格式就自己 savefig 到 figures/ 下。"
        "图不会自动显示给你，不用 plt.show()（沙箱里也没有 IPython）。"
        "沙箱不能联网、不能连数据库（取数用 run_sql）、不能装包。"
        "内核可能重启（超时、恢复会话），变量没了就重新运行定义它们的代码。"
    )
