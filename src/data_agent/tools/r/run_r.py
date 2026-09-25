"""run_r：在沙箱里跑 R。医学统计（meta 分析、偏倚风险图）的主力，模板函数在 templates.R。

和 run_python 共用工作目录：inputs/ 下是用户上传的文件，figures/ 下是图，两种语言靠文件交换数据。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..sandbox import KernelSpec, SandboxTool

HERE = Path(__file__).parent
R_KERNEL = KernelSpec(files=(HERE / "kernel.R", HERE / "templates.R"), command=("Rscript",))


class RunRTool(SandboxTool):
    name = "run_r"
    language = "R"
    missing_name = re.compile(r"object '.+' not found|找不到对象")
    description = (
        "在沙箱里执行 R 代码。内核有状态，变量在调用之间保留。"
        "做 meta 分析、森林图、偏倚风险图时**先用模板函数**（fh_ 开头，RevMan 5 的算法和版式，有测试），"
        "fh_help() 列出全部模板和参数；模板做不了的才自己写，并在回答里说明这部分不是模板。"
        "已经加载 meta、metafor、robvis、readxl、writexl、ggplot2。用户上传的文件在 inputs/ 下。"
        '要给用户看的长表用 save_result(df, "标题") 存下来，会得到编号，回答里写 {{r5}} 引用。'
        "每个可见的顶层表达式都会打印（和 R 控制台一样）；画的图执行完自动保存成 PNG 并告诉你路径，"
        "要 PDF / TIFF 就自己写到 figures/ 下（这次调用自己存了图，就不再自动保存）。"
        "沙箱不能联网、不能装包。内核可能重启（超时、恢复会话），变量没了就重新运行定义它们的代码。"
    )
