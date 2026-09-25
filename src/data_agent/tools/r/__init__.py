"""R 工具：run_r。内核在 kernel.R，模板函数在 templates.R，宿主机那头是 tools/sandbox.py。"""

from .run_r import R_KERNEL, RunRTool

__all__ = ["R_KERNEL", "RunRTool"]
