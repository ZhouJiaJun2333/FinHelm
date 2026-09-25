"""read_file：按行读一个文本文件，带行号（学 Claude Code 的 Read）。

为什么要它：在沙箱里 print 一份 2 万字的手册，要被工具结果的 6000 字上限切成四五段，一段一步；
DABstep 的首个基线里，每道题开头光读文档就花掉约 10 步，三分之一的 trial 步数耗尽。
这个工具的上限大（一次能读完一份手册），读不完就告诉模型下次从哪行接着读。

只读文本：说明文档、手册、代码表、JSON。表格要计算、筛选还是用 run_python —— 一行行看既慢又容易看漏。
rerunnable：只读，旧结果可以被上下文清理，要的话再读一次。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.tools import Tool, ToolOutput
from .paths import SandboxPaths

MAX_CHARS = 40_000       # 一次最多给这么多字符（约 1 万 token），读完一份手册够了
MAX_LINES = 2000
LINE_CHARS = 1000        # 单行太长（压成一行的 JSON）只给开头


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "读一个文本文件（.md、.txt、.csv、.json 等），每行带行号。说明文档、手册、代码表用它读，"
        f"一次最多约 {MAX_CHARS // 1000}k 字符、{MAX_LINES} 行，比在沙箱里 print 省步数；没读完会告诉你下次从哪行接着读。"
        "能读工作目录（inputs/、figures/）和 /data/ 下的文件。"
        "表格数据要计算、筛选、统计，用 run_python 读进来算，别在这里一行行看。Excel 等二进制文件读不了。"
    )
    rerunnable = True
    max_output_chars = MAX_CHARS + 1000      # 留出首尾说明的位置：截断由这里按整行做，不让框架从中间截

    class Args(BaseModel):
        path: str = Field(description="文件路径，比如 /data/manual.md、inputs/说明.txt")
        offset: int = Field(default=1, ge=1, description="从第几行开始读（从 1 数）")
        limit: int = Field(default=MAX_LINES, ge=1, le=MAX_LINES, description="最多读几行")

    def __init__(self, paths: SandboxPaths) -> None:
        self.paths = paths

    def run(self, args: Args) -> ToolOutput:
        path = self.paths.resolve(args.path)
        name = self.paths.display(path)
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ValueError(f"{name} 是二进制文件（Excel、图片之类），读不了。用 run_python 打开。")
        lines = raw.decode("utf-8-sig", errors="replace").splitlines()
        total = len(lines)
        if total == 0:
            return ToolOutput(f"{name} 是空文件。", summary=f"读了 {name}（空）")
        if args.offset > total:
            raise ValueError(f"{name} 只有 {total} 行，offset={args.offset} 超出了。")

        shown: list[str] = []
        size = 0
        start = args.offset
        for n in range(start, min(start + args.limit, total + 1)):
            line = lines[n - 1]
            if len(line) > LINE_CHARS:
                line = line[:LINE_CHARS] + f"…（这一行共 {len(line)} 个字符，只显示了开头）"
            text = f"{n:>6}\t{line}"
            if shown and size + len(text) + 1 > MAX_CHARS:
                break
            shown.append(text)
            size += len(text) + 1
        end = start + len(shown) - 1

        head = f"{name}：共 {total} 行" + ("" if (start, end) == (1, total) else f"，这是第 {start}–{end} 行")
        tail = f"\n（后面还有 {total - end} 行，接着读用 offset={end + 1}）" if end < total else ""
        return ToolOutput(f"{head}\n" + "\n".join(shown) + tail, summary=f"读了 {name} 第 {start}–{end} 行")
