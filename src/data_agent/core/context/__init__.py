"""上下文管理 —— 「对话变长之后怎么办」。

    base.py          Context：存历史，按顺序套用一组编辑工序得到发给模型的视图
                     ContextEdit：一道工序的接口。加新策略 = 写一个新的 ContextEdit
    tool_results.py  ClearOldToolResults：超阈值时把较早的工具结果换成带线索的占位
    turns.py         KeepRecentTurns：按回合裁剪

用法：
    Context()                                     全量保留
    Context([ClearOldToolResults(trigger_tokens=100_000)])
    Context([KeepRecentTurns(6), ClearOldToolResults()])   按顺序套用
"""

from .base import BaseContext, Context, ContextEdit, Measure
from .tool_results import ClearOldToolResults
from .turns import KeepRecentTurns

__all__ = [
    "BaseContext", "Context", "ContextEdit", "Measure",
    "ClearOldToolResults", "KeepRecentTurns",
]
