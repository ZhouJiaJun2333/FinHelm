"""上下文管理 —— 「对话变长之后怎么办」。

    base.py          Context：只追加的历史（消息 + 标记），按顺序套用编辑工序得到视图
                     ContextEdit：一道工序的接口，自己不持有状态
                     Marker：工序做出的决定，作为标记记进历史
    tool_results.py  ClearOldToolResults：超阈值时把较早的工具结果换成带线索的占位
    turns.py         KeepRecentTurns：按回合裁剪

用法：
    Context()                                     全量保留
    Context([ClearOldToolResults(trigger_tokens=100_000)])
    Context([KeepRecentTurns(6), ClearOldToolResults()])   按顺序套用
"""

from .base import BaseContext, Context, ContextEdit, Entry, Marker, Measure
from .tool_results import ClearOldToolResults, ToolResultsCleared
from .turns import KeepRecentTurns

__all__ = [
    "BaseContext", "Context", "ContextEdit", "Entry", "Marker", "Measure",
    "ClearOldToolResults", "ToolResultsCleared", "KeepRecentTurns",
]
