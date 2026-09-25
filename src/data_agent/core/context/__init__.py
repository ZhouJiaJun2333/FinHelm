"""上下文管理：只追加的历史 + 一组按顺序套用的工序（见 base.py）。

    Context([ClearOldToolResults(), CompactHistory(llm_summarizer(llm))])   先清理，不够再压缩
"""

from .base import BaseContext, Context, ContextEdit, Entry, Marker, Measure, Prompt
from .compaction import CompactHistory, HistoryCompacted, Summarize, Summary, llm_summarizer
from .tool_results import ClearOldToolResults, ToolResultsCleared
from .turns import KeepRecentTurns, turn_starts

__all__ = [
    "BaseContext", "Context", "ContextEdit", "Entry", "Marker", "Measure", "Prompt",
    "ClearOldToolResults", "ToolResultsCleared", "KeepRecentTurns", "turn_starts",
    "CompactHistory", "HistoryCompacted", "Summarize", "Summary", "llm_summarizer",
]
