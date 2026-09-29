"""core —— Agent 运行时：主循环、消息、上下文管理，以及模型和工具的接口。

core 不 import 包外的任何模块（tests/test_imports.py 守着）。
"""

from .agent import Agent, AwaitingUser, InterruptedTurn, PendingQuestion
from .context import BaseContext, ClearOldToolResults, Context, ContextEdit, KeepRecentTurns
from .messages import LLMResponse, Message, ToolCall, Usage
from .provider import LLMProvider
from .state import AgentState, ToolRun
from .tools import NeedsUserInput, Tool, ToolOutput, ToolRegistry

__all__ = [
    "Agent", "AgentState", "AwaitingUser", "InterruptedTurn", "PendingQuestion", "ToolRun",
    "BaseContext", "Context", "ContextEdit", "ClearOldToolResults", "KeepRecentTurns",
    "Message", "ToolCall", "LLMResponse", "Usage",
    "LLMProvider", "NeedsUserInput", "Tool", "ToolOutput", "ToolRegistry",
]
