"""core —— Agent 运行时：主循环、消息、上下文管理，以及它依赖的三个接口。

    agent.py      主循环
    messages.py   中立的消息结构（整个项目的通用语）
    provider.py   LLMProvider：模型接口       实现在 llm/
    tools.py      Tool / ToolRegistry：工具接口  实现在 tools/
    context/      上下文管理

**core 不 import 包外的任何模块**（tests/test_imports.py 守着这条）。依赖只有一个方向：
llm/、tools/、app.py、cli.py 依赖 core，core 不知道它们的存在。
"""

from .agent import Agent
from .context import BaseContext, ClearOldToolResults, Context, ContextEdit, KeepRecentTurns
from .messages import LLMResponse, Message, ToolCall, Usage
from .provider import LLMProvider
from .tools import Tool, ToolOutput, ToolRegistry

__all__ = [
    "Agent",
    "BaseContext", "Context", "ContextEdit", "ClearOldToolResults", "KeepRecentTurns",
    "Message", "ToolCall", "LLMResponse", "Usage",
    "LLMProvider", "Tool", "ToolOutput", "ToolRegistry",
]
