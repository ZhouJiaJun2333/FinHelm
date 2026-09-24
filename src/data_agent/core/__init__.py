"""core —— Agent 运行时。不依赖任何具体的模型厂商、工具或数据库。

⚠️ 这里的导出必须是**惰性**的，否则会循环导入：

    settings ──► llm.base ──► core.messages
                                  │
                                  └─ 导入子模块会先执行 core/__init__
                                         │
                                         └─ 如果这里直接 import agent
                                                │
                                                └─ agent 又 import llm.base
                                                       （此时它还没初始化完）→ 炸

    以前没炸，只是因为 app.py 恰好先导入了 core.agent，顺序刚好绕过去了。
    这种「靠导入顺序活着」的代码迟早要出事，所以改成按需导入。
    llm/__init__.py 用的是同一个套路。
"""

from .context import BaseContext, FullContext, TurnWindowContext
from .messages import LLMResponse, Message, ToolCall, Usage

__all__ = [
    "Agent",
    "BaseContext", "FullContext", "TurnWindowContext",
    "Message", "ToolCall", "LLMResponse", "Usage",
]


def __getattr__(name: str):
    # Agent 依赖 llm.base，必须延迟到真正被访问时才导入
    if name == "Agent":
        from .agent import Agent
        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
