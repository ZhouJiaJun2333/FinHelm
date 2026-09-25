"""Agent 运行时的异常：这一轮本身不可信，继续没有意义。

和工具失败不同：工具失败是 is_error 的结果，喂回模型让它自己修。
"""

from __future__ import annotations


class AgentError(RuntimeError):
    pass


class OutputTruncated(AgentError):
    """被 max_tokens 截断：没有工具调用、没有报错，看起来像答完了，其实是半句话。"""


class ModelRefused(AgentError):
    """模型拒绝回答（安全分类器 / 内容过滤）。"""


class UnexpectedStopReason(AgentError):
    """没见过的 stop_reason。宁可炸掉，也不静默当成「完成」。"""


class CompactionFailed(AgentError):
    """写摘要失败（被截断、返回空…）。"""


class ContextOverflow(AgentError):
    """请求超出上下文窗口（provider 把各家报错翻译成它）。Agent 会强制整理一次再重试。"""
