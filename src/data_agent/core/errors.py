"""Agent 运行时的异常。

单独一个文件，是为了让调用方能精确地 catch —— `except AgentError` 兜住
我们自己抛的，不会误伤 psycopg 或 SDK 的异常。

⚠️ 注意这些和「工具执行失败」是两回事：
    工具失败    → ToolOutput(ok=False)，喂回模型让它自己修，**不抛异常**
    这里的异常  → 模型这一轮本身就不可信（被截断/被拒绝），继续下去没意义
"""

from __future__ import annotations


class AgentError(RuntimeError):
    """本项目 Agent 运行时抛出的异常的基类。"""


class OutputTruncated(AgentError):
    """模型输出被 max_tokens 截断 —— 话没说完，不是「完成」。

    这是最阴险的一类失败：没有工具调用、没有报错，看起来就像模型答完了，
    实际返回的是半句话。必须靠 stop_reason 才能识别。
    """


class ModelRefused(AgentError):
    """模型拒绝回答（安全分类器 / 内容过滤）。"""


class UnexpectedStopReason(AgentError):
    """遇到了我们没处理过的 stop_reason。

    宁可炸掉也不要静默当成「完成」—— 静默的错误比崩溃难查一百倍。
    """
