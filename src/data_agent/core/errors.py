"""Agent 运行时的异常。

单独一个文件，是为了让调用方能精确地 catch —— `except AgentError` 兜住
我们自己抛的，不会误伤 psycopg 或 SDK 的异常。

⚠️ 注意这些和「工具执行失败」是两回事：
    工具失败    → ToolOutput(is_error=True)，喂回模型让它自己修，不中断 Agent
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


class CompactionFailed(AgentError):
    """摘要压缩没做成（写摘要的那次调用被截断、返回空…）。

    不静默跳过：跳过的话每一步都会重新触发、重新花钱，而且上下文一直超标。
    抛出来，这一轮按事务语义回滚，用户能看到原因。
    """


class ContextOverflow(AgentError):
    """请求超出了模型的上下文窗口，API 拒收（各家报错写法不同，provider 统一翻译成它）。

    我们的估算说没超、API 说超了 —— 估算偏小，或者配置的窗口比实际大。
    Agent 收到它会强制整理一次（清理 + 压缩）再重试，只试一次。
    """
