"""故障注入：验证检查点用。真实的限流、断网可遇不可求，dev 只有 10 题，自己造。

    python -m evals.run --cases dabstep_dev --inject-errors 0.1
"""

from __future__ import annotations

import random
import zlib

from data_agent.core.provider import LLMProvider


class InjectedFault(RuntimeError):
    """评测故意注入的 API 错误。"""


class FlakyProvider(LLMProvider):
    """包一层真模型：每次请求以 rate 的概率在发出去之前抛错（不花钱），模拟 429、断网。

    seed 按题目和第几次算：同一题同一次，哪一步出错每回都一样，重跑可以复现。
    """

    def __init__(self, inner: LLMProvider, rate: float, seed: str) -> None:
        self.inner = inner
        self.rate = rate
        self.random = random.Random(zlib.crc32(seed.encode()))
        self.injected = 0
        self.model = inner.model
        self.context_window = inner.context_window
        self.native_thinking = inner.native_thinking
        self.vision = inner.vision

    def chat(self, messages, tools=None, system=None, max_tokens=None):
        if self.random.random() < self.rate:
            self.injected += 1
            raise InjectedFault("注入的 API 错误（模拟 429 限流）")
        return self.inner.chat(messages, tools=tools, system=system, max_tokens=max_tokens)
