"""Anthropic（Claude）后端。

两个和 OpenAI 不一样的地方：

1. system prompt 是**顶层参数**，不是 messages 里的一条。

2. 回复是一串 content blocks，可能含 thinking 块。把历史回传给模型时必须原样带回去，
   所以 Message.raw 里存了原生 blocks，转换时优先用它。自己拼 text 回去会丢信息。

3. prompt 缓存要**自己开**。DeepSeek、百炼是自动缓存前缀，什么都不用做；Anthropic 不打
   cache_control 就一个 token 都不缓存。见 _request_kwargs。
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..core.errors import ContextOverflow
from ..core.messages import LLMResponse, Message, ToolCall, Usage
from ..core.provider import LLMProvider
from .overflow import is_context_overflow


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        context_window: int | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window

    # ------------------------------------------------ 中立格式 -> 厂商格式
    @staticmethod
    def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in (tools or [])
        ]

    @staticmethod
    def convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue                       # system 走顶层参数

            if msg.role == "user":
                out.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                if msg.raw is not None:
                    out.append({"role": "assistant", "content": msg.raw})
                else:
                    blocks: list[dict[str, Any]] = []
                    if msg.content:
                        blocks.append({"type": "text", "text": msg.content})
                    for call in msg.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        })
                    out.append({"role": "assistant", "content": blocks})

            elif msg.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                # 连续的工具结果要合并进同一条 user 消息。
                # 拆开发会让模型以后不敢再并行调用工具。
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    # ------------------------------------------------------------- 调用
    def _request_kwargs(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """拼请求参数。缓存用两个断点（Anthropic 文档推荐给 Agent 循环的组合）：

            系统提示词最后一块  显式 cache_control。渲染顺序是 tools → system → messages，
                                所以这一个断点把工具定义和系统提示词一起缓存；后面的消息
                                怎么变（清理、压缩），这一段都能命中
            请求顶层            cache_control = 自动缓存：断点放在最后一个块上、跟着对话往后挪。
                                每次请求写到末尾，下一次请求往回找（最多 20 个位置）就能读到。
                                我们一步只新增两三个位置（助手的文字 / 工具调用、一组工具结果）

        写摘要的请求是「上一次请求的开头 + 追加一条要求」，往回两三个位置就能找到上一轮写下的缓存。
        TTL 用默认的 5 分钟：一轮里的步子远没那么慢，每次读都会续期。
        写缓存比普通输入贵 25%，读缓存是一折；Agent 每步只多写几百到几千 token，划算。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": self.convert_messages(messages),
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
        return kwargs

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = self._request_kwargs(messages, tools, system, max_tokens)
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if is_context_overflow(str(exc)):
                raise ContextOverflow(f"请求超出了 {self.model} 的上下文窗口：{exc}") from exc
            raise

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_content=resp.content,
            usage=self.convert_usage(resp.usage),
            stop_reason=resp.stop_reason,
        )

    @staticmethod
    def convert_usage(u: Any) -> Usage:
        """Anthropic 的 input_tokens 是**扣掉缓存之后**剩下的部分。

        开了缓存以后，一个 50k 的对话 input_tokens 可能只有几百 —— 只看它会以为
        上下文还很空。完整输入要三块加起来，Usage 里已经拆好了，直接对应。
        两个缓存字段在没开缓存时可能是 None。
        """
        return Usage(
            input=u.input_tokens,
            output=u.output_tokens,
            cache_read=u.cache_read_input_tokens or 0,
            cache_write=u.cache_creation_input_tokens or 0,
        )
