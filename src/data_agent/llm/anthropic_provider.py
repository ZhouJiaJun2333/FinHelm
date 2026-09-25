"""Anthropic（Claude）后端。

和 OpenAI 的差异：system 是顶层参数；回复是 content blocks（可能有 thinking），回传时原样带上；
prompt 缓存要自己打 cache_control（DeepSeek、百炼是自动前缀缓存）。
图片原生支持：工具结果里的图直接放进 tool_result 的 content。
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..core.errors import ContextOverflow
from ..core.messages import Image, LLMResponse, Message, ToolCall, Usage
from ..core.provider import LLMProvider
from .overflow import is_context_overflow


class AnthropicProvider(LLMProvider):
    vision = True

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
                out.append({"role": "user", "content": _with_images(msg)})

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
                    "content": _with_images(msg),
                }
                if msg.is_error:
                    block["is_error"] = True
                # 连续的工具结果合并进同一条 user 消息，拆开发会让模型不敢再并行调工具
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
        """两个缓存断点：系统提示词末尾（连同工具定义一起缓存，后面的消息怎么变都能命中），
        加顶层自动缓存（断点跟着对话末尾往后挪）。写摘要的请求往回两三个位置就能读到上一轮的缓存。
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
            # 转成纯 dict（Message.raw 要能存盘读回）
            raw_content=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
            usage=self.convert_usage(resp.usage),
            stop_reason=resp.stop_reason,
        )

    @staticmethod
    def convert_usage(u: Any) -> Usage:
        """input_tokens 已经扣掉缓存部分，三块加起来才是完整输入。"""
        return Usage(
            input=u.input_tokens,
            output=u.output_tokens,
            cache_read=u.cache_read_input_tokens or 0,
            cache_write=u.cache_creation_input_tokens or 0,
        )


def _with_images(msg: Message) -> str | list[dict[str, Any]]:
    """没图就还是字符串：请求体和以前一模一样，缓存不受影响。"""
    if not msg.images:
        return msg.content
    return [*([{"type": "text", "text": msg.content}] if msg.content else []),
            *(_image_block(i) for i in msg.images)]


def _image_block(image: Image) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": image.media_type, "data": image.data}}
