"""OpenAI 兼容后端（DeepSeek、通义、Kimi、智谱、vLLM…，换 base_url + model 就行）。

和 Anthropic 的差异：system 是 messages 第一条；工具参数是 JSON 字符串（可能不合法，要兜住）；
思考模型的 reasoning_content 回传时必须原样带上，否则 400 —— 所以原生 message 存进 Message.raw。
"""

from __future__ import annotations

import json
from typing import Any

from openai import BadRequestError, OpenAI

from ..core.errors import ContextOverflow
from ..core.messages import LLMResponse, Message, ToolCall, Usage
from ..core.provider import LLMProvider
from .overflow import is_context_overflow


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 8192,
        context_window: int | None = None,
        native_thinking: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.native_thinking = native_thinking

    # ------------------------------------------------ 中立格式 -> 厂商格式
    @staticmethod
    def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in (tools or [])
        ]

    @staticmethod
    def convert_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system":
                out.append({"role": "system", "content": msg.content})

            elif msg.role == "user":
                out.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                if msg.raw is not None:
                    # 原样回传，保住 reasoning_content 这类私有字段
                    out.append(dict(msg.raw))
                else:
                    item: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content or None,
                    }
                    if msg.tool_calls:
                        item["tool_calls"] = [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                                },
                            }
                            for c in msg.tool_calls
                        ]
                    out.append(item)

            elif msg.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
        return out

    # ------------------------------------------------------------- 调用
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": self.convert_messages(messages, system),
        }
        if tools:
            kwargs["tools"] = self.convert_tools(tools)

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if is_context_overflow(str(exc)):
                raise ContextOverflow(f"请求超出了 {self.model} 的上下文窗口：{exc}") from exc
            raise
        choice = resp.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # 非法 JSON 不崩：交给 pydantic 校验失败，错误会回到模型那里
                arguments = {"__invalid_json__": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            # exclude_none：去掉 refusal / audio 这类空字段
            raw_content=message.model_dump(exclude_none=True),
            usage=self.convert_usage(resp.usage),
            stop_reason=choice.finish_reason,
        )

    @staticmethod
    def convert_usage(u: Any) -> Usage:
        """prompt_tokens 已含缓存命中，要拆出来。命中数 OpenAI 放在 prompt_tokens_details.cached_tokens，
        DeepSeek 放在顶层的 prompt_cache_hit_tokens。没有 usage 就全 0（不会被当成锚点）。
        """
        if u is None:
            return Usage()

        details = getattr(u, "prompt_tokens_details", None)
        cache_read = (
            getattr(details, "cached_tokens", None)
            or getattr(u, "prompt_cache_hit_tokens", None)
            or 0
        )
        cache_write = getattr(details, "cache_write_tokens", None) or 0
        return Usage(
            input=u.prompt_tokens - cache_read - cache_write,
            output=u.completion_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        )
