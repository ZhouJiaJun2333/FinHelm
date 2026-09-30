"""OpenAI 兼容后端（DeepSeek、通义、Kimi、智谱、vLLM…，换 base_url + model 就行）。

和 Anthropic 的差异：system 是 messages 第一条；工具参数是 JSON 字符串（可能不合法，要兜住）；
思考模型的 reasoning_content 回传时必须原样带上，否则 400 —— 所以原生 message 存进 Message.raw。
tool 消息只能放文字：工具结果里的图片挪到这批工具结果后面的一条 user 消息里（学 pi）。
"""

from __future__ import annotations

import json
from typing import Any

from openai import BadRequestError, OpenAI

from ..core.errors import ContextOverflow
from ..core.messages import INVALID_JSON_ARGS, Image, LLMResponse, Message, ToolCall, Usage
from ..core.provider import MAX_RETRIES, LLMProvider, OnDelta
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
        vision: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=MAX_RETRIES)
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.native_thinking = native_thinking
        self.vision = vision

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
    def convert_messages(messages: list[Message], system: str | None,
                         vision: bool = True) -> list[dict[str, Any]]:
        """vision=False：图片换成一句说明（换了个不能看图的模型接着聊时，历史里可能有图）。"""
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        # 一批工具结果里的图片：tool 消息之间不能插别的消息，等这批结束再发
        pending: list[Image] = []

        for msg in messages:
            # user 不在这里发图：紧跟在工具结果后面的 user 要和这批图并成一条（见下）
            if msg.role not in ("tool", "user") and pending:
                out.append(_tool_images_message(pending))
                pending = []

            if msg.role == "system":
                out.append({"role": "system", "content": msg.content})

            elif msg.role == "user":
                if msg.images and vision:
                    content: str | list[dict[str, Any]] = [
                        *([{"type": "text", "text": msg.content}] if msg.content else []),
                        *(_image_part(i) for i in msg.images),
                    ]
                else:
                    content = _with_omitted(msg)
                if pending:
                    # 这批工具结果的图和紧跟其后的 user（比如步数用完时的收尾提示）是同一个 user 回合，
                    # 并成一条。和 Anthropic 那边一样只并这一种：两条文字 user 挨着是历史里有残留，照原样发
                    item = _tool_images_message(pending)
                    item["content"] += _parts(content)
                    out.append(item)
                    pending = []
                else:
                    out.append({"role": "user", "content": content})

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
                content = msg.content
                if msg.images and vision:
                    content += f"\n\n{TOOL_IMAGES_NOTE}"
                    pending += msg.images
                elif msg.images:
                    content = _with_omitted(msg)
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                })
        if pending:
            out.append(_tool_images_message(pending))
        return out

    # ------------------------------------------------------------- 调用
    def _kwargs(self, messages, tools, system, max_tokens) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": self.convert_messages(messages, system, vision=self.vision),
        }
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
        return kwargs

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if is_context_overflow(str(exc)):
                raise ContextOverflow(f"请求超出了 {self.model} 的上下文窗口：{exc}") from exc
            raise

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        resp = self._create(**self._kwargs(messages, tools, system, max_tokens))
        choice = resp.choices[0]
        # exclude_none：去掉 refusal / audio 这类空字段
        return self._response(choice.message.model_dump(exclude_none=True), resp.usage, choice.finish_reason)

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        *,
        on_delta: OnDelta,
    ) -> LLMResponse:
        """把分块拼回和 chat 一样的原生 message：reasoning_content 回传时要带上，工具参数是一段段 JSON 拼起来的。"""
        chunks = self._create(**self._kwargs(messages, tools, system, max_tokens),
                              stream=True, stream_options={"include_usage": True})
        text: list[str] = []
        thinking: list[str] | None = None        # None = 这个模型不回 reasoning_content
        calls: dict[int, dict[str, str]] = {}
        usage = finish = None
        for chunk in chunks:
            # DeepSeek 把用量放在带 finish_reason 的那块，OpenAI 单独一块（choices 为空）
            usage = chunk.usage or usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            finish = choice.finish_reason or finish
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is not None and thinking is None:
                thinking = []
            if reasoning:
                thinking.append(reasoning)
                on_delta(reasoning, True)
            if delta.content:
                text.append(delta.content)
                on_delta(delta.content, False)
            for part in delta.tool_calls or []:
                call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = part.id or call["id"]
                if part.function is not None:
                    call["name"] += part.function.name or ""
                    call["arguments"] += part.function.arguments or ""

        raw: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if thinking is not None:
            raw["reasoning_content"] = "".join(thinking)
        if calls:
            raw["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for _, c in sorted(calls.items())]
        return self._response(raw, usage, finish)

    def _response(self, raw: dict[str, Any], usage: Any, finish_reason: str | None) -> LLMResponse:
        tool_calls: list[ToolCall] = []
        for call in raw.get("tool_calls") or []:
            fn = call["function"]
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                # 非法 JSON 不崩：交给工具层报错，错误会回到模型那里
                arguments = {INVALID_JSON_ARGS: fn.get("arguments")}
            tool_calls.append(ToolCall(id=call["id"], name=fn["name"], arguments=arguments))

        return LLMResponse(
            text=(raw.get("content") or "").strip(),
            tool_calls=tool_calls,
            raw_content=raw,
            usage=self.convert_usage(usage),
            stop_reason=finish_reason,
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


# ------------------------------------------------------------------ 图片
TOOL_IMAGES_NOTE = "（图片见下一条消息）"
TOOL_IMAGES_HEADER = "[上面工具结果里的图片，按顺序：]"
IMAGE_OMITTED = "[这里有 {n} 张图片，当前模型不能看图，已省略]"


def _image_part(image: Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image.data_url}}


def _tool_images_message(images: list[Image]) -> dict[str, Any]:
    """不是用户说的话，只是把图片带过去。开头一句说明来源，免得模型当成用户新发的图。"""
    return {"role": "user", "content": [{"type": "text", "text": TOOL_IMAGES_HEADER},
                                        *(_image_part(i) for i in images)]}


def _parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content}] if content else []


def _with_omitted(msg: Message) -> str:
    if not msg.images:
        return msg.content
    note = IMAGE_OMITTED.format(n=len(msg.images))
    return f"{msg.content}\n\n{note}" if msg.content else note
