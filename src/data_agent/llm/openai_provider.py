"""OpenAI 兼容后端。

DeepSeek / 通义千问 / Kimi / 智谱 / 本地 vLLM、Ollama 都提供 OpenAI 兼容接口，
换一家只要换 base_url + model，代码完全一样。

和 Anthropic 的三个格式差异：
1. system 是 messages 里的第一条，不是顶层参数。
2. 工具调用的参数是 **JSON 字符串**，不是 dict，要自己 json.loads。
   有些模型会吐出不合法的 JSON，必须兜住 —— 不能让一次解析失败把整个 Agent 搞崩。
3. 思考模型（deepseek-flash、各家的 reasoning 模型）会多返回一个
   `reasoning_content` 字段，而且**要求回传时原样带上**，否则 400：
       The `reasoning_content` in the thinking mode must be passed back to the API.
   所以这里和 Anthropic provider 一样，把原生 message 存进 Message.raw，
   回传时优先用它。

   ⚠️ 这里曾经写着「OpenAI 格式能从中立结构无损还原，不需要存原生内容」——
      那句话是错的。当时没出事纯属运气：调工具时 content 恰好是空的，
      绕过了服务端校验。**别假设中立结构能覆盖所有厂商的私有字段。**
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ..core.messages import LLMResponse, Message, ToolCall, Usage
from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 8192,
        context_window: int | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window

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
                    # 原生 message 原样回传，保住 reasoning_content 这类私有字段。
                    # 自己从中立结构拼回去一定会丢东西。
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
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.convert_messages(messages, system),
        }
        if tools:
            kwargs["tools"] = self.convert_tools(tools)

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # 模型吐了非法 JSON。不要崩 —— 传个特殊标记下去，
                # pydantic 校验会失败，错误信息自然会回到模型那里让它重来。
                arguments = {"__invalid_json__": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            # 存原生 message（含 reasoning_content 等厂商私有字段），回传时原样用它。
            # exclude_none 去掉 refusal / audio 这些没用到的空字段，请求体干净些。
            raw_content=message.model_dump(exclude_none=True),
            usage=self.convert_usage(resp.usage),
            stop_reason=choice.finish_reason,
        )

    @staticmethod
    def convert_usage(u: Any) -> Usage:
        """OpenAI 系的 prompt_tokens **已经包含**缓存命中的部分，要拆出来。

        缓存命中数各家放的位置不一样：
            OpenAI    prompt_tokens_details.cached_tokens
            DeepSeek  prompt_cache_hit_tokens（顶层私有字段，SDK 里是 extra 属性）
        都没有就当 0。缓存写入 OpenAI 系一般不单独报（写缓存不额外收费）。

        有些兼容接口干脆不返回 usage，那就全 0 —— to_message() 不会拿它当锚点。
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
