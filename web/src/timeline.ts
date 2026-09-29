// 时间线：打开会话时用服务器给的历史条目建起来，之后每来一个事件就改一下。
// 和后端 core/state.py 的 reduce 是一个思路：纯函数，(条目, 事件) -> 新条目。

import type { Item, ServerMessage } from "./types";

let nextId = 1;
export const withId = <T extends object>(item: T) => ({ ...item, id: nextId++ });

type Tool = Extract<Item, { kind: "tool" }>;
type Assistant = Extract<Item, { kind: "assistant" }>;

const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

function notice(items: Item[], text: string, level: "info" | "warn" | "error" = "info"): Item[] {
  return [...items, withId({ kind: "notice" as const, text, level })];
}

function patchTool(items: Item[], callId: string, name: string, patch: Partial<Tool>): Item[] {
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i];
    if (it.kind === "tool" && it.call_id === callId) {
      const next = items.slice();
      next[i] = { ...it, ...patch };
      return next;
    }
  }
  // 找不到（服务器重启后接着答的那次提问）就补一条
  return [...items, withId({ kind: "tool" as const, call_id: callId, name, arguments: {}, status: "done" as const,
                              content: "", details: null, elapsed_ms: 0, ...patch })];
}

// 最后一条还在流式输出的回复（属于这一步）
function streamingIndex(items: Item[], step: number): number {
  const i = items.length - 1;
  const last = items[i];
  return last?.kind === "assistant" && last.streaming && last.step === step ? i : -1;
}

export function reduce(items: Item[], msg: ServerMessage): Item[] {
  const d = msg.data;
  switch (msg.type) {
    case "TurnStarted":
      return [...items, withId({ kind: "user" as const, text: d.question })];

    case "TextDelta": {
      const i = streamingIndex(items, d.step);
      const field = d.thinking ? "thinking" : "text";
      if (i < 0) {
        const fresh: Omit<Assistant, "id"> = { kind: "assistant", text: "", thinking: "", streaming: true, step: d.step };
        return [...items, withId({ ...fresh, [field]: d.text })];
      }
      const next = items.slice();
      const it = next[i] as Assistant;
      next[i] = { ...it, [field]: it[field] + d.text };
      return next;
    }

    case "LLMResponded": {
      const i = streamingIndex(items, d.step);
      if (i >= 0) {
        const next = items.slice();
        next[i] = { ...(next[i] as Assistant), text: d.text, streaming: false };
        return next;
      }
      if (!d.text) return items;
      return [...items, withId({ kind: "assistant" as const, text: d.text, thinking: "", streaming: false, step: d.step })];
    }

    case "ToolStarted":
      return [...items, withId({ kind: "tool" as const, call_id: d.call_id, name: d.name, arguments: d.arguments,
                                 status: "running" as const, content: "", details: null, elapsed_ms: 0 })];

    case "ToolFinished":
      return patchTool(items, d.call_id, d.name, {
        status: d.is_error ? "error" : "done", content: d.content, details: d.details, elapsed_ms: d.elapsed_ms });

    case "ToolDenied":
      return patchTool(items, d.call_id, d.name, { status: "denied", content: d.reason });

    case "UserAsked":
      return patchTool(items, d.call_id, d.name, { status: "asking" });

    case "TurnResumed": {
      const answering = items.some((it) => it.kind === "tool" && it.status === "asking");
      if (answering) return items;                        // 回答会作为那次提问的结果出现
      return notice(items, `从第 ${d.steps + 1} 步接着跑${d.message ? `：${d.message}` : ""}`);
    }

    case "TurnContinued":
      return notice(items, `判定还没做完，接着做：${d.nudge}`);
    case "ToolCallRepeated":
      return notice(items, `同样的参数第 ${d.count} 次调用 ${d.name}，已提醒它换个思路`, "warn");
    case "StepLimitReached":
      return notice(items, `用完了 ${d.max_steps} 步` + (d.wrapped_up ? "，已根据现有结果收尾" : `，收尾没成（${d.failure}）`), "warn");
    case "ContextEdited":
      return notice(items, `${d.description}：上下文 ${k(d.tokens_before)} → ${k(d.tokens_after)}`);
    case "ContextEditFailed":
      return notice(items, `压缩没做成：${d.reason}`, "warn");
    case "AutoCompactionPaused":
      return notice(items, `自动压缩连续失败 ${d.failures} 次，本会话不再自动压缩`, "warn");
    case "ContextOverflowed":
      return notice(items, "超出了模型的上下文窗口，强制整理后重试", "warn");

    case "TurnEnded": {
      let next = items.map((it) =>
        it.kind === "assistant" && it.streaming ? { ...it, streaming: false }
        : it.kind === "tool" && it.status === "running" ? { ...it, status: "error" as const } : it);
      // 没流式出来的回答（步数用完的兜底话）补上
      if (d.answer && !next.some((it) => it.kind === "assistant" && it.text === d.answer)) {
        next = [...next, withId({ kind: "assistant" as const, text: d.answer, thinking: "", streaming: false })];
      }
      return next;
    }

    case "ConversationReset":
      return [];
    case "notice":
      return notice(items, d.text);
    case "error":
      return notice(items, d.message, "error");
  }
  return items;
}
