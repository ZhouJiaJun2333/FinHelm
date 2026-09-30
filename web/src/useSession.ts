// 连上一个会话的事件流：先收快照，之后每个事件改时间线和状态。

import { useCallback, useEffect, useRef, useState } from "react";
import { api, Unauthorized } from "./api";
import { reduce, withId } from "./timeline";
import type { AgentState, ApprovalRequest, Info, Interrupted, Item, ServerMessage, Table } from "./types";

export type SessionView = {
  ready: boolean;
  items: Item[];
  state: AgentState | null;
  interrupted: Interrupted;
  results: Table[];
  approvals: ApprovalRequest[];
  info: Info | null;
  busy: boolean;
  error: string;
};

export const LOGGED_OUT = "finhelm:logged-out";

const EMPTY: SessionView = {
  ready: false, items: [], state: null, interrupted: null, results: [], approvals: [], info: null, busy: false, error: "",
};

function apply(view: SessionView, msg: ServerMessage): SessionView {
  const state = msg.state ?? view.state;
  const d = msg.data;
  switch (msg.type) {
    case "snapshot": {
      const items: Item[] = d.items.map((it: Omit<Item, "id">) => withId(it));
      // 连上时正好在输出：半句话在状态里
      const s = msg.state!;
      if (d.busy && (s.streaming_text || s.streaming_thinking)) {
        items.push(withId({ kind: "assistant" as const, text: s.streaming_text, thinking: s.streaming_thinking,
                            streaming: true, step: s.step }));
      }
      return { ready: true, items, state, interrupted: d.interrupted, results: d.results, approvals: d.approvals,
               info: d.info, busy: d.busy, error: "" };
    }
    case "results":
      return { ...view, state, results: d };
    case "approval":
      return { ...view, state, approvals: [...view.approvals, d] };
    case "approval_done":
      return { ...view, state, approvals: view.approvals.filter((a) => a.id !== d.id) };
    case "idle":
      return { ...view, state, busy: false, interrupted: d.interrupted };
    case "TurnStarted":
    case "TurnResumed":
      return { ...view, state, busy: true, interrupted: null, items: reduce(view.items, msg) };
  }
  return { ...view, state, items: reduce(view.items, msg) };
}

export function useSession(id: string | null) {
  const [view, setView] = useState<SessionView>(EMPTY);
  const ready = useRef<Promise<void> | null>(null);

  useEffect(() => {
    setView(EMPTY);
    if (!id) return;
    const source = new EventSource(api.events(id));
    let resolve: () => void;
    ready.current = new Promise((r) => (resolve = r));
    source.onmessage = (e) => {
      const msg: ServerMessage = JSON.parse(e.data);
      setView((v) => apply({ ...v, error: "" }, msg));
      if (msg.type === "snapshot") resolve();
    };
    source.onerror = () => {
      // 服务器返回错误（会话被删了）时浏览器不会重连
      const closed = source.readyState === EventSource.CLOSED;
      setView((v) => ({ ...v, error: closed ? "打不开这个会话" : "正在重连…" }));
      // EventSource 看不到状态码：登录过期的话让 App 回登录页
      api.me().then((me) => { if (me.auth && !me.user) window.dispatchEvent(new Event(LOGGED_OUT)); }).catch(() => {});
    };
    return () => source.close();
  }, [id]);

  // 发请求之前等快照到了，免得漏掉最开始的事件
  const whenReady = useCallback(() => ready.current ?? Promise.resolve(), []);

  const run = useCallback(async (fn: () => Promise<unknown>) => {
    setView((v) => ({ ...v, busy: true, error: "" }));
    try {
      await fn();
    } catch (e) {
      if (e instanceof Unauthorized) {
        window.dispatchEvent(new Event(LOGGED_OUT));
        return;
      }
      setView((v) => ({ ...v, busy: false, items: reduce(v.items, { type: "error", data: { message: String((e as Error).message) } }) }));
    }
  }, []);

  return { view, whenReady, run };
}
