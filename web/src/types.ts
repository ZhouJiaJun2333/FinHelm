// 和后端 web/serialize.py 对应的形状

export type Usage = { input: number; output: number; cache_read: number; cache_write: number };

export type ToolRun = {
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "running" | "done" | "error" | "denied" | "asking";
  elapsed_ms: number;
};

// core/state.py 的 AgentState
export type AgentState = {
  status: "idle" | "thinking" | "tools" | "asking";
  running: boolean;
  question: string;
  step: number;
  streaming_text: string;
  streaming_thinking: string;
  tools: ToolRun[];
  asking: { question: string; options: string[]; call_id: string } | null;
  answer: string;
  error: string;
  turn_usage: Usage;
  context_tokens: number;
  context_window: number | null;
};

export type Table = {
  ref: string;
  title: string;
  source: string;
  sql: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
};

export type Details =
  | ({ kind: "table" } & Table)
  | { kind: "execution"; output: string; value: string | null; error: string | null; figures: string[] };

export type Item =
  | { kind: "user"; id: number; text: string }
  | { kind: "assistant"; id: number; text: string; thinking: string; streaming: boolean; step?: number }
  | {
      kind: "tool";
      id: number;
      call_id: string;
      name: string;
      arguments: Record<string, unknown>;
      status: ToolRun["status"];
      content: string;
      details: Details | null;
      elapsed_ms: number;
    }
  | { kind: "notice"; id: number; text: string; level: "info" | "warn" | "error" };

export type Interrupted = {
  reason: string;
  steps: number;
  question: string;
  pending: { call_id: string; question: string; options: string[] } | null;
} | null;

export type ApprovalRequest = { id: string; server: string; tool: string; arguments: Record<string, unknown> };

export type Info = {
  model: string;
  context_window: number | null;
  tools: string[];
  skills: string[];
  mcp: string[];
  sandbox: boolean;
  database: boolean;
};

export type SessionSummary = { id: string; title: string; updated: number };

// 服务器推来的一条 SSE 消息
export type ServerMessage = { type: string; data: any; state?: AgentState };
