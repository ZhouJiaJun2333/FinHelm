import { useState } from "react";
import { AlertTriangle, ChevronRight, CircleHelp, Info, Play, Plug, ShieldCheck } from "lucide-react";
import type { ApprovalRequest, Interrupted, Item, Table } from "../types";
import { Markdown } from "./Markdown";
import { ToolCard } from "./ToolCard";

type Assistant = Extract<Item, { kind: "assistant" }>;
type Tool = Extract<Item, { kind: "tool" }>;

export function UserMessage({ text }: { text: string }) {
  // 上传文件的前缀单独显示成附件行
  const m = text.match(/^\[用户上传了文件，在 inputs\/ 下：(.+?)\]\n\n([\s\S]*)$/);
  return (
    <div className="user-row">
      <div className="user-msg">
        {m && <div className="attach-line">📎 {m[1]}</div>}
        {m ? m[2] : text}
      </div>
    </div>
  );
}

export function AssistantMessage({ item, results, onOpenResult }:
    { item: Assistant; results: Table[]; onOpenResult: (ref: string) => void }) {
  const [showThinking, setShowThinking] = useState(false);
  const thinkingLive = item.streaming && !item.text;
  return (
    <div className="assistant">
      {item.thinking && (
        <div className={`thinking${showThinking ? " open" : ""}`}>
          <button className="thinking-head" onClick={() => setShowThinking(!showThinking)}>
            <span className={thinkingLive ? "shimmer" : ""}>{thinkingLive ? "思考中…" : "思考过程"}</span>
            <ChevronRight size={14} className="chev" />
          </button>
          {showThinking && <div className="thinking-body">{item.thinking}</div>}
        </div>
      )}
      {item.text && <Markdown text={item.text} results={results} onOpenResult={onOpenResult} />}
      {item.streaming && item.text && <span className="caret" />}
    </div>
  );
}

// ask_user 那次调用：问题 + 选项；答了的显示回答
export function AskCard({ tool, pending, onAnswer }:
    { tool: Tool; pending: boolean; onAnswer: (answer: string) => void }) {
  const question = String(tool.arguments.question ?? "");
  const options = (tool.arguments.options as string[] | undefined) ?? [];
  const answer = tool.content.replace(/^用户回答：/, "");
  return (
    <div className={`ask${pending ? " pending" : ""}`}>
      <div className="ask-q"><CircleHelp size={16} /> {question}</div>
      {pending ? (
        <>
          {options.length > 0 && (
            <div className="ask-options">
              {options.map((o, i) => (
                <button key={o} onClick={() => onAnswer(o)}><span className="kbd">{i + 1}</span>{o}</button>
              ))}
            </div>
          )}
          <div className="muted small">也可以在下面直接写你的回答；不想答就点「让它自己判断」</div>
          <button className="link" onClick={() => onAnswer("")}>让它自己判断</button>
        </>
      ) : tool.status === "done" ? (
        <div className="ask-a">{answer.startsWith("[用户没有回答") ? "（没有回答，让它自己判断）" : answer}</div>
      ) : null}
    </div>
  );
}

export function Notice({ text, level }: { text: string; level: "info" | "warn" | "error" }) {
  const Icon = level === "info" ? Info : AlertTriangle;
  return <div className={`notice ${level}`}><Icon size={13} /> <span>{text}</span></div>;
}

export function InterruptedCard({ turn, onContinue }: { turn: NonNullable<Interrupted>; onContinue: () => void }) {
  const stopped = turn.reason.startsWith("Stopped");
  return (
    <div className="interrupted">
      <div>
        <strong>{stopped ? "已停止" : "这一轮没跑完"}</strong>
        <span className="muted">
          {turn.steps ? ` · 做完了 ${turn.steps} 步` : ""}
          {!stopped && turn.reason ? ` · ${turn.reason}` : ""}
        </span>
        <div className="muted small">进度保留着。继续会从断的地方接着做，也可以在输入框里写一句调整方向后点继续；直接问新问题就放弃这一轮。</div>
      </div>
      <button className="btn primary" onClick={onContinue}><Play size={14} /> 继续</button>
    </div>
  );
}

export function ApprovalCard({ req, onDecide }:
    { req: ApprovalRequest; onDecide: (d: "once" | "session" | "deny") => void }) {
  return (
    <div className="approval">
      <div className="approval-head"><Plug size={16} /> 要调用外部 MCP 服务器 <b>{req.server}</b> 的工具 <code>{req.tool}</code></div>
      <pre className="code args">{JSON.stringify(req.arguments, null, 2)}</pre>
      <div className="approval-actions">
        <button className="btn primary" onClick={() => onDecide("once")}><ShieldCheck size={14} /> 这次允许</button>
        <button className="btn" onClick={() => onDecide("session")}>本会话都允许</button>
        <button className="btn danger" onClick={() => onDecide("deny")}>拒绝</button>
      </div>
    </div>
  );
}

export function ItemView({ item, results, pendingAsk, onOpenResult, onAnswer }: {
  item: Item; results: Table[]; pendingAsk: string | null;
  onOpenResult: (ref: string) => void; onAnswer: (a: string) => void;
}) {
  switch (item.kind) {
    case "user":
      return <UserMessage text={item.text} />;
    case "assistant":
      return <AssistantMessage item={item} results={results} onOpenResult={onOpenResult} />;
    case "tool":
      if (item.name === "ask_user") return <AskCard tool={item} pending={item.call_id === pendingAsk} onAnswer={onAnswer} />;
      return <ToolCard tool={item} onOpenResult={onOpenResult} />;
    case "notice":
      return <Notice text={item.text} level={item.level} />;
  }
}
