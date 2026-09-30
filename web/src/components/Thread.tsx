import { useEffect, useMemo, useRef } from "react";
import { toBlocks } from "../blocks";
import type { ApprovalRequest, Interrupted, Item, Table } from "../types";
import { Activity } from "./Activity";
import { AskCard } from "./Ask";
import { Logo } from "./Logo";
import { Markdown } from "./Markdown";
import { useFileUrl } from "../sessionContext";

type Props = {
  items: Item[];
  results: Table[];
  busy: boolean;
  interrupted: Interrupted;
  approvals: ApprovalRequest[];
  onOpenResult: (ref: string) => void;
  onOpenFile: (url: string) => void;
  onAnswer: (answer: string) => void;
  onContinue: () => void;
  onApprove: (id: string, decision: "once" | "session" | "deny") => void;
};

// 中断原因（异常类名）→ 一个词
const REASONS: [RegExp, string][] = [
  [/Timeout/i, "请求超时"], [/Connection/i, "连接失败"], [/RateLimit|429/i, "被限流"],
  [/OutputTruncated/, "输出被截断"], [/ContextOverflow|CompactionFailed/, "上下文超长"],
];

function shortReason(reason: string) {
  return REASONS.find(([re]) => re.test(reason))?.[1] ?? "出错了";
}

export function Thread(p: Props) {
  const blocks = useMemo(() => toBlocks(p.items), [p.items]);
  const pendingAsk = !p.busy ? p.interrupted?.pending?.call_id ?? null : null;
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // 贴着底部时新内容来了自动往下滚；往上翻了、或者只是点开了某一行，都不动
  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [p.items, p.busy, p.approvals, p.interrupted]);

  const last = blocks.at(-1);
  return (
    <div className="scroll" ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}>
      <div className="thread">
        {blocks.map((b) => {
          switch (b.kind) {
            case "user":
              return <UserMessage key={b.id} text={b.text} onOpenFile={p.onOpenFile} />;
            case "text":
              return <Markdown key={b.id} text={b.item.text} results={p.results} onOpenResult={p.onOpenResult} />;
            case "activity":
              return <Activity key={b.id} steps={b.steps} files={b.files} live={p.busy && b === last}
                               onOpenFile={p.onOpenFile} onOpenResult={p.onOpenResult} />;
            case "ask":
              return <AskCard key={b.id} tool={b.tool} pending={b.tool.call_id === pendingAsk} onAnswer={p.onAnswer} />;
            case "notice":
              return <div key={b.id} className={`notice ${b.level}`}>{b.text}</div>;
          }
        })}

        {p.approvals.map((a) => (
          <div key={a.id} className="approval">
            <div>允许调用 <b>{a.server}</b> 的 <code>{a.tool}</code>？</div>
            <pre className="code">{JSON.stringify(a.arguments, null, 2)}</pre>
            <div className="approval-actions">
              <button className="btn dark" onClick={() => p.onApprove(a.id, "once")}>允许一次</button>
              <button className="btn" onClick={() => p.onApprove(a.id, "session")}>本会话允许</button>
              <button className="btn" onClick={() => p.onApprove(a.id, "deny")}>拒绝</button>
            </div>
          </div>
        ))}

        {!p.busy && p.interrupted && !p.interrupted.pending && (
          <div className="interrupted">
            {p.interrupted.reason.startsWith("Stopped") ? "已停止" : `已中断 · ${shortReason(p.interrupted.reason)}`}
            <button className="text-btn accent" onClick={p.onContinue}>继续</button>
          </div>
        )}

        {p.busy && <div className="spark"><Logo size={18} /></div>}
      </div>
    </div>
  );
}

function UserMessage({ text, onOpenFile }: { text: string; onOpenFile: (url: string) => void }) {
  const fileUrl = useFileUrl();
  // 上传文件的前缀单独显示成附件行，点文件名在右侧面板看。格式见 app.py 的 with_uploads
  const m = text.match(/^\[用户上传了文件，在 inputs\/ 下：(.+?)\]\n\n([\s\S]*)$/);
  const files = m ? m[1].split("、").map((label) => ({ label, name: label.replace(/（[^）]*）$/, "") })) : [];
  return (
    <div className="user-row">
      <div className="user-msg">
        {m && (
          <div className="attach-line">
            {files.map((f, i) => (
              <button key={i} className="attach-file"
                onClick={() => onOpenFile(fileUrl(`inputs/${encodeURIComponent(f.name)}`)!)}>
                {f.label}
              </button>
            ))}
          </div>
        )}
        {m ? m[2] : text}
      </div>
    </div>
  );
}
