import { MessageSquare, PanelLeftClose, SquarePen } from "lucide-react";
import type { SessionSummary } from "../types";
import { Logo } from "./Logo";

type Props = {
  sessions: SessionSummary[];
  current: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onCollapse: () => void;
};

function group(ts: number): string {
  const days = (Date.now() / 1000 - ts) / 86400;
  if (days < 1) return "今天";
  if (days < 7) return "最近 7 天";
  if (days < 30) return "最近 30 天";
  return "更早";
}

export function Sidebar({ sessions, current, onSelect, onNew, onCollapse }: Props) {
  const groups: [string, SessionSummary[]][] = [];
  for (const s of sessions) {
    const g = group(Math.min(s.updated, Date.now() / 1000));
    if (groups.at(-1)?.[0] !== g) groups.push([g, []]);
    groups.at(-1)![1].push(s);
  }
  return (
    <aside className="sidebar">
      <div className="side-top">
        <div className="brand"><Logo size={20} /> FinHelm</div>
        <button className="icon-btn" title="收起侧栏" onClick={onCollapse}><PanelLeftClose size={17} /></button>
      </div>
      <button className="new-chat" onClick={onNew}><SquarePen size={16} /> 新对话</button>
      <nav className="session-list">
        {groups.map(([name, list]) => (
          <div key={name}>
            <div className="group-name">{name}</div>
            {list.map((s) => (
              <button key={s.id} className={`session${s.id === current ? " active" : ""}`} onClick={() => onSelect(s.id)}
                title={s.title || "新对话"}>
                <MessageSquare size={14} />
                <span>{s.title || "新对话"}</span>
              </button>
            ))}
          </div>
        ))}
        {sessions.length === 0 && <div className="muted small pad">还没有对话</div>}
      </nav>
    </aside>
  );
}
