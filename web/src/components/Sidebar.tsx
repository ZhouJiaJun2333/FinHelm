import { useState } from "react";
import { MoreHorizontal, PanelLeftClose, SquarePen } from "lucide-react";
import type { SessionSummary, Trashed } from "../types";
import { Logo } from "./Logo";
import { Menu } from "./Menu";

type Props = {
  project: string;
  user: string | null;
  sessions: SessionSummary[];
  trash: Trashed[];
  current: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onCollapse: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onRestore: (id: string) => void;
  onShowTrash: () => void;
  onLogout: () => void;
};

export function Sidebar(p: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [trashOpen, setTrashOpen] = useState(false);

  return (
    <aside className="sidebar">
      <div className="side-top">
        <div className="brand"><Logo size={18} /> FinHelm</div>
        <button className="icon-btn" title="收起侧栏" onClick={p.onCollapse}><PanelLeftClose size={16} /></button>
      </div>
      <button className="side-item" onClick={p.onNew}><SquarePen size={15} /> 新对话</button>

      <div className="side-section">{p.project}</div>
      <nav className="session-list">
        {p.sessions.map((s) => (
          <div key={s.id} className={`session${s.id === p.current ? " active" : ""}`}>
            {editing === s.id ? (
              <input className="rename" autoFocus defaultValue={s.title}
                onBlur={(e) => { setEditing(null); if (e.target.value.trim() && e.target.value !== s.title) p.onRename(s.id, e.target.value); }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing) (e.target as HTMLInputElement).blur();
                  if (e.key === "Escape") setEditing(null);
                }} />
            ) : (
              <button className="session-main" onClick={() => p.onSelect(s.id)} title={s.title || "新对话"}>
                <span className={`dot${s.busy ? " busy" : ""}`} />
                <span className="ellipsis">{s.title || "新对话"}</span>
              </button>
            )}
            <Menu align="right" trigger={(toggle) => (
              <button className="session-more" onClick={toggle}><MoreHorizontal size={15} /></button>
            )} items={[
              { label: "重命名", onClick: () => setEditing(s.id) },
              { label: "删除", danger: true, onClick: () => p.onDelete(s.id) },
            ]} />
          </div>
        ))}
      </nav>

      <div className="side-bottom">
        <button className="side-item faint" onClick={() => { if (!trashOpen) p.onShowTrash(); setTrashOpen(!trashOpen); }}>
          最近删除{p.trash.length > 0 && trashOpen ? ` (${p.trash.length})` : ""}
        </button>
        {trashOpen && (
          <div className="trash">
            {p.trash.map((t) => (
              <div key={t.id} className="trash-row">
                <span className="ellipsis faint">{t.title || "新对话"}</span>
                <button className="text-btn" onClick={() => p.onRestore(t.id)}>恢复</button>
              </div>
            ))}
            {p.trash.length === 0 && <div className="faint small trash-row">空</div>}
          </div>
        )}
        {p.user && (
          <Menu trigger={(toggle) => (
            <button className="user-row-btn" onClick={toggle}>
              <span className="avatar">{p.user![0].toUpperCase()}</span>
              <span className="ellipsis">{p.user}</span>
            </button>
          )} items={[{ label: "退出登录", onClick: p.onLogout }]} />
        )}
      </div>
    </aside>
  );
}
