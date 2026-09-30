import { useEffect, useState } from "react";
import { Code2, Download, Image, Table2, X } from "lucide-react";
import { api } from "../api";
import type { Table } from "../types";
import { DataTable } from "./DataTable";

export type PanelTarget = { kind: "table"; ref: string } | { kind: "figure"; url: string };

export const sameTarget = (a: PanelTarget, b: PanelTarget) =>
  a.kind === b.kind && (a.kind === "table" ? a.ref === (b as typeof a).ref : a.url === (b as typeof a).url);

type Props = {
  session: string;
  results: Table[];
  tabs: PanelTarget[];
  active: PanelTarget;
  onSelect: (t: PanelTarget) => void;
  onCloseTab: (t: PanelTarget) => void;
  onClose: () => void;
};

// 右侧面板（Claude 的文件面板）：打开过的结果表和图，一个标签一个
export function ResultPanel({ session, results, tabs, active, onSelect, onCloseTab, onClose }: Props) {
  const [full, setFull] = useState<Table | null>(null);
  const [showSql, setShowSql] = useState(false);
  const preview = active.kind === "table" ? results.find((t) => t.ref === active.ref) : undefined;

  // 快照里只带前 20 行，完整的按需取
  useEffect(() => {
    setFull(null);
    if (!preview || preview.rows.length >= preview.row_count) return;
    let alive = true;
    api.result(session, preview.ref).then((t) => alive && setFull(t)).catch(() => {});
    return () => { alive = false; };
  }, [session, preview]);

  const table = full ?? preview;
  const name = (t: PanelTarget) => {
    if (t.kind === "figure") return t.url.split("?")[0].split("/").pop();
    const r = results.find((x) => x.ref === t.ref);
    return r?.title || t.ref;
  };

  return (
    <section className="panel">
      <div className="panel-tabs">
        {tabs.map((t) => (
          <div key={t.kind === "table" ? t.ref : t.url} className={`tab${sameTarget(t, active) ? " active" : ""}`}>
            <button className="tab-main" onClick={() => onSelect(t)}>
              {t.kind === "table" ? <Table2 size={13} /> : <Image size={13} />}
              <span className="ellipsis">{name(t)}</span>
            </button>
            <button className="tab-x" onClick={() => onCloseTab(t)}><X size={12} /></button>
          </div>
        ))}
        <span className="spacer" />
        {table && (
          <>
            {table.sql && (
              <button className={`icon-btn${showSql ? " on" : ""}`} title="SQL" onClick={() => setShowSql(!showSql)}>
                <Code2 size={16} />
              </button>
            )}
            <a className="icon-btn" title="下载 CSV" href={api.csvUrl(session, table.ref)}><Download size={16} /></a>
          </>
        )}
        <button className="icon-btn" title="关闭" onClick={onClose}><X size={16} /></button>
      </div>

      {active.kind === "figure" ? (
        <div className="panel-body figure-view"><img src={active.url} alt="" /></div>
      ) : table ? (
        <div className="panel-body">
          <div className="panel-head">
            <div className="panel-title">{table.title || `结果 ${table.ref}`}</div>
            <div className="faint small">{table.ref} · {table.row_count} 行 × {table.columns.length} 列</div>
          </div>
          {showSql && table.sql && <pre className="code panel-sql">{table.sql}</pre>}
          <DataTable table={table} />
        </div>
      ) : (
        <div className="panel-body faint pad">没有这个结果</div>
      )}
    </section>
  );
}
