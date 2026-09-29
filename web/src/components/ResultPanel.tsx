import { useEffect, useState } from "react";
import { ChevronRight, Download, Image, Table2, X } from "lucide-react";
import { api } from "../api";
import type { Table } from "../types";
import { DataTable } from "./DataTable";

export type PanelTarget = { kind: "table"; ref: string } | { kind: "figure"; url: string };

type Props = {
  session: string;
  results: Table[];
  figures: string[];
  target: PanelTarget;
  onSelect: (t: PanelTarget) => void;
  onClose: () => void;
};

// 右侧面板（相当于 Claude 的 artifacts）：这次对话的结果表和图
export function ResultPanel({ session, results, figures, target, onSelect, onClose }: Props) {
  const [full, setFull] = useState<Table | null>(null);
  const [showSql, setShowSql] = useState(false);
  const preview = target.kind === "table" ? results.find((t) => t.ref === target.ref) : undefined;

  // 快照里只带前 20 行，完整的按需取
  useEffect(() => {
    setFull(null);
    if (target.kind !== "table" || !preview || preview.rows.length >= preview.row_count) return;
    let alive = true;
    api.result(session, target.ref).then((t) => alive && setFull(t)).catch(() => {});
    return () => { alive = false; };
  }, [session, target, preview]);

  const table = full ?? preview;
  return (
    <section className="panel">
      <div className="panel-tabs">
        {results.map((t) => (
          <button key={t.ref} className={`tab${target.kind === "table" && target.ref === t.ref ? " active" : ""}`}
            onClick={() => onSelect({ kind: "table", ref: t.ref })} title={t.title || t.sql}>
            <Table2 size={13} /> {t.ref}{t.title ? ` · ${t.title}` : ""}
          </button>
        ))}
        {figures.map((u, i) => (
          <button key={u} className={`tab${target.kind === "figure" && target.url === u ? " active" : ""}`}
            onClick={() => onSelect({ kind: "figure", url: u })}>
            <Image size={13} /> 图 {i + 1}
          </button>
        ))}
        <button className="icon-btn panel-close" title="关闭面板" onClick={onClose}><X size={17} /></button>
      </div>

      {target.kind === "figure" ? (
        <div className="panel-body figure-view"><img src={target.url} alt="" /></div>
      ) : table ? (
        <div className="panel-body">
          <div className="panel-head">
            <div>
              <div className="panel-title">{table.title || `结果 ${table.ref}`}</div>
              <div className="muted small">
                {table.ref} · {table.row_count} 行 × {table.columns.length} 列 · 来自 {table.source === "sql" ? "SQL" : table.source === "r" ? "R" : "Python"}
                {table.truncated && " · 超过行数上限，只取了前面这些"}
              </div>
            </div>
            <a className="btn" href={api.csvUrl(session, table.ref)}><Download size={14} /> 下载 CSV</a>
          </div>
          {table.sql && (
            <div className={`sql-toggle${showSql ? " open" : ""}`}>
              <button onClick={() => setShowSql(!showSql)}><ChevronRight size={14} className="chev" /> SQL</button>
              {showSql && <pre className="code">{table.sql}</pre>}
            </div>
          )}
          <DataTable table={table} />
          {!full && table.rows.length < table.row_count && <div className="muted small pad">加载全部行…</div>}
        </div>
      ) : (
        <div className="panel-body muted pad">没有这个结果</div>
      )}
    </section>
  );
}
