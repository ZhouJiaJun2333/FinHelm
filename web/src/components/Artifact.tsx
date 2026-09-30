import type { ReactNode } from "react";
import { FILE_ICONS, FILE_LABELS, fileKind, fileName } from "../files";
import type { Table } from "../types";
import { formatCell } from "./DataTable";

// Claude 的 artifact 预览卡：上面缩略图，下面标题、一行说明和「打开」
export function ArtifactCard({ title, subtitle, thumb, onOpen }:
    { title: string; subtitle: string; thumb: ReactNode; onOpen: () => void }) {
  return (
    <div className="artifact" onClick={onOpen}>
      <div className="artifact-thumb">{thumb}</div>
      <div className="artifact-bar">
        <div className="artifact-text">
          <div className="artifact-title">{title}</div>
          <div className="faint small">{subtitle}</div>
        </div>
        <button className="btn" onClick={(e) => { e.stopPropagation(); onOpen(); }}>打开</button>
      </div>
    </div>
  );
}

// 表格的缩略图：前几行缩小画出来
export function TableThumb({ table }: { table: Table }) {
  const cols = table.columns.slice(0, 6);
  return (
    <table className="thumb-table">
      <thead><tr>{cols.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
      <tbody>
        {table.rows.slice(0, 7).map((r, i) => (
          <tr key={i}>{cols.map((_, j) => <td key={j}>{formatCell(r[j])}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

export function ResultCard({ table, refName, onOpen }: { table?: Table; refName: string; onOpen: (ref: string) => void }) {
  if (!table) return <span className="faint">[{refName}]</span>;
  return (
    <ArtifactCard title={table.title || `结果 ${table.ref}`}
      subtitle={`${table.ref} · ${table.row_count} 行 × ${table.columns.length} 列`}
      thumb={<TableThumb table={table} />} onOpen={() => onOpen(table.ref)} />
  );
}

// 工具产出的文件（图、PDF、Excel…）：一个小卡片，点开在右侧面板看。图的图标位置放缩略图
export function FileChip({ url, onOpen }: { url: string; onOpen: (url: string) => void }) {
  const kind = fileKind(url);
  const Icon = FILE_ICONS[kind];
  return (
    <button className="file-chip" onClick={() => onOpen(url)}>
      {kind === "image" ? <img className="file-chip-thumb" src={url} alt="" /> : <Icon size={16} />}
      <span className="file-chip-text">
        <span className="ellipsis">{fileName(url)}</span>
        <span className="faint small">{FILE_LABELS[kind]}</span>
      </span>
    </button>
  );
}
