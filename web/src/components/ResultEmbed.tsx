import { Maximize2, Table2 } from "lucide-react";
import type { Table } from "../types";
import { DataTable } from "./DataTable";

const PREVIEW = 6;

// 嵌在回答里的结果：标题 + 前几行，点开在右边面板看全部
export function ResultEmbed({ table, refName, onOpen }: { table?: Table; refName: string; onOpen: (ref: string) => void }) {
  if (!table) return <span className="chip muted">找不到结果 {refName}</span>;
  return (
    <div className="embed">
      <button className="embed-head" onClick={() => onOpen(table.ref)}>
        <Table2 size={15} />
        <span className="embed-title">{table.title || `结果 ${table.ref}`}</span>
        <span className="muted">{table.ref} · {table.row_count} 行 × {table.columns.length} 列</span>
        <Maximize2 size={14} className="embed-open" />
      </button>
      <DataTable table={table} maxRows={PREVIEW} compact />
      {table.row_count > PREVIEW && (
        <button className="embed-more" onClick={() => onOpen(table.ref)}>查看全部 {table.row_count} 行</button>
      )}
    </div>
  );
}
