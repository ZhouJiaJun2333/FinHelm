import type { Table } from "../types";

const isNumber = (v: unknown) => typeof v === "number";

export function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString("zh-CN") : v.toLocaleString("zh-CN", { maximumFractionDigits: 4 });
  }
  return String(v);
}

export function DataTable({ table, maxRows, compact }: { table: Table; maxRows?: number; compact?: boolean }) {
  const rows = maxRows ? table.rows.slice(0, maxRows) : table.rows;
  // 整列都是数字（或空）的右对齐
  const numeric = table.columns.map((_, c) => rows.length > 0 && rows.every((r) => r[c] === null || isNumber(r[c])));
  return (
    <div className={`datatable${compact ? " compact" : ""}`}>
      <table>
        <thead>
          <tr>
            <th className="rownum">#</th>
            {table.columns.map((c, i) => <th key={i} className={numeric[i] ? "num" : ""}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="rownum">{i + 1}</td>
              {r.map((v, j) => <td key={j} className={numeric[j] ? "num" : ""}>{formatCell(v)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
