import { useState } from "react";
import {
  Ban, BookOpen, Brain, Check, ChevronRight, Code2, Database, Download, FileText, Image, Library, ListTree,
  Loader2, Plug, Search, Sparkles, TableProperties, X,
} from "lucide-react";
import type { Item } from "../types";
import { DataTable } from "./DataTable";

type Tool = Extract<Item, { kind: "tool" }>;

const TOOLS: Record<string, { label: string; icon: typeof Database }> = {
  run_sql: { label: "查询数据库", icon: Database },
  describe_table: { label: "查看表结构", icon: TableProperties },
  list_tables: { label: "列出数据表", icon: ListTree },
  export_csv: { label: "导出 CSV", icon: Download },
  run_python: { label: "运行 Python", icon: Code2 },
  run_r: { label: "运行 R", icon: Code2 },
  read_file: { label: "读取文件", icon: FileText },
  view_image: { label: "查看图片", icon: Image },
  load_skill: { label: "加载技能", icon: Sparkles },
  remember: { label: "写入记忆", icon: Brain },
  read_memory: { label: "读取记忆", icon: Brain },
  list_docs: { label: "列出文档", icon: Library },
  search_docs: { label: "检索文档", icon: Search },
  read_doc: { label: "阅读文档", icon: BookOpen },
};

function describe(name: string) {
  if (TOOLS[name]) return TOOLS[name];
  const mcp = name.match(/^mcp__(.+?)__(.+)$/);
  return { label: mcp ? `${mcp[1]} · ${mcp[2]}` : name, icon: Plug };
}

// 卡片标题后面那句：模型写的用途，没有就取 SQL / 代码 / 参数的第一行
function subtitle(args: Record<string, unknown>): string {
  const pick = args.purpose ?? args.query ?? args.sql ?? args.code ?? args.table ?? args.path ?? args.name ?? "";
  return String(pick).split("\n").find((l) => l.trim())?.trim() ?? "";
}

function StatusIcon({ status }: { status: Tool["status"] }) {
  if (status === "running") return <Loader2 size={14} className="spin" />;
  if (status === "done") return <Check size={14} className="ok" />;
  if (status === "denied") return <Ban size={14} className="bad" />;
  if (status === "asking") return <span className="dot-pulse" />;
  return <X size={14} className="bad" />;
}

export function ToolCard({ tool, onOpenResult }: { tool: Tool; onOpenResult: (ref: string) => void }) {
  const [open, setOpen] = useState(false);
  const { label, icon: Icon } = describe(tool.name);
  const code = (tool.arguments.sql ?? tool.arguments.code) as string | undefined;
  const rest = Object.fromEntries(Object.entries(tool.arguments).filter(([k]) => k !== "sql" && k !== "code"));
  const d = tool.details;

  return (
    <div className={`tool ${tool.status}${open ? " open" : ""}`}>
      <button className="tool-head" onClick={() => setOpen(!open)}>
        <Icon size={15} className="tool-icon" />
        <span className="tool-label">{label}</span>
        <span className="tool-sub">{subtitle(tool.arguments)}</span>
        {tool.elapsed_ms > 0 && <span className="tool-ms">{fmtMs(tool.elapsed_ms)}</span>}
        <StatusIcon status={tool.status} />
        <ChevronRight size={14} className="chev" />
      </button>
      {d?.kind === "table" && !open && (
        <button className="tool-result-chip" onClick={() => onOpenResult(d.ref)}>
          {d.ref} · {d.row_count} 行 × {d.columns.length} 列
        </button>
      )}
      {d?.kind === "execution" && d.figures.length > 0 && (
        <div className="figures">{d.figures.map((u) => <a key={u} href={u} target="_blank" rel="noreferrer"><img src={u} alt="" /></a>)}</div>
      )}
      {open && (
        <div className="tool-body">
          {code && <pre className="code">{code}</pre>}
          {Object.keys(rest).length > 0 && <pre className="code args">{JSON.stringify(rest, null, 2)}</pre>}
          {d?.kind === "table" ? (
            <>
              <DataTable table={d} maxRows={10} compact />
              <button className="link" onClick={() => onOpenResult(d.ref)}>在面板中打开 {d.ref}</button>
            </>
          ) : tool.content ? (
            <pre className={`output${tool.status === "error" ? " err" : ""}`}>{tool.content}</pre>
          ) : tool.status === "running" ? <div className="muted small">执行中…</div> : null}
        </div>
      )}
    </div>
  );
}

function fmtMs(ms: number) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}
