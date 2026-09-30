import {
  BookOpen, Brain, Code2, Database, Download, FileText, Image, Library, ListTree, Plug, Search, Sparkles,
  TableProperties, type LucideIcon,
} from "lucide-react";

// label：做完了怎么说；doing：正在做的时候怎么说
const TOOLS: Record<string, { label: string; doing: string; icon: LucideIcon }> = {
  run_sql: { label: "查询数据库", doing: "正在查询数据库", icon: Database },
  describe_table: { label: "查看表结构", doing: "正在查看表结构", icon: TableProperties },
  list_tables: { label: "列出数据表", doing: "正在列出数据表", icon: ListTree },
  export_csv: { label: "导出 CSV", doing: "正在导出 CSV", icon: Download },
  run_python: { label: "运行 Python", doing: "正在运行 Python", icon: Code2 },
  run_r: { label: "运行 R", doing: "正在运行 R", icon: Code2 },
  read_file: { label: "读取文件", doing: "正在读取文件", icon: FileText },
  view_image: { label: "查看图片", doing: "正在查看图片", icon: Image },
  load_skill: { label: "加载技能", doing: "正在加载技能", icon: Sparkles },
  remember: { label: "写入记忆", doing: "正在写入记忆", icon: Brain },
  read_memory: { label: "读取记忆", doing: "正在读取记忆", icon: Brain },
  list_docs: { label: "列出文档", doing: "正在列出文档", icon: Library },
  search_docs: { label: "检索文档", doing: "正在检索文档", icon: Search },
  read_doc: { label: "阅读文档", doing: "正在阅读文档", icon: BookOpen },
};

export function toolMeta(name: string) {
  if (TOOLS[name]) return TOOLS[name];
  const mcp = name.match(/^mcp__(.+?)__(.+)$/);
  const label = mcp ? `${mcp[1]} · ${mcp[2]}` : name;
  return { label, doing: `正在调用 ${label}`, icon: Plug };
}

// 标题后面那句：模型写的用途，没有就取 SQL / 代码 / 参数的第一行
export function toolSubtitle(args: Record<string, unknown>): string {
  const pick = args.purpose ?? args.query ?? args.sql ?? args.code ?? args.table ?? args.path ?? args.name ?? "";
  return String(pick).split("\n").find((l) => l.trim())?.trim() ?? "";
}
