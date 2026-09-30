import { File, FileSpreadsheet, FileText, Image, type LucideIcon } from "lucide-react";

// 工具产出、用户上传的文件：按扩展名决定右侧面板怎么预览
export type FileKind = "image" | "pdf" | "sheet" | "other";

const KINDS: [RegExp, FileKind][] = [
  [/\.(png|jpe?g|gif|svg|webp)$/i, "image"],
  [/\.pdf$/i, "pdf"],
  [/\.(xlsx|xlsm|csv)$/i, "sheet"],
];

const path = (url: string) => url.split("?")[0];

export const fileKind = (url: string): FileKind => KINDS.find(([re]) => re.test(path(url)))?.[1] ?? "other";

export const fileName = (url: string) => decodeURIComponent(path(url).split("/").pop() ?? "");

export const sheetsUrl = (url: string) => path(url).replace("/files/", "/sheets/");

export const FILE_ICONS: Record<FileKind, LucideIcon> = { image: Image, pdf: FileText, sheet: FileSpreadsheet, other: File };

export const FILE_LABELS: Record<FileKind, string> = { image: "图片", pdf: "PDF", sheet: "表格", other: "文件" };
