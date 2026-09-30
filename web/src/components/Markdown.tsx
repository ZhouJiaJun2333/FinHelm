import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Table } from "../types";
import { ResultCard } from "./Artifact";
import { useFileUrl } from "../sessionContext";

const REF = /\{\{\s*(r\d+)\s*\}\}/g;

type Props = { text: string; results: Table[]; onOpenResult: (ref: string) => void };

// 回答里的 {{r3}} 换成结果预览卡（CLI 里是展开成表格）。按编号切开，文字段照常渲染 Markdown
export function Markdown({ text, results, onOpenResult }: Props) {
  const fileUrl = useFileUrl();
  const parts = text.split(REF);
  return (
    <div className="md">
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <ResultCard key={i} table={results.find((t) => t.ref === part)} refName={part} onOpen={onOpenResult} />
        ) : part.trim() ? (
          <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}
            components={{ table: ({ children }) => <div className="md-table"><table>{children}</table></div>,
                          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
                          img: ({ src, alt }) => <img className="md-img" src={fileUrl(src as string)} alt={alt ?? ""} /> }}>
            {part}
          </ReactMarkdown>
        ) : null,
      )}
    </div>
  );
}
