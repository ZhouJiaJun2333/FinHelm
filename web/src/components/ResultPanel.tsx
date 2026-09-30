import { useEffect, useState } from "react";
import { Code2, Download, Table2, X } from "lucide-react";
import { api } from "../api";
import { FILE_ICONS, fileKind, fileName, sheetsUrl } from "../files";
import type { Table } from "../types";
import { DataTable } from "./DataTable";

export type PanelTarget = { kind: "table"; ref: string } | { kind: "file"; url: string };

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

// 右侧面板（Claude 的文件面板）：打开过的结果表和文件，一个标签一个
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
    if (t.kind === "file") return fileName(t.url);
    const r = results.find((x) => x.ref === t.ref);
    return r?.title || t.ref;
  };

  return (
    <section className="panel">
      <div className="panel-tabs">
        {tabs.map((t) => {
          const Icon = t.kind === "table" ? Table2 : FILE_ICONS[fileKind(t.url)];
          return (
            <div key={t.kind === "table" ? t.ref : t.url} className={`tab${sameTarget(t, active) ? " active" : ""}`}>
              <button className="tab-main" onClick={() => onSelect(t)}>
                <Icon size={13} />
                <span className="ellipsis">{name(t)}</span>
              </button>
              <button className="tab-x" onClick={() => onCloseTab(t)}><X size={12} /></button>
            </div>
          );
        })}
        <span className="spacer" />
        {active.kind === "file" && (
          <a className="icon-btn" title="下载" href={active.url} download={fileName(active.url)}><Download size={16} /></a>
        )}
        {active.kind === "table" && table && (
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

      {active.kind === "file" ? (
        <FileView key={active.url} url={active.url} />
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

function FileView({ url }: { url: string }) {
  switch (fileKind(url)) {
    case "image":
      return <ImageView url={url} />;
    case "pdf":
      // 浏览器自带的 PDF 阅读器；关掉左边的缩略图栏，宽度撑满
      return <div className="panel-body"><iframe className="pdf-view" src={`${url}#navpanes=0&view=FitH`} title={fileName(url)} /></div>;
    case "sheet":
      return <SheetView url={url} />;
    default:
      return <div className="panel-body faint pad">这个文件不能预览，可以点右上角下载。</div>;
  }
}

// 图默认缩到面板宽；300 dpi 的图缩下来小字发虚，点一下放大看细节，再点缩回去
function ImageView({ url }: { url: string }) {
  const [zoomed, setZoomed] = useState(false);
  return (
    <div className={`panel-body figure-view${zoomed ? " zoomed" : ""}`}>
      <img src={url} alt="" title={zoomed ? "点击缩小" : "点击放大"} onClick={() => setZoomed(!zoomed)} />
    </div>
  );
}

// Excel / CSV：一个工作表一个按钮，下面是表格
function SheetView({ url }: { url: string }) {
  const [sheets, setSheets] = useState<Table[] | null>(null);
  const [error, setError] = useState("");
  const [index, setIndex] = useState(0);

  useEffect(() => {
    let alive = true;
    api.sheets(sheetsUrl(url)).then((s) => alive && setSheets(s)).catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [url]);

  if (error) return <div className="panel-body faint pad">{error}</div>;
  if (!sheets) return <div className="panel-body faint pad">读取中…</div>;
  const sheet = sheets[Math.min(index, sheets.length - 1)];
  if (!sheet) return <div className="panel-body faint pad">没有工作表</div>;
  return (
    <div className="panel-body">
      <div className="panel-head">
        {sheets.length > 1 && (
          <div className="sheet-tabs">
            {sheets.map((s, i) => (
              <button key={i} className={i === index ? "on" : ""} onClick={() => setIndex(i)}>{s.title}</button>
            ))}
          </div>
        )}
        <div className="faint small">
          {sheet.row_count} 行 × {sheet.columns.length} 列{sheet.truncated && `，只显示前 ${sheet.rows.length} 行`}
        </div>
      </div>
      <DataTable table={sheet} />
    </div>
  );
}
