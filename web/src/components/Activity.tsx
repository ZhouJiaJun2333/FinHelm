import { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { Step, Tool } from "../blocks";
import { toolMeta, toolSubtitle } from "../toolMeta";
import { DataTable } from "./DataTable";
import { ArtifactCard } from "./Artifact";

type Props = { steps: Step[]; figures: string[]; live: boolean; onOpenFigure: (url: string) => void;
               onOpenResult: (ref: string) => void };

// 一段「思考 + 工具调用」：平时是一行淡灰小字，点开是每一步
export function Activity({ steps, figures, live, onOpenFigure, onOpenResult }: Props) {
  const [open, setOpen] = useState(false);
  const tools = steps.filter((s) => s.kind === "tool");
  const failed = tools.filter((s) => s.tool.status === "error" || s.tool.status === "denied").length;
  const running = live ? tools.find((s) => s.tool.status === "running") : undefined;

  let label: React.ReactNode;
  if (running) {
    label = <span className="shimmer">{toolMeta(running.tool.name).doing}…</span>;
  } else if (live && steps.at(-1)?.kind === "thinking") {
    label = <span className="shimmer">思考中…</span>;
  } else if (tools.length === 0) {
    label = "思考过程";
  } else if (tools.length === 1) {
    const t = tools[0].tool;
    const sub = toolSubtitle(t.arguments);
    label = <>{toolMeta(t.name).label}{sub && <span className="faint"> · {sub}</span>}</>;
  } else {
    label = `用了 ${tools.length} 个工具`;
  }

  return (
    <div className="activity">
      <button className={`activity-line${open ? " open" : ""}`} onClick={() => setOpen(!open)}>
        <span className="activity-label">{label}</span>
        {failed > 0 && !running && <span className="bad-text"> · {failed} 个出错</span>}
        <ChevronRight size={14} className="chev" />
      </button>
      {open && (
        <div className="steps">
          {steps.map((s) => s.kind === "thinking"
            ? <ThinkingStep key={`t${s.id}`} text={s.text} />
            : <ToolStep key={s.id} tool={s.tool} onOpenResult={onOpenResult} />)}
        </div>
      )}
      {figures.map((u) => (
        <ArtifactCard key={u} title={u.split("?")[0].split("/").pop() ?? "图"} subtitle="图片"
          thumb={<img src={u} alt="" />} onOpen={() => onOpenFigure(u)} />
      ))}
    </div>
  );
}

function ThinkingStep({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="step">
      <button className={`step-line${open ? " open" : ""}`} onClick={() => setOpen(!open)}>
        <span>思考</span><ChevronRight size={13} className="chev" />
      </button>
      {open && <div className="step-body thinking-text">{text}</div>}
    </div>
  );
}

function ToolStep({ tool, onOpenResult }: { tool: Tool; onOpenResult: (ref: string) => void }) {
  const [open, setOpen] = useState(false);
  const { label, icon: Icon } = toolMeta(tool.name);
  const code = (tool.arguments.sql ?? tool.arguments.code) as string | undefined;
  const rest = Object.fromEntries(Object.entries(tool.arguments).filter(([k]) => k !== "sql" && k !== "code"));
  const d = tool.details;
  const bad = tool.status === "error" || tool.status === "denied";
  return (
    <div className="step">
      <button className={`step-line${open ? " open" : ""}${bad ? " bad-text" : ""}`} onClick={() => setOpen(!open)}>
        <Icon size={13} />
        <span>{label}</span>
        <span className="faint ellipsis">{toolSubtitle(tool.arguments)}</span>
        <ChevronRight size={13} className="chev" />
      </button>
      {open && (
        <div className="step-body">
          {code && <pre className="code">{code}</pre>}
          {Object.keys(rest).length > 0 && <pre className="code">{JSON.stringify(rest, null, 2)}</pre>}
          {d?.kind === "table" ? (
            <div className="step-table" onClick={() => onOpenResult(d.ref)}>
              <DataTable table={d} maxRows={8} compact />
            </div>
          ) : tool.content ? <pre className={`code out${bad ? " bad-text" : ""}`}>{tool.content}</pre> : null}
        </div>
      )}
    </div>
  );
}
