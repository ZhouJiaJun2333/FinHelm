import { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { Step, Tool } from "../blocks";
import { toolMeta, toolSubtitle } from "../toolMeta";
import { DataTable } from "./DataTable";
import { FileChip } from "./Artifact";
import { Subtasks } from "./Subtasks";

type Props = { steps: Step[]; files: string[]; live: boolean; onOpenFile: (url: string) => void;
               onOpenResult: (ref: string) => void };

// 一段「思考 + 工具调用」：平时是一行淡灰小字，点开是每一步
export function Activity({ steps, files, live, onOpenFile, onOpenResult }: Props) {
  const [open, setOpen] = useState(false);
  const tools = steps.filter((s) => s.kind === "tool");
  const failed = tools.filter((s) => s.tool.status === "error" || s.tool.status === "denied").length;
  const running = live ? tools.find((s) => s.tool.status === "running") : undefined;

  let label: React.ReactNode;
  if (running?.tool.name === "delegate") {
    const n = (running.tool.arguments.tasks as unknown[] | undefined)?.length ?? 0;
    const done = running.tool.children?.filter((c) => c.status !== "running").length ?? 0;
    label = <span className="shimmer">{n} 个子任务进行中{done ? `，${done} 个已完成` : ""}…</span>;
  } else if (running) {
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
            : <ToolStep key={s.id} tool={s.tool} live={live} onOpenFile={onOpenFile} onOpenResult={onOpenResult} />)}
        </div>
      )}
      {files.length > 0 && (
        <div className="file-chips">
          {files.map((u) => <FileChip key={u} url={u} onOpen={onOpenFile} />)}
        </div>
      )}
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

function ToolStep({ tool, live, onOpenFile, onOpenResult }:
    { tool: Tool; live: boolean; onOpenFile: (url: string) => void; onOpenResult: (ref: string) => void }) {
  // 分派的子任务默认展开：它们就是这一步在做的事
  const [open, setOpen] = useState(tool.name === "delegate");
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
      {open && tool.name === "delegate" ? (
        <Subtasks tool={tool} live={live} onOpenFile={onOpenFile} onOpenResult={onOpenResult} />
      ) : open && (
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
