import { useState } from "react";
import { Bot, ChevronRight } from "lucide-react";
import { toBlocks, type Tool } from "../blocks";
import { toolMeta } from "../toolMeta";
import type { Subtask } from "../types";
import { Activity } from "./Activity";
import { Markdown } from "./Markdown";

type Props = { tool: Tool; live: boolean; onOpenFile: (url: string) => void; onOpenResult: (ref: string) => void };

const byNumber = (a: Subtask, b: Subtask) => Number(a.task.slice(1)) - Number(b.task.slice(1));

// delegate 展开后：一个子任务一行，再点开是它自己的过程（和主对话一样的淡灰一行）和交回的结论
export function Subtasks({ tool, live, onOpenFile, onOpenResult }: Props) {
  const children = [...(tool.children ?? [])].sort(byNumber);
  // 刚分派、子任务的事件还没到：先按参数列出来
  const planned = (tool.arguments.tasks as { agent: string; title: string }[] | undefined) ?? [];
  if (!children.length) {
    return (
      <div className="subtasks">
        {planned.map((t, i) => (
          <div key={i} className="step-line"><Bot size={13} /><span>{t.title}</span><span className="faint">{t.agent}</span></div>
        ))}
      </div>
    );
  }
  return (
    <div className="subtasks">
      {children.map((c) => <SubtaskRow key={c.task} task={c} live={live} onOpenFile={onOpenFile} onOpenResult={onOpenResult} />)}
    </div>
  );
}

function SubtaskRow({ task, live, onOpenFile, onOpenResult }:
    { task: Subtask; live: boolean; onOpenFile: (url: string) => void; onOpenResult: (ref: string) => void }) {
  const [open, setOpen] = useState(false);
  const blocks = toBlocks(task.items);
  const tools = task.items.filter((it) => it.kind === "tool");
  const running = tools.find((it) => it.kind === "tool" && it.status === "running");
  const prompt = task.items.find((it) => it.kind === "user");

  let state: React.ReactNode;
  if (task.status === "running" && live) {
    state = <span className="shimmer">{running ? toolMeta(running.name).doing : "思考中"}…</span>;
  } else if (task.status === "error") {
    state = <span className="bad-text">出错了</span>;
  } else if (task.status === "stopped" || task.status === "running") {
    state = "已停止";
  } else {
    state = tools.length ? `用了 ${tools.length} 个工具` : "完成";
  }

  return (
    <div className="subtask">
      <button className={`step-line${open ? " open" : ""}`} onClick={() => setOpen(!open)}>
        <Bot size={13} />
        <span className="subtask-title">{task.title}</span>
        <span className="faint">{task.agent} · {state}</span>
        <ChevronRight size={13} className="chev" />
      </button>
      {open && (
        <div className="subtask-body">
          {prompt?.kind === "user" && <div className="subtask-prompt">{prompt.text}</div>}
          {blocks.map((b) => {
            switch (b.kind) {
              case "activity":
                // 文件在外面那一行统一列，这里不重复
                return <Activity key={b.id} steps={b.steps} files={[]} live={live && task.status === "running"}
                                 onOpenFile={onOpenFile} onOpenResult={onOpenResult} />;
              case "text":
                return <div key={b.id} className="subtask-answer"><Markdown text={b.item.text} results={[]} onOpenResult={onOpenResult} /></div>;
              case "notice":
                return <div key={b.id} className={`notice ${b.level}`}>{b.text}</div>;
              default:
                return null;
            }
          })}
        </div>
      )}
    </div>
  );
}
