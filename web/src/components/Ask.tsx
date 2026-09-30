import { useState } from "react";
import { ArrowUp } from "lucide-react";
import type { Tool } from "../blocks";

// ask_user：问题 + 编号选项 + 「其他」（点开在卡片里写），答过的只留问答
export function AskCard({ tool, pending, onAnswer }: { tool: Tool; pending: boolean; onAnswer: (a: string) => void }) {
  const [other, setOther] = useState(false);
  const [text, setText] = useState("");
  const question = String(tool.arguments.question ?? "");
  const options = (tool.arguments.options as string[] | undefined) ?? [];

  if (!pending) {
    const answer = tool.content.replace(/^用户回答：/, "");
    return (
      <div className="asked">
        <div className="faint">{question}</div>
        {tool.status === "done" && <div className="asked-a">{answer.startsWith("[用户没有回答") ? "已跳过" : answer}</div>}
      </div>
    );
  }

  const submit = () => text.trim() && onAnswer(text.trim());
  return (
    <div className="ask">
      <div className="ask-q">{question}</div>
      <div className="ask-options">
        {options.map((o, i) => (
          <button key={o} className="ask-option" onClick={() => onAnswer(o)}>
            <span className="ask-n">{i + 1}</span><span>{o}</span>
          </button>
        ))}
        {other ? (
          <div className="ask-option ask-other">
            <span className="ask-n">{options.length + 1}</span>
            <input autoFocus value={text} placeholder="其他" onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                // 吃掉这个回车：卡片答完就没了，不吞的话它会落到别的按钮上
                if (e.key === "Enter" && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.blur(); submit(); }
              }} />
            <button className="send small" disabled={!text.trim()} onClick={submit}><ArrowUp size={14} /></button>
          </div>
        ) : (
          <button className="ask-option" onClick={() => setOther(true)}>
            <span className="ask-n">{options.length + 1}</span><span className="faint">其他…</span>
          </button>
        )}
      </div>
      <div className="ask-foot"><button className="text-btn" onClick={() => onAnswer("")}>跳过</button></div>
    </div>
  );
}
