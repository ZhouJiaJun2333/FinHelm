import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, Plus, Square, X } from "lucide-react";

type Props = {
  busy: boolean;
  disabled?: boolean;
  placeholder: string;
  uploads: { name: string; size: number }[];
  canUpload: boolean;
  autoFocus?: boolean;
  footer?: ReactNode;               // 输入框下面右边那一行（模型、上下文用量）
  onSend: (text: string) => void;
  onStop: () => void;
  onUpload: (files: File[]) => void;
  onRemoveUpload?: (name: string) => void;
};

export function Composer(p: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 280)}px`;
  }, [text]);

  const submit = () => {
    if (p.busy || p.disabled || !text.trim()) return;
    p.onSend(text.trim());
    setText("");
  };

  return (
    <div className="composer-wrap">
      <div className={`composer${p.disabled ? " disabled" : ""}`}
        onDragOver={(e) => { if (p.canUpload) e.preventDefault(); }}
        onDrop={(e) => { if (!p.canUpload) return; e.preventDefault(); p.onUpload([...e.dataTransfer.files]); }}>
        {p.uploads.length > 0 && (
          <div className="uploads">
            {p.uploads.map((u) => (
              <span key={u.name} className="upload-chip">
                {u.name}
                {p.onRemoveUpload && <button onClick={() => p.onRemoveUpload!(u.name)}><X size={12} /></button>}
              </span>
            ))}
          </div>
        )}
        <div className="composer-row">
          <textarea ref={area} rows={1} value={text} autoFocus={p.autoFocus} placeholder={p.placeholder}
            disabled={p.disabled}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              // 输入法选词时的回车不是发送
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }} />
          {p.busy ? (
            <button className="send stop" title="停止" onClick={p.onStop}><Square size={11} fill="currentColor" /></button>
          ) : (
            <button className="send" title="发送" disabled={!text.trim() || p.disabled} onClick={submit}>
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
      <div className="composer-foot">
        <button className="icon-btn small" title="上传文件" disabled={!p.canUpload} onClick={() => file.current?.click()}>
          <Plus size={16} />
        </button>
        <input ref={file} type="file" multiple hidden
          onChange={(e) => { p.onUpload([...(e.target.files ?? [])]); e.target.value = ""; }} />
        <span className="spacer" />
        {p.footer}
      </div>
    </div>
  );
}
