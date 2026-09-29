import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square, X } from "lucide-react";

type Props = {
  busy: boolean;
  placeholder: string;
  uploads: { name: string; size: number }[];
  canUpload: boolean;
  autoFocus?: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onUpload: (files: File[]) => void;
  onRemoveUpload?: (name: string) => void;
};

export function Composer({ busy, placeholder, uploads, canUpload, autoFocus, onSend, onStop, onUpload, onRemoveUpload }: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const file = useRef<HTMLInputElement>(null);

  // 高度跟着内容长，最多 12 行左右
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 280)}px`;
  }, [text]);

  const submit = () => {
    if (busy || !text.trim()) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="composer"
      onDragOver={(e) => { if (canUpload) e.preventDefault(); }}
      onDrop={(e) => { if (!canUpload) return; e.preventDefault(); onUpload([...e.dataTransfer.files]); }}>
      {uploads.length > 0 && (
        <div className="uploads">
          {uploads.map((u) => (
            <span key={u.name} className="upload-chip">
              📎 {u.name} <span className="muted">{kb(u.size)}</span>
              {onRemoveUpload && <button onClick={() => onRemoveUpload(u.name)}><X size={12} /></button>}
            </span>
          ))}
        </div>
      )}
      <textarea ref={area} rows={1} value={text} autoFocus={autoFocus} placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // 输入法选词时的回车不是发送
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }} />
      <div className="composer-bar">
        <button className="icon-btn" title={canUpload ? "上传文件（Excel、CSV、PDF…）" : "没开沙箱，上传了模型也读不了"}
          disabled={!canUpload} onClick={() => file.current?.click()}>
          <Paperclip size={17} />
        </button>
        <input ref={file} type="file" multiple hidden
          onChange={(e) => { onUpload([...(e.target.files ?? [])]); e.target.value = ""; }} />
        <span className="hint">Enter 发送 · Shift+Enter 换行</span>
        {busy ? (
          <button className="send stop" title="停止" onClick={onStop}><Square size={13} fill="currentColor" /></button>
        ) : (
          <button className="send" title="发送" disabled={!text.trim()} onClick={submit}><ArrowUp size={17} /></button>
        )}
      </div>
    </div>
  );
}

function kb(n: number) {
  return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
}
