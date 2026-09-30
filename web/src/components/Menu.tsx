import { useEffect, useRef, useState, type ReactNode } from "react";

export type MenuItem = { label: string; onClick: () => void; danger?: boolean; disabled?: boolean };

// 点按钮弹出、点外面收起的小菜单
export function Menu({ trigger, items, align = "left" }:
    { trigger: (toggle: () => void) => ReactNode; items: MenuItem[]; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div className="menu-wrap" ref={box}>
      {trigger(() => setOpen(!open))}
      {open && (
        <div className={`menu ${align}`}>
          {items.map((it) => (
            <button key={it.label} className={it.danger ? "danger" : ""} disabled={it.disabled}
              onClick={() => { setOpen(false); it.onClick(); }}>
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
