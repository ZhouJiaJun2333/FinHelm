import { createContext, useContext } from "react";

// 当前会话 id：回答里的相对路径（figures/a.png）要换成这个会话 work 目录的文件地址
export const SessionContext = createContext<string | null>(null);

export function useFileUrl() {
  const id = useContext(SessionContext);
  return (src?: string) => {
    if (!src || !id || /^(https?:|data:|\/api\/)/.test(src)) return src;
    const path = src.replace(/^\.?\//, "").replace(/^work\//, "");
    return `/api/sessions/${encodeURIComponent(id)}/files/${path}`;
  };
}
