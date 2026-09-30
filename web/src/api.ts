import type { Me, SessionSummary, Table, Trashed } from "./types";

// 没登录（或登录过期）时抛这个，App 收到就回登录页
export class Unauthorized extends Error {}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (res.status === 401) throw new Unauthorized();
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const s = (id: string) => `/api/sessions/${encodeURIComponent(id)}`;

export const api = {
  me: () => call<Me>("GET", "/api/me"),
  login: (name: string, password: string) => call("POST", "/api/login", { name, password }),
  logout: () => call("POST", "/api/logout"),
  sessions: () => call<SessionSummary[]>("GET", "/api/sessions"),
  create: () => call<{ id: string }>("POST", "/api/sessions"),
  rename: (id: string, title: string) => call("PATCH", s(id), { title }),
  remove: (id: string) => call("DELETE", s(id)),
  trash: () => call<Trashed[]>("GET", "/api/trash"),
  restore: (id: string) => call("POST", `/api/trash/${encodeURIComponent(id)}/restore`),
  send: (id: string, text: string) => call("POST", `${s(id)}/messages`, { text }),
  resume: (id: string, text = "") => call("POST", `${s(id)}/continue`, { text }),
  stop: (id: string) => call("POST", `${s(id)}/stop`),
  reset: (id: string) => call("POST", `${s(id)}/reset`),
  compact: (id: string) => call("POST", `${s(id)}/compact`),
  approve: (id: string, rid: string, decision: "once" | "session" | "deny") =>
    call("POST", `${s(id)}/approvals/${rid}`, { decision }),
  result: (id: string, ref: string) => call<Table>("GET", `${s(id)}/results/${ref}`),
  csvUrl: (id: string, ref: string) => `${s(id)}/results/${ref}/csv`,
  sheets: (url: string) => call<{ sheets: Table[] }>("GET", url).then((r) => r.sheets),
  events: (id: string) => `${s(id)}/events`,
  async upload(id: string, files: File[]) {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await fetch(`${s(id)}/uploads`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`上传失败：${res.statusText}`);
    return (await res.json()) as { files: { name: string; size: number }[] };
  },
};
