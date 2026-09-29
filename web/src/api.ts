import type { SessionSummary } from "./types";

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const s = (id: string) => `/api/sessions/${encodeURIComponent(id)}`;

export const api = {
  sessions: () => call<SessionSummary[]>("GET", "/api/sessions"),
  create: () => call<{ id: string }>("POST", "/api/sessions"),
  send: (id: string, text: string) => call("POST", `${s(id)}/messages`, { text }),
  resume: (id: string, text = "") => call("POST", `${s(id)}/continue`, { text }),
  stop: (id: string) => call("POST", `${s(id)}/stop`),
  reset: (id: string) => call("POST", `${s(id)}/reset`),
  compact: (id: string) => call("POST", `${s(id)}/compact`),
  approve: (id: string, rid: string, decision: "once" | "session" | "deny") =>
    call("POST", `${s(id)}/approvals/${rid}`, { decision }),
  result: (id: string, ref: string) => call<import("./types").Table>("GET", `${s(id)}/results/${ref}`),
  csvUrl: (id: string, ref: string) => `${s(id)}/results/${ref}/csv`,
  events: (id: string) => `${s(id)}/events`,
  async upload(id: string, files: File[]) {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await fetch(`${s(id)}/uploads`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`上传失败：${res.statusText}`);
    return (await res.json()) as { files: { name: string; size: number }[] };
  },
};
