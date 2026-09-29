import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Archive, Eraser, MoreHorizontal, PanelLeftOpen, PanelRight, SquarePen } from "lucide-react";
import { api } from "./api";
import { useSession } from "./useSession";
import type { AgentState, SessionSummary } from "./types";
import { Sidebar } from "./components/Sidebar";
import { Composer } from "./components/Composer";
import { ApprovalCard, InterruptedCard, ItemView } from "./components/Messages";
import { ResultPanel, type PanelTarget } from "./components/ResultPanel";
import { Logo } from "./components/Logo";
import { SessionContext } from "./sessionContext";

const idFromHash = () => decodeURIComponent(location.hash.replace(/^#\/?s\//, "")) || null;

// 窄屏（手机）上侧栏和面板是盖在对话上面的
const narrow = () => window.innerWidth <= 900;

const SUGGESTIONS = ["库里有哪些表？各有多少行？", "2025 年哪个大区的销售额最高？", "按月看一下订单量的趋势，画张图"];

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [current, setCurrent] = useState<string | null>(idFromHash);
  const [sidebar, setSidebar] = useState(() => !narrow());
  const [panel, setPanel] = useState<PanelTarget | null>(null);
  const [uploads, setUploads] = useState<{ name: string; size: number }[]>([]);
  const [queued, setQueued] = useState<string | null>(null);     // 新对话的第一句：等连上再发
  const [menu, setMenu] = useState(false);
  const { view, run } = useSession(current);

  const refresh = useCallback(() => api.sessions().then(setSessions).catch(() => {}), []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    const onHash = () => setCurrent(idFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  // 一轮开始（标题有了）和结束时刷新会话列表
  useEffect(() => { if (view.ready) refresh(); }, [view.ready, view.busy, refresh]);

  const open = (id: string | null) => {
    location.hash = id ? `/s/${id}` : "";
    setCurrent(id);
    setPanel(null);
    setUploads([]);
    if (narrow()) setSidebar(false);
  };

  const ensureSession = async () => {
    if (current) return current;
    const { id } = await api.create();
    open(id);
    return id;
  };

  const send = async (text: string) => {
    if (!current) {
      setQueued(text);
      await ensureSession();
      return;
    }
    setUploads([]);
    run(() => api.send(current, text));
  };

  useEffect(() => {
    if (queued && current && view.ready) {
      setQueued(null);
      setUploads([]);
      run(() => api.send(current, queued));
    }
  }, [queued, current, view.ready, run]);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    const id = await ensureSession();
    const { files: saved } = await api.upload(id, files);
    setUploads((u) => [...u, ...saved]);
  };

  const pendingAsk = view.interrupted?.pending?.call_id ?? null;
  const answer = (a: string) => current && run(() => pendingAsk && a === "" ? api.resume(current, "") : api.send(current, a));
  const figures = useMemo(() => view.items.flatMap((it) =>
    it.kind === "tool" && it.details?.kind === "execution" ? it.details.figures : []), [view.items]);
  const showPanel = (t: PanelTarget | null) => {
    // 窄屏上打开面板时收起侧栏，给对话留地方（Claude desktop 也这样）
    if (t && !panel && window.innerWidth < 1400) setSidebar(false);
    setPanel(t);
  };
  const openResult = (ref: string) => showPanel({ kind: "table", ref });
  const title = view.items.find((it) => it.kind === "user")?.text.replace(/^\[用户上传了文件[^\]]*\]\n\n/, "") ?? "";
  const busy = view.busy || !!queued;
  const landing = !current || (view.ready && view.items.length === 0 && !busy);
  const canUpload = view.info ? view.info.sandbox : true;

  return (
    <SessionContext.Provider value={current}>
    <div className={`app${sidebar ? "" : " no-sidebar"}${panel ? " with-panel" : ""}`}>
      {sidebar && <Sidebar sessions={sessions} current={current} onSelect={open} onNew={() => open(null)}
                           onCollapse={() => setSidebar(false)} />}

      <main className="main">
        <header className="topbar">
          {!sidebar && (
            <>
              <button className="icon-btn" title="展开侧栏" onClick={() => setSidebar(true)}><PanelLeftOpen size={17} /></button>
              <button className="icon-btn" title="新对话" onClick={() => open(null)}><SquarePen size={17} /></button>
            </>
          )}
          <div className="topbar-title">{landing ? "" : title.split("\n")[0] || "新对话"}</div>
          {view.state && !landing && <StatusPill state={view.state} busy={busy} />}
          {view.state && !landing && <ContextMeter state={view.state} window={view.info?.context_window ?? null} />}
          {view.results.length + figures.length > 0 && (
            <button className={`btn ghost${panel ? " active" : ""}`} onClick={() => showPanel(panel ? null :
                view.results.length ? { kind: "table", ref: view.results.at(-1)!.ref } : { kind: "figure", url: figures.at(-1)! })}>
              <PanelRight size={15} /> 结果 {view.results.length + figures.length}
            </button>
          )}
          {current && !landing && (
            <div className="menu-wrap">
              <button className="icon-btn" onClick={() => setMenu(!menu)}><MoreHorizontal size={17} /></button>
              {menu && (
                <div className="menu" onMouseLeave={() => setMenu(false)}>
                  <button disabled={busy} onClick={() => { setMenu(false); run(() => api.compact(current)); }}>
                    <Archive size={14} /> 压缩上下文
                  </button>
                  <button disabled={busy} onClick={() => { setMenu(false); api.reset(current).catch(() => {}); }}>
                    <Eraser size={14} /> 清空对话
                  </button>
                </div>
              )}
            </div>
          )}
        </header>

        {view.error && <div className="banner">{view.error}</div>}

        {landing ? (
          <div className="landing">
            <div className="greet"><Logo size={34} /><h1>今天想分析点什么？</h1></div>
            <Composer busy={busy} placeholder="问一个关于数据的问题，或者上传文件…" uploads={uploads} canUpload={canUpload}
              autoFocus onSend={send} onStop={() => {}} onUpload={upload} />
            <div className="suggestions">
              {SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
            </div>
            {view.info && (
              <div className="env muted small">
                {view.info.model} · {view.info.database ? "已连数据库" : "未连数据库"} · {view.info.sandbox ? "沙箱可用" : "无沙箱"}
                {view.info.skills.length > 0 && ` · 技能 ${view.info.skills.join("、")}`}
                {view.info.mcp.length > 0 && ` · MCP ${view.info.mcp.join("、")}`}
              </div>
            )}
          </div>
        ) : (
          <>
            <Conversation>
              {view.items.map((it) => (
                <ItemView key={it.id} item={it} results={view.results} pendingAsk={busy ? null : pendingAsk}
                  onOpenResult={openResult} onAnswer={answer} />
              ))}
              {busy && view.state?.status === "thinking" && !lastIsStreaming(view.items) && (
                <div className="working"><Logo size={16} /><span className="shimmer">思考中…</span></div>
              )}
              {view.approvals.map((a) => (
                <ApprovalCard key={a.id} req={a} onDecide={(d) => current && api.approve(current, a.id, d)} />
              ))}
              {!busy && view.interrupted && !view.interrupted.pending && (
                <InterruptedCard turn={view.interrupted} onContinue={() => current && run(() => api.resume(current))} />
              )}
            </Conversation>
            <div className="composer-dock">
              <Composer busy={busy} uploads={uploads} canUpload={canUpload}
                placeholder={pendingAsk && !busy ? "写下你的回答，或者点上面的选项…" : "接着问…"}
                onSend={send} onStop={() => current && api.stop(current)} onUpload={upload} />
            </div>
          </>
        )}
      </main>

      {panel && current && (
        <ResultPanel session={current} results={view.results} figures={figures} target={panel}
          onSelect={setPanel} onClose={() => setPanel(null)} />
      )}
    </div>
    </SessionContext.Provider>
  );
}

function lastIsStreaming(items: ReturnType<typeof useSession>["view"]["items"]) {
  const last = items.at(-1);
  return last?.kind === "assistant" && last.streaming;
}

// 贴着底部时新内容来了自动往下滚；用户往上翻了就不打扰
function Conversation({ children }: { children: React.ReactNode }) {
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });
  return (
    <div className="scroll" ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}>
      <div className="thread">{children}</div>
    </div>
  );
}

function StatusPill({ state, busy }: { state: AgentState; busy: boolean }) {
  const [text, cls] =
    state.status === "asking" ? ["等你回答", "ask"]
    : !busy ? ["空闲", "idle"]
    : state.status === "tools" ? ["执行工具", "work"]
    : ["思考中", "work"];
  return <span className={`pill ${cls}`}><span className="pill-dot" />{text}{busy && state.step ? ` · 第 ${state.step} 步` : ""}</span>;
}

function ContextMeter({ state, window }: { state: AgentState; window: number | null }) {
  const total = state.context_window ?? window;
  if (!total || !state.context_tokens) return null;
  const pct = Math.min(1, state.context_tokens / total);
  const r = 7, c = 2 * Math.PI * r;
  return (
    <span className="meter" title={`上下文 ${state.context_tokens.toLocaleString()} / ${total.toLocaleString()} token`}>
      <svg width="18" height="18" viewBox="0 0 18 18">
        <circle cx="9" cy="9" r={r} className="meter-bg" />
        <circle cx="9" cy="9" r={r} className="meter-fg" strokeDasharray={`${c * pct} ${c}`} transform="rotate(-90 9 9)" />
      </svg>
      {(pct * 100).toFixed(pct < 0.1 ? 1 : 0)}%
    </span>
  );
}
