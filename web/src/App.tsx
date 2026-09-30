import { useCallback, useEffect, useState } from "react";
import { ChevronDown, PanelLeftOpen, PanelRight, SquarePen } from "lucide-react";
import { api, Unauthorized } from "./api";
import { LOGGED_OUT, useSession } from "./useSession";
import type { AgentState, Me, SessionSummary, Trashed } from "./types";
import { Sidebar } from "./components/Sidebar";
import { Composer } from "./components/Composer";
import { Thread } from "./components/Thread";
import { ResultPanel, sameTarget, type PanelTarget } from "./components/ResultPanel";
import { Logo } from "./components/Logo";
import { Login } from "./components/Login";
import { Menu } from "./components/Menu";
import { SessionContext } from "./sessionContext";
import { filesOf } from "./blocks";

const idFromHash = () => decodeURIComponent(location.hash.replace(/^#\/?s\//, "")) || null;
// 窄屏（手机）上侧栏和面板盖在对话上面
const narrow = () => window.innerWidth <= 900;

const SUGGESTIONS = ["库里有哪些表？", "2025 年哪个大区的销售额最高？", "按月看订单量的趋势"];

export default function App() {
  const [me, setMe] = useState<Me | null>(null);
  const loadMe = useCallback(() => api.me().then(setMe).catch(() => {}), []);
  useEffect(() => { loadMe(); }, [loadMe]);
  useEffect(() => {
    const out = () => setMe((m) => (m ? { ...m, user: null } : m));
    window.addEventListener(LOGGED_OUT, out);
    return () => window.removeEventListener(LOGGED_OUT, out);
  }, []);

  if (!me) return null;
  if (me.auth && !me.user) return <Login onDone={loadMe} />;
  return <Workspace me={me} onLogout={() => api.logout().finally(loadMe)} />;
}

function Workspace({ me, onLogout }: { me: Me; onLogout: () => void }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [trash, setTrash] = useState<Trashed[]>([]);
  const [current, setCurrent] = useState<string | null>(idFromHash);
  const [sidebar, setSidebar] = useState(() => !narrow());
  const [tabs, setTabs] = useState<PanelTarget[]>([]);
  const [active, setActive] = useState<PanelTarget | null>(null);
  const [uploads, setUploads] = useState<{ name: string; size: number }[]>([]);
  const [queued, setQueued] = useState<string | null>(null);     // 新对话的第一句：等连上再发
  const { view, run } = useSession(current);

  const guard = useCallback(<T,>(p: Promise<T>) => p.catch((e) => {
    if (e instanceof Unauthorized) window.dispatchEvent(new Event(LOGGED_OUT));
  }), []);
  const refresh = useCallback(() => guard(api.sessions().then(setSessions)), [guard]);
  const refreshTrash = useCallback(() => guard(api.trash().then(setTrash)), [guard]);
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
    setTabs([]);
    setActive(null);
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

  const showPanel = (t: PanelTarget) => {
    // 窄屏上打开面板时收起侧栏，给对话留地方（Claude desktop 也这样）
    if (!active && window.innerWidth < 1400) setSidebar(false);
    setTabs((ts) => (ts.some((x) => sameTarget(x, t)) ? ts : [...ts, t]));
    setActive(t);
  };
  const closeTab = (t: PanelTarget) => {
    const rest = tabs.filter((x) => !sameTarget(x, t));
    setTabs(rest);
    if (active && sameTarget(active, t)) setActive(rest.at(-1) ?? null);
  };

  const rename = async (id: string, title: string) => { await guard(api.rename(id, title)); refresh(); };
  const remove = async (id: string) => {
    if (!confirm("删除这个对话？可以在「最近删除」里恢复。")) return;
    await guard(api.remove(id).then(() => { if (id === current) open(null); }).catch((e) => alert(e.message)));
    refresh();
    refreshTrash();
  };
  const restore = async (id: string) => { await guard(api.restore(id)); refresh(); refreshTrash(); };

  const pendingAsk = !view.busy ? view.interrupted?.pending ?? null : null;
  const answer = (a: string) => current && run(() => (a ? api.send(current, a) : api.resume(current, "")));
  const files = view.items.flatMap(filesOf);
  const firstQuestion = view.items.find((it) => it.kind === "user")?.text.replace(/^\[用户上传了文件[^\]]*\]\n\n/, "") ?? "";
  const title = sessions.find((s) => s.id === current)?.title || firstQuestion.split("\n")[0] || "新对话";
  const busy = view.busy || !!queued;
  const landing = !current || (view.ready && view.items.length === 0 && !busy);
  const canUpload = view.info ? view.info.sandbox : true;
  const footer = <ContextInfo model={me.model} state={view.state} />;

  return (
    <SessionContext.Provider value={current}>
      <div className="app">
        {sidebar && (
          <Sidebar project={me.project} user={me.auth ? me.user : null} sessions={sessions} trash={trash}
            current={current} onSelect={open} onNew={() => open(null)} onCollapse={() => setSidebar(false)}
            onRename={rename} onDelete={remove} onRestore={restore} onShowTrash={refreshTrash} onLogout={onLogout} />
        )}

        <main className="main">
          <header className="topbar">
            {!sidebar && (
              <>
                <button className="icon-btn" title="展开侧栏" onClick={() => setSidebar(true)}><PanelLeftOpen size={16} /></button>
                <button className="icon-btn" title="新对话" onClick={() => open(null)}><SquarePen size={16} /></button>
              </>
            )}
            {!landing && current && (
              <Menu trigger={(toggle) => (
                <button className="title-btn" onClick={toggle}>
                  <span className="ellipsis">{title}</span><ChevronDown size={14} />
                </button>
              )} items={[
                { label: "重命名", onClick: () => { const t = prompt("重命名", title); if (t?.trim()) rename(current, t); } },
                { label: "压缩上下文", disabled: busy, onClick: () => run(() => api.compact(current)) },
                { label: "清空对话", disabled: busy, onClick: () => { api.reset(current).catch(() => {}); } },
                { label: "删除", danger: true, disabled: busy, onClick: () => remove(current) },
              ]} />
            )}
            <span className="spacer" />
            {view.error && <span className="faint small">{view.error}</span>}
            {view.results.length + files.length > 0 && (
              <button className={`icon-btn${active ? " on" : ""}`} title="结果"
                onClick={() => active ? setActive(null) : showPanel(tabs.at(-1) ??
                  (view.results.length ? { kind: "table", ref: view.results.at(-1)!.ref } : { kind: "file", url: files.at(-1)! }))}>
                <PanelRight size={16} />
              </button>
            )}
          </header>

          {landing ? (
            <div className="landing">
              <div className="greet"><Logo size={30} /><h1>今天想分析点什么？</h1></div>
              <Composer busy={busy} placeholder="问一个关于数据的问题…" uploads={uploads} canUpload={canUpload}
                autoFocus footer={footer} onSend={send} onStop={() => {}} onUpload={upload} />
              <div className="suggestions">
                {SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
              </div>
            </div>
          ) : (
            <>
              <Thread items={view.items} results={view.results} busy={busy} interrupted={view.interrupted}
                approvals={view.approvals}
                onOpenResult={(ref) => showPanel({ kind: "table", ref })}
                onOpenFile={(url) => showPanel({ kind: "file", url })}
                onAnswer={answer}
                onContinue={() => current && run(() => api.resume(current))}
                onApprove={(rid, d) => current && guard(api.approve(current, rid, d))} />
              <div className="composer-dock">
                <Composer busy={busy} disabled={!!pendingAsk} uploads={uploads} canUpload={canUpload} footer={footer}
                  placeholder={pendingAsk ? "请先回答上面的问题" : "接着问…"}
                  onSend={send} onStop={() => current && guard(api.stop(current))} onUpload={upload} />
              </div>
            </>
          )}
        </main>

        {active && current && (
          <ResultPanel session={current} results={view.results} tabs={tabs} active={active}
            onSelect={setActive} onCloseTab={closeTab} onClose={() => setActive(null)} />
        )}
      </div>
    </SessionContext.Provider>
  );
}

// 输入框下面右边：模型名 + 上下文用量小圆环
function ContextInfo({ model, state }: { model: string; state: AgentState | null }) {
  const total = state?.context_window;
  const used = state?.context_tokens ?? 0;
  const pct = total && used ? Math.min(1, used / total) : 0;
  const r = 6, c = 2 * Math.PI * r;
  return (
    <span className="context-info">
      <span>{model}</span>
      {pct > 0 && (
        <span className="meter" title={`上下文 ${used.toLocaleString()} / ${total!.toLocaleString()} token（${(pct * 100).toFixed(1)}%）`}>
          <svg width="15" height="15" viewBox="0 0 15 15">
            <circle cx="7.5" cy="7.5" r={r} className="meter-bg" />
            <circle cx="7.5" cy="7.5" r={r} className="meter-fg" strokeDasharray={`${Math.max(c * pct, 1.5)} ${c}`}
              transform="rotate(-90 7.5 7.5)" />
          </svg>
        </span>
      )}
    </span>
  );
}
