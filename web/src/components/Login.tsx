import { useState } from "react";
import { api } from "../api";
import { Logo } from "./Logo";

export function Login({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPending(true);
    setError("");
    try {
      await api.login(name.trim(), password);
      onDone();
    } catch {
      setError("用户名或密码不对");
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="login">
      <form className="login-box" onSubmit={submit}>
        <div className="login-brand"><Logo size={26} /> FinHelm</div>
        <input autoFocus placeholder="用户名" autoComplete="username" value={name} onChange={(e) => setName(e.target.value)} />
        <input type="password" placeholder="密码" autoComplete="current-password" value={password}
          onChange={(e) => setPassword(e.target.value)} />
        {error && <div className="bad-text small">{error}</div>}
        <button className="btn dark wide" disabled={pending || !name.trim() || !password}>登录</button>
      </form>
    </div>
  );
}
