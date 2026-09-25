"""宿主机这一头：起内核进程、收发消息、超时就杀掉重来。

内核默认跑在 Docker 里：断网、只读根目录、非 root、限 CPU / 内存 / 进程数，只挂载会话的工作目录。
测试用 local() 直接起一个本机进程，协议一样。第一次执行时才启动，会话结束时 close()。

两道超时：内核里的软超时打断用户代码，变量还在；软超时拦不住（卡在 C 代码里）时，
宿主机的硬超时杀掉整个内核，变量就都没了。
"""

from __future__ import annotations

import json
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KERNEL = Path(__file__).with_name("kernel.py")
STARTUP_TIMEOUT_S = 60           # 冷启动容器 + import pandas
HARD_TIMEOUT_GRACE_S = 10

# 给一个结果编号，返回内核要的数据：{"columns", "rows", "dates", "truncated"} 或 {"error"}
Resolve = Callable[[str], dict[str, Any]]


class SandboxUnavailable(RuntimeError):
    """内核起不来：Docker 没开、镜像没建……要人处理，不是模型改代码能解决的。"""


@dataclass(frozen=True, slots=True)
class Execution:
    output: str = ""                 # print 出来的
    value: str | None = None         # 最后一行表达式的值
    error: str | None = None
    figures: list[Path] = field(default_factory=list)
    restarted: bool = False          # 内核被杀掉重启了，之前的变量都没了


class Sandbox:
    def __init__(
        self,
        command: list[str],
        work_dir: Path,
        resolve: Resolve,
        *,
        timeout_s: float = 60,
        grace_s: float = HARD_TIMEOUT_GRACE_S,
        kill_command: list[str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.command = command
        self.work_dir = work_dir
        self.resolve = resolve
        self.timeout_s = timeout_s
        self.grace_s = grace_s               # 软超时之后再等多久才硬杀
        self.kill_command = kill_command     # docker：杀掉 docker run 客户端不会停容器
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._log: Any = None

    # ------------------------------------------------------------ 构造
    @classmethod
    def docker(
        cls, image: str, work_dir: Path, resolve: Resolve, *,
        timeout_s: float = 60, memory: str = "2g", cpus: float = 2,
    ) -> "Sandbox":
        work_dir = work_dir.resolve()
        name = f"finhelm-sandbox-{secrets.token_hex(4)}"
        command = [
            "docker", "run", "-i", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "--memory", memory, "--memory-swap", memory, "--cpus", str(cpus), "--pids-limit", "128",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
            "--mount", f"type=bind,source={work_dir},target=/work",
            "--mount", f"type=bind,source={KERNEL.resolve()},target=/opt/finhelm/kernel.py,readonly",
            "-w", "/work", "-e", "MPLCONFIGDIR=/tmp", "-e", "HOME=/tmp",
            image, "python", "-u", "/opt/finhelm/kernel.py",
        ]
        return cls(command, work_dir, resolve, timeout_s=timeout_s,
                   kill_command=["docker", "kill", name])

    @classmethod
    def local(cls, work_dir: Path, resolve: Resolve, *, timeout_s: float = 60,
              grace_s: float = HARD_TIMEOUT_GRACE_S) -> "Sandbox":
        """不隔离，只给测试用。"""
        return cls([sys.executable, "-u", str(KERNEL)], work_dir, resolve,
                   timeout_s=timeout_s, grace_s=grace_s, cwd=work_dir)

    # ------------------------------------------------------------ 执行
    def run(self, code: str) -> Execution:
        self._ensure_started()
        deadline = time.monotonic() + self.timeout_s + self.grace_s
        try:
            self._send({"op": "exec", "code": code, "timeout": self.timeout_s})
            while True:
                msg = self._receive(deadline)
                if msg["op"] == "need":
                    self._send({"op": "data", **self._payload(msg["ref"])})
                elif msg["op"] == "done":
                    return Execution(
                        output=msg["output"], value=msg["value"], error=msg["error"],
                        figures=[self.work_dir / f for f in msg["figures"]],
                    )
        except TimeoutError:
            self.close()
            return Execution(error=f"执行超过 {self.timeout_s:g} 秒没有返回，内核被强制重启。",
                             restarted=True)
        except EOFError:
            detail = self._log_tail()
            self.close()
            return Execution(error="内核意外退出（常见原因是内存超限）。" + (f"\n{detail}" if detail else ""),
                             restarted=True)

    def _payload(self, ref: str) -> dict[str, Any]:
        try:
            return {"ref": ref, **self.resolve(ref)}
        except Exception as exc:  # noqa: BLE001 —— 内核那头在等回话，不能让它干等
            return {"ref": ref, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------ 进程
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure_started(self) -> None:
        if self.running:
            return
        self.close()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._log = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                cwd=self.cwd, text=True, encoding="utf-8", bufsize=1,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailable(f"启动不了沙箱：找不到 {self.command[0]}。") from exc
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()
        try:
            self._receive(time.monotonic() + STARTUP_TIMEOUT_S)
        except (TimeoutError, EOFError) as exc:
            detail = self._log_tail()
            self.close()
            raise SandboxUnavailable(f"沙箱没能启动。{detail}") from exc

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: queue.Queue) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _send(self, msg: dict[str, Any]) -> None:
        try:
            self._proc.stdin.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:          # 管道断了 / 已经关了
            raise EOFError from exc

    def _receive(self, deadline: float) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError from None
        if line is None:
            raise EOFError
        return json.loads(line)

    def _log_tail(self, chars: int = 1500) -> str:
        if self._log is None:
            return ""
        self._log.seek(0)
        return self._log.read().decode("utf-8", errors="replace")[-chars:].strip()

    def close(self) -> None:
        """杀掉内核。下次 run() 会重新启动一个空的。"""
        if self._proc is not None:
            if self.kill_command and self._proc.poll() is None:
                subprocess.run(self.kill_command, capture_output=True, timeout=30)
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait(timeout=30)
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
