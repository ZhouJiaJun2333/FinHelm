"""沙箱的宿主机这一头：起内核进程、收发消息、超时就杀掉重来。Python 和 R 的内核共用。

内核默认跑在 Docker 里：断网、只读根目录、非 root、限 CPU / 内存 / 进程数，只挂载会话的工作目录。
测试用 local() 直接起一个本机进程，协议一样。第一次执行时才启动，会话结束时 close()。
协议见 tools/python/kernel.py 开头：按行说 JSON，内核要 SQL 结果时回头来问（need / data）。

两道超时：内核里的软超时打断用户代码，变量还在；软超时拦不住（卡在 C 代码里）时，
宿主机的硬超时杀掉整个内核，变量就都没了。
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, Field

from ..core.tools import Tool, ToolOutput

STARTUP_TIMEOUT_S = 60           # 冷启动容器 + 加载 pandas / meta
HARD_TIMEOUT_GRACE_S = 10
MOUNT = "/opt/finhelm"
FIGURES_HEADER = "图表已保存（用户能看到这些文件）："

# 给一个结果编号，返回内核要的数据：{"columns", "rows", "dates", "truncated"} 或 {"error"}
Resolve = Callable[[str], dict[str, Any]]
# 内核 save_result(表) 时存进结果仓库：收 {"title", "columns", "rows"}，回 {"ref", "rows", "truncated"} 或 {"error"}
Save = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """一种语言的内核：files[0] 是入口脚本，其余（比如模板库）一起只读挂进容器。"""

    files: tuple[Path, ...]
    command: tuple[str, ...]              # 容器里怎么跑入口脚本，比如 ("python", "-u")
    env: tuple[tuple[str, str], ...] = ()


class SandboxUnavailable(RuntimeError):
    """内核起不来：Docker 没开、镜像没建……要人处理，不是模型改代码能解决的。"""


@dataclass(frozen=True, slots=True)
class Execution:
    output: str = ""                 # 打印出来的
    value: str | None = None         # 最后一行表达式的值（R 的可见值直接打印进 output）
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
        save: Save | None = None,
    ) -> None:
        self.command = command
        self.work_dir = work_dir
        self.resolve = resolve
        self.save = save
        self.timeout_s = timeout_s
        self.grace_s = grace_s               # 软超时之后再等多久才硬杀
        self.kill_command = kill_command     # docker：杀掉 docker run 客户端不会停容器
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._log: Any = None
        self._cancelled = False

    # ------------------------------------------------------------ 构造
    @classmethod
    def docker(
        cls, image: str, kernel: KernelSpec, work_dir: Path, resolve: Resolve, *,
        timeout_s: float = 60, memory: str = "2g", cpus: float = 2, save: Save | None = None,
        data_dir: Path | None = None,
    ) -> "Sandbox":
        """data_dir：场景包自带的数据，只读挂到容器的 /data/。"""
        work_dir = work_dir.resolve()
        name = f"finhelm-sandbox-{secrets.token_hex(4)}"
        mounts = [arg for f in kernel.files for arg in
                  ("--mount", f"type=bind,source={f.resolve()},target={MOUNT}/{f.name},readonly")]
        env = [arg for key, value in (("HOME", "/tmp"), *kernel.env) for arg in ("-e", f"{key}={value}")]
        command = [
            "docker", "run", "-i", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "--memory", memory, "--memory-swap", memory, "--cpus", str(cpus), "--pids-limit", "128",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
            "--mount", f"type=bind,source={work_dir},target=/work", *mounts,
            *(("--mount", f"type=bind,source={data_dir.resolve()},target=/data,readonly") if data_dir else ()),
            "-w", "/work", *env,
            image, *kernel.command, f"{MOUNT}/{kernel.files[0].name}",
        ]
        return cls(command, work_dir, resolve, timeout_s=timeout_s,
                   kill_command=["docker", "kill", name], save=save)

    @classmethod
    def local(cls, kernel: KernelSpec, work_dir: Path, resolve: Resolve, *, timeout_s: float = 60,
              grace_s: float = HARD_TIMEOUT_GRACE_S, save: Save | None = None) -> "Sandbox":
        """不隔离，只给测试用：用本机的 Python 跑 Python 内核。"""
        return cls([sys.executable, "-u", str(kernel.files[0])], work_dir, resolve,
                   timeout_s=timeout_s, grace_s=grace_s, cwd=work_dir, save=save)

    # ------------------------------------------------------------ 执行
    def run(self, code: str) -> Execution:
        self._ensure_started()
        self._cancelled = False
        deadline = time.monotonic() + self.timeout_s + self.grace_s
        try:
            self._send({"op": "exec", "code": code, "timeout": self.timeout_s})
            while True:
                msg = self._receive(deadline)
                if msg["op"] == "need":
                    self._send({"op": "data", **self._payload(msg["ref"])})
                elif msg["op"] == "save":
                    self._send({"op": "saved", **self._store(msg)})
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
            if self._cancelled:
                return Execution(error="用户中断了执行，内核已重启。", restarted=True)
            return Execution(error="内核意外退出（常见原因是内存超限）。" + (f"\n{detail}" if detail else ""),
                             restarted=True)

    def _payload(self, ref: str) -> dict[str, Any]:
        try:
            return {"ref": ref, **self.resolve(ref)}
        except Exception as exc:  # noqa: BLE001 —— 内核那头在等回话，不能让它干等
            return {"ref": ref, "error": f"{type(exc).__name__}: {exc}"}

    def _store(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self.save is None:
            return {"error": "这里不能存结果编号。"}
        try:
            return self.save(msg)
        except Exception as exc:  # noqa: BLE001 —— 同上，内核在等
            return {"error": f"{type(exc).__name__}: {exc}"}

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

    def cancel(self) -> None:
        """别的线程调：杀掉正在执行的内核，run() 那边读到 EOF 返回「用户中断」。收尾（close）留给 run() 做。"""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._cancelled = True
        if self.kill_command:
            subprocess.run(self.kill_command, capture_output=True, timeout=30)
        if proc.poll() is None:
            proc.kill()

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


# ================================================================ 工具
class SandboxTool(Tool):
    """run_python / run_r 的共同部分：把代码交给内核，把执行结果整理成模型看的文字。

    不是 rerunnable：内核有状态，重跑一次会改变里面的变量。
    """

    language: ClassVar[str]
    # 变量找不到的报错（Python 的 NameError、R 的 object 'x' not found）：可能是内核重启过
    missing_name: ClassVar[re.Pattern]

    class Args(BaseModel):
        code: str = Field(description="要执行的代码")

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def cancel(self) -> None:
        self.sandbox.cancel()

    def run(self, args: Args) -> ToolOutput:
        ex = self.sandbox.run(args.code)
        return ToolOutput(self.render(ex), _summarize(ex), details=ex, is_error=ex.error is not None)

    def render(self, ex: Execution) -> str:
        parts = []
        if ex.output.strip():
            parts.append(ex.output.rstrip())
        if ex.value is not None:
            parts.append(ex.value)
        if ex.figures:
            parts.append(FIGURES_HEADER + "\n" + "\n".join(f"- {p}" for p in ex.figures))
        if ex.error:
            parts.append(f"出错了：\n{ex.error.rstrip()}")
        restart = f"{self.language} 内核重启过，之前定义的变量都没了。需要的话重新运行定义它们的代码，数据用 load_result 重新取。"
        if ex.restarted:
            parts.append(restart)
        elif ex.error and self.missing_name.search(ex.error):
            parts.append("如果这个变量是之前的调用里定义的：内核可能重启过（超时、恢复会话），重新运行定义它的代码。")
        return "\n\n".join(parts) or "执行成功，没有输出。要看结果就打印出来，或者把表达式放在最后一行。"


def saved_figures(content: str) -> list[Path]:
    """从模型看到的那份文字里找回产出的文件（render 的反向）。历史里只存了文字，重新打开会话时靠它重建预览。"""
    _, found, rest = content.partition(FIGURES_HEADER + "\n")
    if not found:
        return []
    lines = rest.split("\n\n", 1)[0].splitlines()
    return [Path(line[2:]) for line in lines if line.startswith("- ")]


def _summarize(ex: Execution) -> str:
    if ex.error:
        return "报错：" + ex.error.strip().splitlines()[-1]
    text = f"输出 {(ex.output + (ex.value or '')).count(chr(10)) + 1} 行"
    return text + (f"，{len(ex.figures)} 张图" if ex.figures else "")
