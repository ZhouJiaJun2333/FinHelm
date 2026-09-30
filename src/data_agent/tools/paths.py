"""模型写的文件路径 → 宿主机上的路径。view_image、read_file 共用。

模型写的路径有好几种来源：沙箱里工作目录挂在 /work、场景包的数据挂在 /data（只读），从自己代码里抄来的
常带着它们；run_python 报图的时候给的是宿主机的绝对路径；也可能写相对路径（按工作目录算）。
只放行这两个目录下的文件。
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

CONTAINER_WORK = PurePosixPath("/work")
CONTAINER_DATA = PurePosixPath("/data")


class SandboxPaths:
    def __init__(self, work_dir: Path, data_dir: Path | None = None, figures_dir: Path | None = None) -> None:
        self.work_dir = work_dir.resolve()
        self.data_dir = data_dir.resolve() if data_dir is not None else None
        # 子 Agent 的沙箱把 figures/ 挂到了别处（Sandbox.docker 的 figures_dir），它说的 figures/a.png 在这里
        self.figures_dir = figures_dir.resolve() if figures_dir is not None else None

    @property
    def allowed(self) -> str:
        return "工作目录（inputs/、figures/）" + ("和 /data/" if self.data_dir else "")

    def resolve(self, raw: str) -> Path:
        """找不到文件抛 FileNotFoundError，不在允许的目录里抛 ValueError。"""
        raw = raw.strip()
        posix = PurePosixPath(raw)
        for mount, root in ((CONTAINER_WORK, self.work_dir), (CONTAINER_DATA, self.data_dir)):
            if root is not None and posix.is_relative_to(mount):
                path = root / posix.relative_to(mount)
                break
        else:
            path = Path(raw)
            path = path if path.is_absolute() else self.work_dir / path
        path = path.resolve()                  # 先解开 ..，再判断在不在允许的目录里
        shared = self.work_dir / "figures"
        if self.figures_dir is not None and path.is_relative_to(shared) and not path.is_relative_to(self.figures_dir):
            path = self.figures_dir / path.relative_to(shared)
        if not any(root is not None and path.is_relative_to(root) for root in (self.work_dir, self.data_dir)):
            raise ValueError(f"只能读{self.allowed}下的文件。")
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {raw}。")
        return path

    def display(self, path: Path) -> str:
        """给模型看的写法：工作目录里的写相对路径（figures/fig-1.png），数据目录里的写 /data/manual.md。"""
        if self.data_dir is not None and path.is_relative_to(self.data_dir):
            return (CONTAINER_DATA / path.relative_to(self.data_dir).as_posix()).as_posix()
        if self.figures_dir is not None and path.is_relative_to(self.figures_dir):
            return f"figures/{path.relative_to(self.figures_dir).as_posix()}"
        return path.relative_to(self.work_dir).as_posix()
