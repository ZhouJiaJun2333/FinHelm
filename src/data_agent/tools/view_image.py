"""view_image：把一张图发给模型看（学 Codex 的 view_image）。

主要用途：模型自己写代码画的图，交付前看一眼。文字重叠、被裁切、单位写错，不看图是发现不了的。
主模型按需「拉」图，不另设审图员（Claude Code、pi、Codex 都是这样）。

只能看会话工作目录（inputs/、figures/）和场景包数据目录（/data/）下的文件。太大的图缩小后再发，原文件不动。
rerunnable：只读，旧结果可以被上下文清理掉，要的话再看一次。
模型不能看图时不注册这个工具（app.py），提示词里「交付前看一眼」那句也跟着消失。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from ..core.messages import Image
from ..core.tools import Tool, ToolOutput
from .paths import SandboxPaths

# 长边上限。Anthropic 建议不超过 1568，再大服务端也会缩，白花上传；R 出的 1600×1000 缩一点点
MAX_EDGE = 1568
# 能原样发的格式，其余（TIFF、BMP…）转成 PNG
MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}


class ViewImageTool(Tool):
    name = "view_image"
    description = (
        "查看一张图片（PNG、JPG 等），图片会直接发给你看。用来检查自己画的图：文字有没有重叠、被裁切，"
        "图例、坐标轴标签、单位、数字对不对；也能看用户上传的图片。"
        "只能看工作目录和 /data/ 下的文件，路径写工具返回的路径或相对路径（比如 figures/fig-1.png）。"
        "PDF、SVG 看不了：另存一份 PNG 再看。"
    )
    rerunnable = True

    class Args(BaseModel):
        path: str = Field(description="图片路径，比如 figures/fig-1.png")

    def __init__(self, paths: SandboxPaths) -> None:
        self.paths = paths
        self.work_dir = paths.work_dir

    def run(self, args: Args) -> ToolOutput:
        try:
            path = self.paths.resolve(args.path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{exc}{self._listing()}") from None
        name = self.paths.display(path)
        raw = path.read_bytes()
        try:
            with PILImage.open(io.BytesIO(raw)) as img:
                image, note = self._encode(img, raw)
        except UnidentifiedImageError:
            raise ValueError(f"{name} 不是能看的图片格式。PDF、SVG 请另存一份 PNG 再看。") from None
        return ToolOutput(f"{name}（{note}）", summary=f"看了 {name}", images=(image,))

    def _listing(self) -> str:
        """找不到时告诉模型有哪些图，它多半是把文件名记错了。"""
        found = sorted(p.relative_to(self.work_dir).as_posix()
                       for folder in ("figures", "inputs") if (self.work_dir / folder).is_dir()
                       for p in (self.work_dir / folder).iterdir()
                       if p.is_file() and not p.name.startswith("."))
        return f"工作目录里有：{'、'.join(found[:30])}" if found else "figures/ 和 inputs/ 下都没有文件。"

    @staticmethod
    def _encode(img: PILImage.Image, raw: bytes) -> tuple[Image, str]:
        """缩到 MAX_EDGE 以内。没缩、格式也能直接发就发原文件的字节。"""
        width, height = img.size
        media_type = MEDIA_TYPES.get(img.format or "")
        fmt = "JPEG" if img.format == "JPEG" else "PNG"
        note = f"{width}×{height}"
        if max(width, height) <= MAX_EDGE and media_type:
            data = raw
        else:
            img.thumbnail((MAX_EDGE, MAX_EDGE), PILImage.Resampling.LANCZOS)
            if max(width, height) > MAX_EDGE:
                note += f"，缩小到 {img.width}×{img.height} 给你看"
            if fmt == "JPEG" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif fmt == "PNG" and img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                img = img.convert("RGBA")
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            data, media_type = buffer.getvalue(), MEDIA_TYPES[fmt]
        image = Image(media_type, base64.b64encode(data).decode("ascii"), img.width, img.height)
        return image, note
