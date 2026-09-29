"""知识库：一个文档目录就是一个知识库（学 Bedrock Knowledge Bases / Dify：知识库单独存在，应用去挂载）。

项目在设置里写挂哪几个目录（DOCS_DIRS），search_docs 等工具只在挂上的库里找。索引跟着目录走，不跟着项目走：
两个项目挂同一个目录，用的是同一份索引；换个项目挂别的目录，就是另一份索引，不会互相删。

    <RAG_DIR>/
        parsed/          解析缓存（按文件内容），所有知识库共用
        embeddings/      嵌入缓存（按片文本），所有知识库共用
        collections/<文档目录的绝对路径 slug>/<配置>-<指纹>/     这个目录的索引

文档：目录下（含子目录）的 PDF，文档名 = 相对路径去掉扩展名（子目录里的写成 年报/2023）。
元数据：目录下可以放一个 metadata.jsonl，每行 {"doc": 文档名, "company": …, "period": …}，
会拼进每片的上下文前缀，list_docs 也列出来（Bedrock 是一份文档一个 metadata 文件，我们一个目录一个）。

第一次用到时才加载：文档没变就只读清单和片，有新文档才解析、编码。同一个进程只加载一次
（评测并发跑几道题时共用一份，不然每道题各占几百 MB）；会话中途往目录里加的文档，重启才看得到。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..memory import project_slug
from .document import Document
from .index import Hit, Index, IndexSpec, SearchSpec
from .parsers import ParsedCache

DOC_SUFFIXES = (".pdf",)
METADATA = "metadata.jsonl"

_loaded: dict[Path, Index] = {}
# 加载和查询都串行：嵌入、重排模型第一次加载不能两个线程一起来，GPU 上一次也只跑一个
_lock = threading.RLock()


@dataclass(frozen=True, slots=True)
class Collection:
    name: str                   # 给模型看的知识库名（默认是目录名）
    docs_dir: Path
    root: Path                  # RAG_DIR
    spec: IndexSpec = field(default_factory=IndexSpec)

    @property
    def folder(self) -> Path:
        return self.root / "collections" / project_slug(self.docs_dir) / self.spec.folder_name

    def files(self) -> dict[str, Path]:
        """文档名 → 文件。"""
        return {p.relative_to(self.docs_dir).with_suffix("").as_posix(): p
                for p in sorted(self.docs_dir.rglob("*")) if p.suffix.lower() in DOC_SUFFIXES and p.is_file()}

    def metadata(self) -> dict[str, dict]:
        path = self.docs_dir / METADATA
        if not path.is_file():
            return {}
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d.pop("doc")] = d
        return out

    def index(self, progress: bool = False) -> Index:
        with _lock:
            if self.folder not in _loaded:
                _loaded[self.folder] = Index.sync(self.spec, self.files(), self.folder, self.parsed,
                                                  self.root / "embeddings", self.metadata(), progress)
            return _loaded[self.folder]

    def search(self, query: str, search: SearchSpec, docs: Sequence[str] | None = None) -> list[Hit]:
        with _lock:
            return self.index().search(query, search, docs)

    def document(self, name: str) -> Document:
        """整份文档的解析结果（read_doc 读整页用）。name 要是索引里有的。"""
        entry = self.index().docs[name]
        doc = self.parsed.get(self.files()[name], self.spec.parser, entry["meta"], sha=entry["sha"])
        doc.name = name
        return doc

    @property
    def parsed(self) -> ParsedCache:
        return ParsedCache(self.root / "parsed")
