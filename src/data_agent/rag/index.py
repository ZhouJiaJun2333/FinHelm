"""索引和检索。

    IndexSpec   建索引时的配置：解析器、分片、嵌入模型。改了要重建，按指纹各存一个目录，几套可以并存
    SearchSpec  查询时的配置：用哪几路召回、候选多少、RRF 参数、重排模型、返回几条。改了立刻生效

增量更新：索引是一批文档的镜像，Index.sync 只处理新增、改过、删掉的文档（按文件内容哈希判断），
没变的文档连片带向量原样留着。三层缓存各管一段：

    解析    ParsedCache      按文件内容哈希：同名换了内容会重新解析
    嵌入    EmbeddingCache   按「模型 + 片文本」：改了分片或解析规则，文字没变的片不用重新编码
    BM25    不存盘，加载时现建：全局 IDF 总是最新的，15 万片几秒钟

    <索引目录>/                 放在哪由调用方定（知识库放在 collections/<文档目录>/<spec.folder_name>）
        spec.json       这个索引是怎么建的
        manifest.json   每份文档：内容哈希、大小和修改时间（没变就不用重新算哈希，学 git 的 index）、元数据、页数、片数
        docs/<文档>.jsonl   这份文档的片（id 从 0 数，加载时再按文档名顺序排成全局 id）
        docs/<文档>.npy     这份文档每片一个向量（float16，归一化）；没配嵌入模型就没有

换嵌入模型、分片方式、解析规则（版本号）会改指纹，是一个新的索引目录；但解析和嵌入缓存照样能用上。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bm25 import BM25
from .chunk import Chunk, ChunkSpec, chunk_document
from .dense import Embedder, top_k
from .embed_cache import EmbeddingCache
from .parsers import PARSERS, ParsedCache, file_sha
from .rerank import Reranker

CHUNK_VERSION = 1        # 改了分片逻辑就加一：旧索引的指纹对不上，会重建


@dataclass(frozen=True, slots=True)
class IndexSpec:
    parser: str = "pymupdf"
    chunk: ChunkSpec = field(default_factory=ChunkSpec)
    embedder: str = ""               # 空 = 只建 BM25
    max_length: int = 1024           # 嵌入模型一次最多读多少 token（超出截断）

    @property
    def fingerprint(self) -> str:
        d = {**asdict(self), "parser_version": PARSERS[self.parser][1], "chunk_version": CHUNK_VERSION}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    @property
    def label(self) -> str:
        model = self.embedder.split("/")[-1] if self.embedder else "bm25only"
        return f"{self.parser}-{self.chunk.label}-{model}"

    @property
    def folder_name(self) -> str:
        """索引目录名：同一批文档，不同配置的索引各一个目录。"""
        return f"{self.chunk.label}-{self.fingerprint}"


@dataclass(frozen=True, slots=True)
class SearchSpec:
    retrievers: tuple[str, ...] = ("bm25", "dense")
    candidates: int = 50             # 每一路召回多少条，融合后也留这么多给重排
    rrf_k: int = 60                  # RRF：score = Σ 1 / (rrf_k + 名次)
    reranker: str = ""               # 空 = 不重排
    top_k: int = 10

    @property
    def label(self) -> str:
        return "+".join(self.retrievers) + (f"→{self.reranker.split('/')[-1]}" if self.reranker else "")


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int]            # 在每一路召回里排第几（1 起），没召回到就不在里面


@dataclass(slots=True)
class Changes:
    """上一次 sync 改了哪些文档（文档名）。"""
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"新增 {len(self.added)}、更新 {len(self.changed)}、删除 {len(self.removed)}"


class Index:
    def __init__(self, spec: IndexSpec, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        self.spec, self.chunks, self.vectors = spec, chunks, vectors
        self.changes = Changes()
        self.docs: dict[str, dict] = {}              # 从清单读的：文档名 → 内容哈希、元数据、页数、片数
        self.bm25 = BM25(c.search_text for c in chunks)
        self._by_doc: dict[str, list[int]] = {}
        for c in chunks:
            self._by_doc.setdefault(c.doc, []).append(c.id)

    # ------------------------------------------------------------ 建、读
    @classmethod
    def sync(cls, spec: IndexSpec, pdfs: Mapping[str, Path], folder: Path, parsed: ParsedCache,
             embeddings: Path, meta: Mapping[str, dict] | None = None, progress: bool = False) -> "Index":
        """让 folder 里的索引和这批文档（文档名 → 文件）一致：新增、改过（内容或元数据变了）的
        重新解析（有缓存）→ 分片 → 嵌入（embeddings 目录下的缓存），不在这批里的删掉，没变的不动。
        每处理完一份就更新清单，中途断了下次从断的地方接着做。文档名可以带 /（子目录）。"""
        (folder / "docs").mkdir(parents=True, exist_ok=True)
        (folder / "spec.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=1), encoding="utf-8")
        manifest = _read_manifest(folder)
        changes = Changes()

        for name in sorted(set(manifest) - set(pdfs)):
            del manifest[name]
            _write_manifest(folder, manifest)
            for suffix in (".jsonl", ".npy"):
                (folder / "docs" / f"{name}{suffix}").unlink(missing_ok=True)
            changes.removed.append(name)

        cache = EmbeddingCache(embeddings, Embedder(spec.embedder, spec.max_length)) if spec.embedder else None
        for n, (name, pdf) in enumerate(sorted(pdfs.items()), 1):
            stat = pdf.stat()
            old = manifest.get(name)
            same_file = old and old["size"] == stat.st_size and old["mtime_ns"] == stat.st_mtime_ns
            sha = old["sha"] if same_file else file_sha(pdf)
            doc_meta = (meta or {}).get(name) or {}
            if old and old["sha"] == sha and old["meta"] == doc_meta:
                if not same_file:                           # 只是时间戳变了（复制、touch）：记下新的，下次不用再算哈希
                    manifest[name] = {**old, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                    _write_manifest(folder, manifest)
                continue
            doc = parsed.get(pdf, spec.parser, doc_meta, sha=sha)
            doc.name = name                                 # 子目录里的文档名带路径，片上记的也要是它
            chunks = chunk_document(doc, spec.chunk)
            vectors, encoded = cache.embed([c.search_text for c in chunks]) if cache is not None else (None, 0)
            if old:                                         # 先从清单里拿掉再写文件：断在写文件中间，下次当新增重做
                del manifest[name]
                _write_manifest(folder, manifest)
            (folder / "docs" / name).parent.mkdir(parents=True, exist_ok=True)
            _write_chunks(folder / "docs" / f"{name}.jsonl", chunks)
            if vectors is not None:
                np.save(folder / "docs" / f"{name}.npy", vectors)
            manifest[name] = {"sha": sha, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "meta": doc_meta, "pages": doc.pages, "chunks": len(chunks)}
            _write_manifest(folder, manifest)
            (changes.changed if old else changes.added).append(name)
            if progress:
                print(f"  [{n}/{len(pdfs)}] {'更新' if old else '新增'} {name}：{len(chunks)} 片"
                      + (f"，新编码 {encoded}" if cache is not None else ""), flush=True)

        index = cls.load(folder)
        index.changes = changes
        return index

    @classmethod
    def load(cls, folder: Path) -> "Index":
        d = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
        spec = IndexSpec(d["parser"], ChunkSpec(**d["chunk"]), d["embedder"], d["max_length"])
        manifest = _read_manifest(folder)
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for name in sorted(manifest):
            for line in (folder / "docs" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                c = json.loads(line)
                chunks.append(Chunk(len(chunks), c["doc"], tuple(c["pages"]), c["text"], c["context"],
                                    tuple(c["section"])))
            if spec.embedder:
                vectors.append(np.load(folder / "docs" / f"{name}.npy"))
        vectors = [v for v in vectors if len(v)]
        index = cls(spec, chunks, np.concatenate(vectors) if vectors else None)
        index.docs = manifest
        return index

    # ------------------------------------------------------------ 查
    def search(self, query: str, spec: SearchSpec = SearchSpec(), docs: Sequence[str] | None = None) -> list[Hit]:
        """docs：只在这几份文档里找（元数据过滤）。None = 全部。"""
        allowed = self._allowed(docs)
        runs: dict[str, list[tuple[int, float]]] = {}
        for name in spec.retrievers:
            if name == "bm25":
                runs[name] = self.bm25.search(query, spec.candidates, allowed)
            elif name == "dense":
                if self.vectors is None:
                    raise ValueError(f"索引 {self.spec.label} 没有向量（建索引时没配嵌入模型）")
                q = Embedder(self.spec.embedder, self.spec.max_length).embed([query], query=True)[0]
                runs[name] = top_k(self.vectors, q, spec.candidates, allowed)
            else:
                raise ValueError(f"没有叫 {name} 的召回方式：bm25 / dense")
        hits = _fuse(runs, spec.rrf_k)[:spec.candidates]
        hits = [Hit(self.chunks[i], score, ranks) for i, score, ranks in hits]
        if spec.reranker and hits:
            scores = Reranker(spec.reranker, self.spec.max_length).scores(query, [h.chunk.search_text for h in hits])
            hits = [Hit(h.chunk, s, h.ranks) for h, s in sorted(zip(hits, scores), key=lambda x: -x[1])]
        return hits[:spec.top_k]

    def _allowed(self, docs: Sequence[str] | None) -> np.ndarray | None:
        if docs is None:
            return None
        mask = np.zeros(len(self.chunks), dtype=bool)
        for d in docs:
            mask[self._by_doc.get(d, [])] = True
        return mask


def _fuse(runs: dict[str, list[tuple[int, float]]], k: int) -> list[tuple[int, float, dict[str, int]]]:
    """RRF（倒数排名融合）：只看名次不看分数，BM25 和余弦的分数尺度不同也能直接合。只有一路时就是它自己的顺序。"""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, results in runs.items():
        for rank, (i, _) in enumerate(results, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + rank)
            ranks.setdefault(i, {})[name] = rank
    return [(i, s, ranks[i]) for i, s in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------- 存盘
def _read_manifest(folder: Path) -> dict[str, dict]:
    path = folder / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["docs"] if path.is_file() else {}


def _write_manifest(folder: Path, docs: dict[str, dict]) -> None:
    """先写临时文件再替换：断在中间，清单要么是旧的要么是新的。"""
    tmp = folder / "manifest.tmp"
    tmp.write_text(json.dumps({"docs": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(folder / "manifest.json")


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
