"""索引和检索。

    IndexSpec   建索引时的配置：解析器、分片、嵌入模型。改了要重建，按指纹各存一个目录，几套可以并存
    SearchSpec  查询时的配置：用哪几路召回、候选多少、RRF 参数、重排模型、返回几条。改了立刻生效

    <root>/<分片方式>-<指纹>/
        spec.json       这个索引是怎么建的
        chunks.jsonl    所有片
        vectors.npy     每片一个向量（float16，归一化）；没配嵌入模型就没有
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .bm25 import BM25
from .chunk import Chunk, ChunkSpec, chunk_document
from .dense import Embedder, top_k
from .parsers import PARSERS, ParsedCache
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


class Index:
    def __init__(self, spec: IndexSpec, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        self.spec, self.chunks, self.vectors = spec, chunks, vectors
        self.bm25 = BM25(c.search_text for c in chunks)
        self._by_doc: dict[str, list[int]] = {}
        for c in chunks:
            self._by_doc.setdefault(c.doc, []).append(c.id)

    # ------------------------------------------------------------ 建、读
    @classmethod
    def build(cls, spec: IndexSpec, pdfs: Iterable[Path], root: Path, parsed: ParsedCache,
              meta: dict[str, dict] | None = None, progress: bool = False) -> "Index":
        """解析（有缓存）→ 分片 →（可选）嵌入，存到 root 下。已经建过同样配置的直接读。"""
        folder = root / f"{spec.chunk.label}-{spec.fingerprint}"
        if (folder / "chunks.jsonl").is_file():
            return cls.load(folder)
        chunks: list[Chunk] = []
        for pdf in pdfs:
            doc = parsed.get(pdf, spec.parser, (meta or {}).get(pdf.stem))
            chunks += chunk_document(doc, spec.chunk, start_id=len(chunks))
        vectors = None
        if spec.embedder:
            vectors = Embedder(spec.embedder, spec.max_length).embed([c.search_text for c in chunks],
                                                                     progress=progress)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "spec.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=1), encoding="utf-8")
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        if vectors is not None:
            np.save(folder / "vectors.npy", vectors)
        return cls(spec, chunks, vectors)

    @classmethod
    def load(cls, folder: Path) -> "Index":
        d = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
        spec = IndexSpec(d["parser"], ChunkSpec(**d["chunk"]), d["embedder"], d["max_length"])
        chunks = []
        for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            chunks.append(Chunk(c["id"], c["doc"], tuple(c["pages"]), c["text"], c["context"], tuple(c["section"])))
        vectors = np.load(folder / "vectors.npy") if (folder / "vectors.npy").is_file() else None
        return cls(spec, chunks, vectors)

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
