"""向量召回用的嵌入模型：本地 sentence-transformers。模型从 Hugging Face 缓存读（HF_HUB_CACHE），
第一次用时加载进显卡，同一个进程里只加载一次。

向量归一化后存 float16：几十万片 × 1024 维不到 1 GB，查的时候直接点积（= 余弦相似度）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

# 常用模型的查询前缀：bge 系列的中文模型查询要加指令，bge-m3 不用
QUERY_PREFIX = {"BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章："}


@lru_cache(maxsize=4)
def _model(name: str, max_length: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=_device(), model_kwargs={"torch_dtype": "float16"}
                                if _device() == "cuda" else {})
    model.max_seq_length = max_length
    return model


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, name: str, max_length: int = 1024, batch_size: int = 32) -> None:
        self.name, self.max_length, self.batch_size = name, max_length, batch_size

    def embed(self, texts: Sequence[str], *, query: bool = False, progress: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX.get(self.name, "") + t for t in texts]
        vectors = _model(self.name, self.max_length).encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=progress)
        return vectors.astype(np.float16)


def top_k(vectors: np.ndarray, query: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    """点积最大的 k 个 (片序号, 分数)。allowed：布尔数组，只在为 True 的片里找。"""
    scores = vectors.astype(np.float32) @ query.astype(np.float32)
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
