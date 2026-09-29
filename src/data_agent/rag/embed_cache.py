"""嵌入缓存：按「模型 + 片文本」存向量，文本没变就不再编码。

改解析规则、改分片参数、文档换了新版本时，大部分片的文字和原来一样；bge-m3 编码 15.8 万片要二十多分钟，
重新算这些没变的片是浪费。查询不走缓存（每次都不一样，也很快）。

    <root>/<模型名>-L<max_length>/
        pack-0001.keys    每行一个片文本的 SHA-1
        pack-0001.npy     对应的向量（float16，和 keys 一行对一行）
        pack-0002 …       每次有新文本就追加一包，旧包不改：写到一半断了，最多丢最后一包

同一个模型换了 max_length（截断长度）结果会变，所以算在目录名里。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .dense import Embedder


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.embedder = embedder
        self.folder = root / f"{embedder.name.replace('/', '--')}-L{embedder.max_length}"
        self._where: dict[str, tuple[int, int]] = {}      # key → (第几包, 第几行)
        self._packs: list[np.ndarray] = []
        for i, keys in enumerate(sorted(self.folder.glob("pack-*.keys"))):
            self._packs.append(np.load(keys.with_suffix(".npy"), mmap_mode="r"))
            for row, key in enumerate(keys.read_text(encoding="ascii").split()):
                self._where[key] = (i, row)

    def __len__(self) -> int:
        return len(self._where)

    def embed(self, texts: Sequence[str], progress: bool = False) -> tuple[np.ndarray, int]:
        """返回 (向量, 其中新编码了几条)。一批里重复的文本只编码一次。"""
        keys = [text_key(t) for t in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in self._where:
                missing.setdefault(key, text)
        if missing:
            self._add(list(missing), self.embedder.embed(list(missing.values()), progress=progress))
        out = np.empty((len(texts), self._dim()), dtype=np.float16)
        for n, key in enumerate(keys):
            pack, row = self._where[key]
            out[n] = self._packs[pack][row]
        return out, len(missing)

    def _add(self, keys: list[str], vectors: np.ndarray) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        name = self.folder / f"pack-{len(self._packs) + 1:04d}"
        np.save(name.with_suffix(".npy"), vectors.astype(np.float16))
        # keys 最后写：程序在两次写之间断了，这一包没有 keys 文件，下次就当没有
        tmp = name.with_suffix(".tmp")
        tmp.write_text("\n".join(keys) + "\n", encoding="ascii")
        tmp.replace(name.with_suffix(".keys"))
        i = len(self._packs)
        self._packs.append(vectors.astype(np.float16))
        for row, key in enumerate(keys):
            self._where[key] = (i, row)

    def _dim(self) -> int:
        return self._packs[0].shape[1] if self._packs else 0
