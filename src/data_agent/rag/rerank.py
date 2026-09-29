"""重排：交叉编码器把（问题, 片）一起读一遍打分，比召回时各自编码准，但慢，只给召回的前几十条用。"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence


@lru_cache(maxsize=2)
def _model(name: str, max_length: int):
    import torch
    from sentence_transformers import CrossEncoder

    cuda = torch.cuda.is_available()
    return CrossEncoder(name, device="cuda" if cuda else "cpu", max_length=max_length,
                        model_kwargs={"torch_dtype": "float16"} if cuda else {})


class Reranker:
    def __init__(self, name: str, max_length: int = 1024, batch_size: int = 16) -> None:
        self.name, self.max_length, self.batch_size = name, max_length, batch_size

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        out = _model(self.name, self.max_length).predict([(query, t) for t in texts], batch_size=self.batch_size)
        return [float(s) for s in out]
