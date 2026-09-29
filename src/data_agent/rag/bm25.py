"""BM25（Okapi）：关键词召回，手写的倒排索引。

财报里的科目名、专有名词（「Purchases of property, plant and equipment」）靠字面匹配比向量稳；
但问题用分析师的说法（「capital expenditure」）、财报用科目名时它就搜不到 —— 这要靠向量召回和重排补。

切词学 Lucene 的英文分析器：
    转小写；字母和数字拆开（fy2018 → fy 2018）；去停用词（the、is、what…）；
    极简词干（EnglishMinimalStemFilter 的规则：复数变单数，millions → million、statements → statement）
中文按单字 + 相邻两字（没有分词器也能用）。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

_TOKEN = re.compile(r"[a-z]+|[0-9]+(?:[.,][0-9]+)*|[一-鿿]+")

# Lucene 的英文停用词表，加上问题里常见的虚词
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the their then there these they this
to was will with what which who whom whose why how when where does do did doing has have had having i you we our
your its from than so can could should would may might must shall been being am were
""".split())

K1, B = 1.5, 0.75


def _stem(word: str) -> str:
    """复数变单数，照抄 Lucene 的 EnglishMinimalStemmer：
    -ies → -y（前面不是 a/e）；-ies/-aes/-oes/-ees、-us、-ss 不动；其它 -s 去掉。"""
    n = len(word)
    if n < 3 or word[-1] != "s" or word[-2] in "us":
        return word
    if word[-2] == "e":
        if n > 3 and word[-3] == "i" and word[-4] not in "ae":
            return word[:-3] + "y"
        if word[-3] in "iaoe":
            return word
    return word[:-1]


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if "一" <= tok[0] <= "鿿":
            out += list(tok) + [tok[i:i + 2] for i in range(len(tok) - 1)]
        elif tok[0].isdigit():
            out.append(tok.replace(",", ""))           # 1,577 和 1577 算同一个
        elif tok not in STOPWORDS:
            out.append(_stem(tok))
    return out


class BM25:
    def __init__(self, texts: Iterable[str]) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)   # 词 → [(片序号, 词频)]
        self.lengths: list[int] = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.n = len(self.lengths)
        self.avg = sum(self.lengths) / self.n if self.n else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int, allowed: Sequence[bool] | None = None) -> list[tuple[int, float]]:
        """前 k 个 (片序号, 分数)。allowed：按片序号的布尔表，只在为 True 的片里找（按文档过滤）。"""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                if allowed is not None and not allowed[i]:
                    continue
                norm = K1 * (1 - B + B * self.lengths[i] / self.avg)
                scores[i] += idf * tf * (K1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
