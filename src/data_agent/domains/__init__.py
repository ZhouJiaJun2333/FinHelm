"""场景包：一个业务库需要的行业知识。换场景 = .env 里改 DOMAIN，代码不动。

    schema    数据在哪个 schema（连接的 search_path 也设成它）
    subject   一句话说这是个什么库
    rules     业务约定：跨表的口径、编码的含义、踩过的坑。列的含义放数据库 COMMENT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Domain:
    name: str
    schema: str
    subject: str
    rules: str


def get_domain(name: str) -> Domain:
    from .financial import FINANCIAL
    from .shop import SHOP

    domains = {d.name: d for d in (SHOP, FINANCIAL)}
    if name not in domains:
        raise ValueError(f"没有叫 {name} 的场景包，可选：{', '.join(domains)}")
    return domains[name]
