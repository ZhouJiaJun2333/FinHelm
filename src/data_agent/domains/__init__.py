"""场景包：一个业务库需要的全部「行业知识」—— 数据在哪、是什么、业务口径怎么算。

内核（core/、tools/）不知道任何行业：同一个 Agent 换一个场景包，就从电商分析师变成银行分析师。
换场景 = .env 里改 DOMAIN=financial，代码一行不动。

    shop.py        假电商库（自己造的数据，自建评测题库 shop / shop_multi 用它）
    financial.py   BIRD Mini-Dev 的 financial 库：一家捷克银行的真实脱敏数据

一个场景包现在有：
    schema    数据在哪个 schema（连接的 search_path 也设成它，SQL 里可以不写 schema 前缀）
    subject   一句话说这是个什么库，拼进系统提示词开头
    rules     业务约定：口径、编码的含义、踩过的坑。拼进系统提示词
以后还会有：术语表、指标定义、示例问答、这个场景专用的工具。

为什么业务约定放提示词、列的含义放数据库注释（COMMENT）：
    列注释跟着表走，describe_table 查到哪张表就看到哪张的，不占系统提示词；
    口径是「跨表的规则」（销售额只算 completed、要扣折扣），没有一张表能挂，只能放提示词。
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
