"""场景包：一个业务场景需要的行业知识和能力。换场景 = .env 里改 DOMAIN，代码不动。

    schema    数据在数据库的哪个 schema（连接的 search_path 也设成它）；None = 不连数据库，只分析上传的文件
    subject   一句话说数据是什么
    rules     业务约定：口径、编码的含义、踩过的坑。列的含义放数据库 COMMENT
    tools     要哪几类工具：sql / python / r。沙箱工具还要 .env 里开着（没装 Docker 可以关）
    data_dir  随场景包一起给的数据文件（相对项目根目录），只读挂进沙箱的 /data/；None = 只有用户上传的文件
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Domain:
    name: str
    schema: str | None
    subject: str
    rules: str
    tools: tuple[str, ...] = ("sql", "python")
    role: str = "数据分析师"
    ambiguity_example: str = '比如"最好的客户"是按金额还是按频次'
    data_dir: str | None = None


def get_domain(name: str) -> Domain:
    from .financial import FINANCIAL
    from .payments import PAYMENTS
    from .research import RESEARCH
    from .shop import SHOP

    domains = {d.name: d for d in (SHOP, FINANCIAL, RESEARCH, PAYMENTS)}
    if name not in domains:
        raise ValueError(f"没有叫 {name} 的场景包，可选：{', '.join(domains)}")
    return domains[name]
