"""场景包：BIRD Mini-Dev 的 financial 库（一家捷克银行 1993~1998 年的脱敏数据，PKDD'99 Berka 数据集）。

导库和题目见 evals/bird/prepare.py。

这里只写「看表结构和列说明就能知道、但模型容易忽略」的东西：表之间怎么连、编码的含义在哪查。
**不写**针对某道题的口径 —— BIRD 每道题自带「提示（外部知识）」，题库里拼在问题后面了。
往这里加东西之前想一想：真实用户用这个库也会需要它吗？只对评测题有用的就是作弊。
"""

from . import Domain

FINANCIAL = Domain(
    name="financial",
    schema="financial",
    subject="一家捷克银行的业务库（账户、客户、贷款、交易、信用卡）",
    rules="""\
- 编码都是捷克语（比如 `POPLATEK MESICNE`、`VYDAJ`），含义写在列注释里 —— `describe_table` 能看到。
  用户说「月度出账」「取款」这类意思时，先查列注释对上编码，别猜。
- `district` 表的列叫 `a2`~`a16`，含义（区名、所属地区、人口、平均工资、失业率…）同样在列注释里。
- 客户和账户是多对多，通过 `disp` 连：`disp.type` 区分账户所有人（`OWNER`）和授权使用人（`DISPONENT`）。
  问「账户的所有人」要限定 `OWNER`。信用卡 `card` 挂在 `disp` 上，不直接挂在账户上。
- `loan`、`order`、`trans` 都挂在账户（`account_id`）上。`order` 是 SQL 关键字，写成 `"order"`。
- 用户在问题里给了提示或口径（比如某个编码代表什么、某个差值怎么算），照提示来，并在回答里说明。""",
)
