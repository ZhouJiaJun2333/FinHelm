"""场景包：支付处理商的交易数据（DABstep 评测的数据，Adyen 合成，CC BY 4.0）。

数据不在数据库里，是一组文件：交易明细 CSV、手续费规则 JSON、商户资料、业务手册。
下载见 evals/dabstep/prepare.py，放在 data/dabstep/context/，只读挂进沙箱的 /data/。

和 financial 一样：这里只写真实分析师拿到这批数据时也会被告知的东西（文件是什么、先读手册），
**不写**针对某道题的口径 —— 费用怎么算、规则怎么匹配，都在手册里，让模型自己读。
"""

from . import Domain

PAYMENTS = Domain(
    name="payments",
    schema=None,
    subject="支付处理商的交易数据、商户资料和手续费规则",
    tools=("python",),
    role="支付数据分析师",
    ambiguity_example="比如「费用」指某一笔交易的还是整个月合计的",
    data_dir="data/dabstep/context",
    rules="""\
- 数据在 `/data/` 下（只读）：
  - `payments.csv`：交易明细，一行一笔，列说明在 `payments-readme.md`
  - `fees.json`：手续费规则，一条规则一个 ID
  - `merchant_data.json`：商户资料（账户类型、MCC、收单行、结算周期…）
  - `merchant_category_codes.csv`、`acquirer_countries.csv`：代码表
  - `manual.md`：业务手册 —— 费用怎么算、规则里每个字段什么意思、怎么和交易匹配、欺诈相关的定义
- **每道题先读手册里相关的部分**，术语和算法以手册为准，不要按常识猜。手册里没写的才自己判断，并在回答里说明。
- 手续费规则怎么和交易对上：
  - 规则里某个字段是 null **或空列表 `[]`**，都表示这个字段适用于所有取值（比如 `is_credit` 为 null 的规则
    信用卡、借记卡都适用；`merchant_category_code` 为 `[]` 的规则适用于所有 MCC）。
  - 一笔交易要**同时**满足一条规则的全部条件，这条规则才适用于它。问「某商户适用哪些规则」时，
    要拿每笔交易逐条去对，不能把各个字段分开检查再拼起来。
- 金额单位是欧元。""",
)
