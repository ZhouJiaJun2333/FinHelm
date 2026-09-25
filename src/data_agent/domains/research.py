"""场景包：医学科研，系统评价 / meta 分析。不连数据库，分析研究者上传的文件（Excel、CSV）。

算法和版式对齐 RevMan 5（Cochrane 系统评价的标准软件）：R 的 meta 包 settings.meta("RevMan5")，
森林图 layout = "RevMan5"。偏倚风险图用 robvis，RoB 1 和 RoB 2 都支持，用户指定。
模板函数在 tools/r/templates.R，这里只写模型需要知道的约定。
"""

from . import Domain

RESEARCH = Domain(
    name="research",
    schema=None,
    subject="研究数据（纳入研究的各组事件数或均数、效应量、偏倚风险判定等）",
    tools=("python", "r"),
    role="医学统计分析师",
    ambiguity_example="比如效应量用 RR 还是 OR、用固定效应还是随机效应",
    rules="""\
- **先说分析计划再跑**：结局、效应量、效应模型、合并方法。用户指定了就照用户的；没指定时用下面的默认，
  并在回答里写明用了什么、可以换成什么。
- 默认（RevMan 5 的算法）：二分类用 RR，Mantel-Haenszel 法；连续变量同一量表用 MD、不同量表用 SMD（Hedges' g）；
  已经算好的效应量（HR、校正后的 OR）用通用倒方差法。效应模型默认随机效应（DerSimonian-Laird）。
- **不要根据 I² 自动切换固定 / 随机效应**（Cochrane 手册不推荐）。I² 用来描述异质性，模型按预先定的来。
- 偏倚风险的判定**来自用户的数据**，不要自己判。RoB 1（7 个领域，Low / Unclear / High）还是
  RoB 2（5 个领域，Low / Some concerns / High）由用户指定；没说就看列名和取值推断，并在回答里说明。
- 整理数据时核对：每项研究一行；事件数不超过总人数；有零事件的研究照常放进去（模板按 RevMan 的规则处理，
  两组都是零事件的研究不参与合并）。发现数据有问题先告诉用户，不要自己改数。
- 报告格式：效应量和置信区间保留两位小数，写成「RR 0.67（95% CI 0.54–0.84）」；P 值保留三位，
  小于 0.001 写「P < 0.001」；I² 写成整数百分比。
- 文件里如果有患者个人信息（姓名、身份证号、住院号、电话），不要打印出来：只看列名和汇总。""",
)
