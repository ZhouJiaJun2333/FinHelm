# FinHelm

金融垂直领域的通用 Agent（掌舵的 helm）：取数、分析、算指标、出报告，SQL 只是第一个工具。

从零手写的 Agent 骨架，不依赖 LangChain 之类的框架。目标是**每一行都看得懂**，
同时把后面要加的东西（上下文压缩、记忆、RAG）的插槽都预留好。

分层思路参考 [earendil-works/pi](https://github.com/earendil-works/pi)：
统一 LLM 抽象 → agent 运行时（工具调用 + 状态） → 持久化 → UI。

---

## 快速开始

```bash
conda activate agentlearn
pip install -r requirements.txt
```

**1. 起数据库**（Postgres + 假的电商数据，容器首次启动时自动建表灌数据）

```bash
cd docker && docker compose up -d
```

**2. 配 key**

```bash
copy .env.example .env
```

**3. 跑**

```bash
python run.py
```

每次对话有一个会话目录 `sessions/<会话ID>/`：对话日志 `session.jsonl`、查询结果 `results.jsonl`、
导出的 CSV `exports/`。启动时会打印会话 ID，接着上次聊：

```bash
python run.py --resume
```

（不写 ID 就是最近一次，也可以 `--resume <会话ID>`。）

跑测试（不需要 key，也不需要数据库）：

```bash
pytest
```

### 想看明白内部怎么转，这几个脚本比读代码快

| 脚本 | 回答什么问题 | 需要 |
|---|---|---|
| `examples/event_demo.py` | 事件机制怎么把「运行」和「显示」解耦 | — |
| `docs/what_is_in_context.py` | 一次提问背后发了几次请求？历史里到底存了什么 | — |
| `docs/state_sketch.py` | 设计草图：事件 + reducer 版状态机（没接进主程序） | — |
| `docs/why_tools_resent.py` | 工具为什么必须每轮重发？不发会怎样 | key |
| `docs/raw_http_body.py` | 真实 HTTP 请求体长什么样，`tools` 待在哪 | key |

---

## 目录结构

```
.
├── run.py                      入口（薄壳，真正逻辑在 src/）
├── docker/
│   ├── docker-compose.yml      Postgres（端口 5433，避开常用的 5432）
│   └── initdb/                 容器首次启动自动执行
│       ├── 01_schema.sql       建表 + 表/列注释
│       ├── 02_seed.sql         造假数据
│       └── 03_readonly_role.sql 只读账号 agent_ro
├── src/data_agent/
│   ├── app.py                  ★ 组装层：唯一知道「零件怎么拼」的地方
│   ├── settings.py             配置（.env）
│   ├── prompts.py              系统提示词
│   ├── cli.py                  终端界面（只管显示）
│   │
│   ├── core/                   ★ Agent 运行时（不 import 包外任何模块，tests/test_imports.py 守着）
│   │   ├── messages.py           统一消息结构 = 整个项目的「通用语」
│   │   ├── provider.py           LLMProvider 接口（实现在 llm/）
│   │   ├── tools.py              Tool / ToolOutput / ToolRegistry 工具框架（具体工具在 tools/）
│   │   ├── events.py             运行事件（解耦「运行」和「展示」）
│   │   ├── errors.py             运行时异常（截断 / 拒绝 / 未知停止原因 / 上下文超长 / 压缩失败）
│   │   ├── context/              上下文管理：只追加的历史（消息 + 标记）+ 一组按顺序套用的编辑工序
│   │   │   ├── base.py             Context / ContextEdit / Marker；工序不持有状态，决定记成标记
│   │   │   ├── tool_results.py     ClearOldToolResults：超阈值时把旧工具结果换成带线索的占位
│   │   │   ├── compaction.py       CompactHistory：清理后还太大，把较早的回合换成模型写的摘要
│   │   │   └── turns.py            KeepRecentTurns：按回合裁剪
│   │   ├── tokens.py             上下文用量：锚点 + 增量估算（/context 命令看）
│   │   └── agent.py              主循环 ← 心脏（run() 约 50 行，
│   │                             其余是 stop_reason 分诊和两个钩子）
│   │
│   ├── llm/                    ★ 各家模型的实现（换厂商只动这个包）
│   │   ├── overflow.py           认出各家「上下文超长」的报错
│   │   ├── anthropic_provider.py
│   │   └── openai_provider.py    DeepSeek/千问/Kimi/vLLM 都走这个
│   │
│   ├── tools/                  ★ 具体工具，按领域分包
│   │   └── sql/
│   │       ├── list_tables.py
│   │       ├── describe_table.py
│   │       ├── run_sql.py
│   │       ├── export_csv.py       用户要文件时按编号重跑导出
│   │       └── results.py          结果仓库：r1、r2… 只存一份，run_sql / 导出 / 界面 / 评测共用
│   │
│   ├── domains/                ★ 场景包：一个业务库的行业知识（schema、业务约定），.env 里 DOMAIN= 选
│   │   ├── shop.py               自己造的电商库
│   │   └── financial.py          BIRD 的捷克银行库
│   │
│   ├── session/                ★ 会话落盘（学 pi / Claude Code 的 JSONL 日志）
│   │   ├── store.py              会话目录；每轮成功之后追加日志；读回历史（--resume）
│   │   └── codec.py              消息 + 标记 ↔ JSON
│   │
│   └── db/                     ★ 数据访问层
│       ├── connection.py         连接 + 只读保护
│       └── introspection.py      读 schema（Agent 的「眼睛」）
│
├── examples/
│   └── event_demo.py           事件/回调机制的最小演示
├── docs/                       可运行的「为什么这么写」说明（见上面的表）
├── evals/                      评测：真模型跑标准题、自动判分（见下面「评测」一节）
│   ├── cases/shop.jsonl          20 道题，答案存标准 SQL
│   ├── runner.py / graders.py    跑一道题 / 判分（判分器有单元测试）
│   ├── report.py / run.py        报告 / 命令行入口
│   └── runs/                     每次运行的记录（不进 git）
└── tests/                      329 个用例，全部不需要 key 和数据库（共用的假模型在 fakes.py）
    ├── test_agent_loop.py                    主循环行为
    ├── test_stop_reason_and_finish_turn.py   完成判定 + 结束钩子
    ├── test_provider_conversion.py           两家 provider 的格式转换
    └── test_sql_guard.py                     SQL 防线
```

### 依赖方向（只能单向，不能反过来）

```
cli ──┐
      ├──> app ──> { llm, tools/sql, db } ──> core
tests ┘                  tools/sql ──> db ──> Postgres
```

**`core/` 不 import 包外的任何模块**：它自己定义需要的接口（`LLMProvider`、`Tool`、`BaseContext`），
llm/ 和 tools/ 去实现（依赖倒置）。这是整个结构的地基：
正因为这样，`tests/test_agent_loop.py` 才能塞个假模型把主循环完整测一遍。

---

## 八个核心概念

### 1. Agent 本质上就是一个循环

```python
while True:
    response = 模型(历史, 工具表)
    if 没有工具调用:
        return 答案
    执行工具, 把结果塞回历史
```

（这是骨架。真实的 `run()` 在此之上多了两件事：判断「模型是不是真的答完了」
和「要不要让它接着干」—— 见概念 7、8。）

`core/agent.py` 就是这个，再没别的。剩下的复杂度全在
「历史怎么管」「工具怎么写」「提示词怎么写」——所以那三块各自独立成模块。

### 2. 模型没有手

模型做的唯一一件事是**输出一段 JSON**说「我想调用 `run_sql`，参数是这个」。
真正连 Postgres 的是你本机的 `db/connection.py`。
模型连执行成功了没有都不知道，全靠你把结果当成一条消息发回去。

推论：**Agent 的能力边界 = 你给了哪几个工具 + 工具里写了什么**，安全边界也一样。
跑一下 `docs/raw_http_body.py` 看实物 —— 那是字面意义上「线上传的那串字节」。

### 3. 模型的世界 = 它的工具能感知到的东西

你觉得「显然应该知道」的事，模型不一定有通道知道。
Agent 不听话的时候，先问一句：**它到底有没有办法知道这件事？**
多半是缺工具或缺上下文，而不是提示词不够凶。

这也是为什么 `list_tables` 必须存在 —— 没有它，你问「有什么数据」它只能反问你。

### 4. 统一消息格式是抽象层的关键

Anthropic 用 content blocks（`text`/`tool_use`/`tool_result`/`thinking`），
OpenAI 用 `tool_calls` + `role="tool"`，参数一个是 dict 一个是 JSON 字符串。
`core/messages.py` 定义中立结构，翻译全关在 `llm/` 里。

⚠️ **但中立结构永远是各家格式的交集，厂商私有字段在交集之外。**
Anthropic 的 thinking block、DeepSeek 的 `reasoning_content` 都是这一类，
而且**要求原样回传**，自己从中立结构拼回去一定会丢。
所以 `Message.raw` 不是给 Anthropic 特设的补丁，它是抽象层的必要组成部分：
中立结构让上层代码通用，`raw` 保证回传时不丢东西。**两家 provider 都要有。**

### 5. 工具报错不能中断 Agent

`Tool.execute()` 兜住所有异常，转成错误消息喂回模型，让它自己读错误、修正、重试。
这是 Agent 能「自愈」的关键。参数校验失败也一样——不抛异常，而是告诉模型「你参数传错了」。

### 6. ⚠️ 裁剪历史不能一刀切

`assistant` 的 `tool_calls` 和后面 `role="tool"` 的结果**必须成对**。
从中间截断，API 直接 400。要裁就以「一个完整回合」为单位——
`core/context/turns.py::KeepRecentTurns` 演示了做法，`tests/` 里有对应的测试。
注意「回合」从**真人提问**开始：Agent 自己补的 nudge 也是 `role="user"`，
但它标了 `meta.synthetic`，不算新回合，否则切口会落在一轮中间、把原来的问题切掉。

### 7. ⚠️ 判断「完成了」不能只看有没有工具调用

被 `max_tokens` 截断时**同样没有工具调用**，但那是话说到一半被砍了。
只看 `tool_calls` 的话，你会把半句话当成最终答案返回，而且全程不报错。

必须查 `stop_reason`（`core/agent.py::_check_stop_reason`）：

| `stop_reason` | 含义 | 处理 |
|---|---|---|
| `end_turn` / `stop` / `tool_use` / `tool_calls` | 正常 | 放行 |
| `max_tokens` / `length` | **被截断** | 抛 `OutputTruncated` |
| `refusal` / `content_filter` | 被拒绝 | 抛 `ModelRefused` |
| 其他 | 没见过 | 抛 `UnexpectedStopReason`（宁可炸也不静默） |

**而且分诊必须在写进历史之前。** 被截断的回复绝不能留在历史里：半截话会被模型
当成自己说过的结论；更致命的是截断发生在工具调用中途时，历史里会留下一个
没有结果的 `tool_call`，下一次请求直接 400，而且**此后每一轮都报同样的错**——
一轮失败升级成整个会话报废，用户只能 `/reset`。

> 可迁移的原则：**上下文是只追加的共享状态，写入前先确认这条记录值得永久保留。**
> 工具执行失败要记（模型得看见错误才能改），截断的半截回复不记（它没能发生完）。

### 8. 结束条件应该是可替换的

默认规则很简单：请求了工具就继续，没请求就结束。但有时模型**以为**自己答完了，
而你不同意——比如它只分析了三个维度中的一个。

`finish_turn_hook` 就是那个否决权（对应 Claude Code 的 `Stop` hook、pi 的 `finishTurn`）：

```python
def hook(outcome: TurnOutcome) -> TurnDecision:
    if outcome.requested_tools:
        return TurnDecision.keep_going()
    missing = [k for k in ("品类", "渠道") if k not in outcome.response.text]
    if missing:
        return TurnDecision.keep_going(f"{'、'.join(missing)}你还没分析，继续。")
    return TurnDecision.end()

build_application(finish_turn_hook=hook)
```

那个 `nudge` 字符串会作为一条 `user` 消息进历史——**它就是你本来要手打的那句追问**。

两个注意点：
- 交互式场景其实用不上（你自己打一句"继续"就行）。它的价值在**无人值守**：
  定时任务、API 服务、批处理，没人能替模型踩油门。
- **任何能让 Agent 继续的钩子都必须自带刹车**，否则一个写错的钩子能烧光额度。
  `max_steps` 是最后一道保险，但钩子自己也该数着次数。

---

## 安全：三道独立防线

`run_sql` 会执行模型生成的 SQL，这是唯一有风险的地方。防线有三层，不指望任何单点：

| 层 | 位置 | 做什么 |
|---|---|---|
| 1 | `tools/sql/run_sql.py::_validate` | 只允许单条 SELECT/WITH，去注释后再查写操作关键字 |
| 2 | `db/connection.py` | 只读事务 + 语句超时 + 强制行数上限 |
| 3 | `docker/initdb/03_readonly_role.sql` | `agent_ro` 账号**物理上就没有写权限** |

第 1 层是可以被绕过的（注释、大小写、奇怪语法…），所以**第 3 层才是真正的底线**。
永远不要只靠关键字黑名单做安全。

验证一下：

```bash
docker exec dataagent-postgres psql -U agent_ro -d analytics -c "DELETE FROM orders WHERE order_id=1;"
```

会报 `ERROR: cannot execute DELETE in a read-only transaction`。

---

## 数据库里有什么

`shop` schema，电商订单模型，四张表都写了中文注释——
**表注释和列注释是最廉价的业务上下文**，Agent 通过 `describe_table` 自动读到，不用塞进提示词。

| 表 | 行数 | 说明 |
|---|---|---|
| `customers` | 200 | 客户，`region` 故意留了 ~5% 的 NULL |
| `products` | 40 | 商品，有 `unit_price` 和 `unit_cost`，可以算毛利 |
| `orders` | 5000 | 订单头，`status` 有 completed/cancelled/returned |
| `order_items` | ~12500 | 明细，实付 = `quantity × unit_price × (1 - discount)` |

⚠️ 已知问题：`products.product_name` 不唯一 —— 40 个商品只有 8 个名字（每个名字 5 个商品），
是造数据时命名规则写错了。按商品名分组会把 5 个商品合在一起。出评测题时发现的，还没修。

几个可以拿来练手的问题：

- `2025 年哪个大区销售额最高？给我前三名和占比`
- `哪些商品毛利率最低？是不是卖得越多亏得越多`
- `线上和线下渠道的客单价差多少`
- `退货率最高的是哪个品类`
- `有多少客户只下过一单`

---

## 评测

单元测试回答「代码有没有按设计运行」（假模型、几秒跑完）；评测回答「Agent 能不能把活干好」
（真模型、要花钱，改提示词 / 换模型 / 改策略时手动跑）。

```bash
python -m evals.run                          # 20 道题 × 3 次
python -m evals.run --model qwen3.6-flash    # 换个模型比一比
python -m evals.run --only shop-003 --trials 1
```

**怎么判**（`evals/graders.py`，每条规则都有单元测试）：

| 判分项 | 规则 |
|---|---|
| 结果对 | Agent 跑过的 SQL 里，有一条查出了标准答案（重跑它，和标准 SQL 的结果比）。不管顺序、列名；多几列、比例乘了 100、NULL 显示成「未填写」都算对 |
| 严格 | 而且列数也一样（接近 BIRD 的判法） |
| **回答对** | 结果对，而且标准答案里的数字在最终回答里都说到了（「810.05 万」「33.2%」这类写法都认） |
| 　兜底 | SQL 没对上，但标准答案只有一个算出来的数（比例、平均数）、回答里说到了（比如在回答里自己算的增长率），也算回答对，报告里单独列出。整数（ID、个数）不兜底：第一版认过，真跑出了误判 —— 标准答案的账户号只是在候选表里出现过 |

和 BIRD 官方判法的区别：BIRD 只看**最后一条** SQL、要求结果完全一样。冒烟测试里 3 道 Agent 答对的题
按那种判法会全错 —— 它先查出答案，最后又查一条明细给你对比口径。跑 BIRD 时两种分数都报。

**看什么**：pass@1（跑一次答对的概率）、pass^k（连跑 k 次都对的题占比，看稳不稳）、
平均步数 / token / 耗时、按标签分类的准确率、失败分类，以及和上一次运行的逐题对比。
每次运行记下 git 版本、模型、提示词和题库的指纹 —— 分数离开这些就说不清是什么条件下跑出来的。

**基线**（2026-09-24，deepseek-v4.1-flash，20 题 × 3 次）：回答对 **100%**，pass^3 100%，
严格只有 35%（Agent 几乎总会顺手多给几列，比如订单数），平均 3.3 步、13.6 秒、输入约 6.4k token。
60 次里有 21 次是靠**不是最后一条**的 SQL 对上的 —— 按 BIRD 只看最后一条的判法，这些都会被判错。

⚠️ 100% 说明这 20 道题对这个模型**太简单了**：它只能防退步，量不出改进。下一步要加难题
（BIRD 的 financial 库、更刁钻的口径；多轮 + 压缩后追问见下一节）。

已知局限：回答核对只查「该说的数说没说」，不查「有没有多说错的」—— 比如问「超过 16 个」
却把正好 16 个的也列进去，结果核对可能照样通过。目前靠抽查运行记录兜底。

### 多轮会话：清理、压缩之后还答得对吗

```bash
python -m evals.run --cases shop_multi --trials 2    # 3 段会话，每段 14~19 轮
```

单题都是短对话，上下文管理一次都不会触发。`shop_multi` 让同一个 Agent 连着回答十几轮：
**填充轮**（列清单，不判分）把上下文撑大，逼出清理和压缩；**回忆轮**（`match: answer`，只看回答里的数）
问前面算过的结果；还有依赖前文的追问和「第一轮定的口径，压缩后还守不守」。
题目由 `evals/cases/gen_shop_multi.py` 生成。

门槛按真实配置的比例缩小，写在每段会话的 `settings` 里。缩多少看单个工具结果多大 ——
结果不跟着缩，门槛太低会一来结果就清，缓存数字失真。起初结果约 4k token，用 3 万 / 4.5 万（1 万时一轮连清 5 次）；
run_sql 只给预览之后结果最多约 1.5k，改成清理 8000 / 压缩 1.2 万（1 万 / 1.5 万试过，触发太晚，回忆轮大多在整理之前）。

报告多出：每段会话清理 / 压缩了几次、在第几轮；回忆轮重查了几次；缓存命中率另拆出
**清理后**、**压缩后**那一次调用 —— 历史被改，缓存就断在改动的位置。

**基线**（2026-09-25，df78559，deepseek-v4.1-flash，3 段 × 2 次）：14 个判分轮全对，回忆 8 次全对（6 次重查）。
每段约 105 万输入 token、7 分钟。缓存命中 87%；清理后那次调用 43%，压缩后 8%。
没命中的 token 里 55% 是每步新增的内容（躲不掉），**34% 是清理 / 压缩造成的** —— 第 5 步要省的就是这部分。
（之后给 B、C 两段各加了 3 条填充，保证每次都会压缩，下次要重跑基线。）

**第 5 步的两个实验**（2026-09-25，改用 DeepSeek 官方端点）：
- 清理节奏：「现状 / 只清大结果 + 低水位 / 不清理」三组准确率一样，未命中的 token 打平；
  不清理的总输入多 17%，算上缓存价反而更贵 —— 清理的收益在于之后每次请求都少带一截。维持现状。
- 写摘要：原样发对话、末尾追加「请写摘要」（学 Claude Code），替代序列化成文本（学 pi）。
  写摘要命中 27% → 99%，每段会话未命中 13.9 万 → 9.1 万（−35%），一份摘要的输出 5.8k → 1.6k token；
  准确率、回忆、口径约定都没掉。序列化的写法已删掉。
- 收尾：Anthropic 那一路开了缓存（系统提示词断点 + 顶层自动缓存，未实测）；输出上限 8192 → 32768
  （思考 token 算输出，列上百行清单会被截断）；判分器的数值容差改成按写出来的精度算
  （固定 0.005 对 5% 量级的比例太宽）；「各渠道退货率」写明全部年份。

**第 5 步之后的基线**（2026-09-25，6e4e167，DeepSeek 官方 deepseek-flash，3 段 × 2 次）：
28 个判分轮全对，回忆 8/8（全部重查）；每段约 119 万输入 token、4.4 分钟，缓存命中 91%，
清理后那次调用 35%、压缩后 10%、写摘要 99%。

**加了查不到的回忆题**（2026-09-25，393abd6，同上配置）：前面的回忆题 Agent 都靠重新查答对，
测不出摘要记没记住。B、C 两段各加一道：用户在开头口头给个目标（「比 2024 年增长 12%」「线上毛利率目标 35%」），
末尾让按这个目标算 —— 数据库里没有，只能靠上下文 / 摘要。4 次全对，都是压缩 1~2 次之后（滚动摘要也没丢）。
32 个判分轮全对，缓存命中 92%。报告新增「一轮里整理 ≥ 2 次」：0 / 110 轮。
连着几轮都清理（如第 13~16 轮各清一次）是填充轮每轮加 2 万多、门槛只有 3 万，每轮都该清，不是同一轮反复整理。

**大结果在源头处理**（2026-09-25，c913cc2 / 0013876 / 109cf0b）：一个工具结果两个读者（学 pi 的 content / details）——
模型要够推理的最少信息，用户要完整数据。run_sql 的结果编号 r1、r2…：20 行以内原样给模型，更多只给前 10 行；
完整结果（≤1 万行）走 `details` 给界面。模型要别的行就改 SQL 再查（只读查询重跑是同一份数据，
所以不像 pi 那样要翻页工具）。回答里写 `{{r3}}`，界面在那个位置展示整张表，模型不用逐行抄。评测按展开后的回答判分。
文件只在用户要的时候写：终端 `/save`（默认最近的结果，`/save r3 [文件名]` 指定），或者在对话里说「导出」（`export_csv` 工具），两边都按编号重跑当时的 SQL。
编号和结果存在 `tools/sql/results.py` 的结果仓库里，只有一份（以前 CLI、run_sql、评测各存一份）。它不放进对话历史：
历史一轮失败就整轮回滚，但那一轮的结果用户已经看到了，还可能 `/save` —— 仓库记的是「用户见过什么」，只增不减。
（pi、Claude Code 自动把大结果存进临时 / 会话目录，那是给**模型**回头翻的 —— 它们的输出只有一份；
我们的数据能重查，不需要为模型落盘。以后接结果不能重拿的工具再做，见「下一步扩展」。）

| DeepSeek 官方 deepseek-flash | 改前 | 改后 |
|:--|--:|--:|
| 单题库回答对（20 题 × 3） | 100% | 100% |
| 单题库平均输出 token | 944 | 769 |
| 多轮回答对（32 轮） | 100% | 100% |
| 填充轮（列清单）平均输出 token | 4,679 | 502（73/78 轮用了引用） |
| 每段会话输入 token / 其中未命中 | 124 万 / 9.3 万 | 46 万 / 1.2 万 |
| 每段会话耗时 | 249s | 87s |
| 上下文峰值 | 4~7 万 | 1.4~2.2 万 |
| 缓存命中 | 92% | 97% |

副作用：上下文再也涨不到评测门槛（清理 3 万），清理 / 压缩一次都没触发 —— 多轮评测门槛随后降到 8000 / 1.2 万（见上文）。
回忆轮重查从 12/12 降到 6/12：小结果整份留在上下文里，模型直接用。

**当前多轮基线**（2026-09-25，门槛 8000 / 1.2 万，runs/20260925-130738「门槛8000」）：32/32 全对，查不到的回忆 4/4；
每段都在回忆轮之前清理或压缩 1~3 次；每段 36 万输入 token、92s，缓存 94%（清理后 25%、压缩后 26%、写摘要 95%）。
这次发现判分器把「多一行合计」（Agent 用 ROLLUP）判成行数不对，改成去掉唯一一行文字标签是合计的行再比，重判了存档。

### BIRD Mini-Dev：financial 库（公开评测）

自建题库都满分了，量不出改进。[BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) 是公开的 text-to-SQL 评测，
financial 库是一家捷克银行的真实脱敏数据（PKDD'99，账户 / 客户 / 贷款 / 交易 / 信用卡，交易表 105 万行），32 道题。
许可证 CC BY-SA 4.0，数据不进 git（`data/` 在 .gitignore 里）。

**准备**（一次性）：

```bash
mkdir -p data/bird
curl -L -o data/bird/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
curl -L -o data/bird/mini_dev_pg.json https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_pg-00000-of-00001.json
cd data/bird && unzip minidev.zip "minidev/MINIDEV_postgresql/BIRD_dev.sql" "minidev/MINIDEV/dev_databases/financial/database_description/*" && cd ../..
python -m evals.bird.prepare
```

`minidev.zip` 约 800 MB（11 个库的 PostgreSQL 导出都在里面），我们只导 financial 的 8 张表到 `financial` schema；
题目用 Hugging Face 上 2025-07 修订过的版本（zip 里的是旧版）。BIRD 随库发的列说明写成了 `COMMENT ON COLUMN`，
`describe_table` 会给模型看 —— 不然 `a2`~`a16`、捷克语编码只能猜。

**跑**：`python -m evals.run --cases bird_financial --trials 3`。题库第一行 `{"settings": {"domain": "financial"}}`
自动切到 financial 场景包（`domains/financial.py`），不用改 .env。

和 BIRD 官方判法的两处不同，报告里两种分数都有：
- **看哪条 SQL**：主分数看 Agent 跑过的任何一条；「只看最后一条 SQL」和「提交轮」两行是 BIRD 的判法。
- **去重**：BIRD 比的是 `set(结果)`，所以题目用 `distinct` 模式（去重后比集合）。有 3 道题的标准 SQL 查出大量重复行
  （比如 461 行全是 `DISPONENT`），按我们原来的 `set`（重复行要一一对上）写了 DISTINCT 的 Agent 会被判错。

**提交轮**：Agent 回答真人时会多给几列上下文（名字、笔数、最大最小值）、把数 ROUND 好看，
对人是更好的回答，按 BIRD「结果集合完全一样」的规则却是错的 —— 首个基线里 29% 的作答是
「值对了、格式不对」。所以题库第一行带了 `submit`：每题答完，评测再追问一句「交一条只返回所问列、
不 ROUND 的 SQL」，BIRD 判法看这一条。这是评测的输出格式，放在评测里（`evals/bird/prepare.py` 的
`SUBMIT`），不写进场景包，Agent 对人的回答不受影响；提交轮的步数和 token 单独统计，不算进主分数。

**标注存疑**：5 道题的标准 SQL 确实错了（比如 q115 的居民数是文本列、按文本排序选错了区；
q152 一个区有几个账户就被算几次），理由和我们补的写法在 `prepare.py` 的 `DISPUTED` 里，每条都在库里核对过。
补的写法排在原版后面：主分数哪种都认，BIRD 判法和官方判分只认原版。题意有歧义的（账户所有人还是
全部客户）不补 —— 那是题目本身的难度。

**官方判分**：上面的分数都是我们的判分器算的。要对外说的分数用 BIRD 官方脚本再算一遍：

```bash
mkdir -p data/bird/official
curl -L -o data/bird/official/evaluation_ex.py https://raw.githubusercontent.com/bird-bench/mini_dev/main/evaluation/evaluation_ex.py
curl -L -o data/bird/official/evaluation_utils.py https://raw.githubusercontent.com/bird-bench/mini_dev/main/evaluation/evaluation_utils.py
pip install psycopg2-binary func_timeout pymysql     # 官方脚本的依赖，只有这一步用
python -m evals.bird.official evals/runs/<一次 bird_financial 的运行>
```

每个 trial 交提交轮那条 SQL（没有提交轮就交最后一条执行成功的），报官方的 EX（分难度）。官方脚本只改了数据库连接（写死在脚本里，
官方 README 让用户自己改），判分逻辑不动。先把标准答案本身当预测交一遍，必须 100 分 —— 官方脚本
连不上库也只会静默记 0 分。

官方脚本比我们严一处：**连 Python 类型都比**。标准 SQL 常写 `CAST(... AS REAL)`，查出来是 float；
Agent 写 `100.0 * ...`，查出来是 Decimal —— 前 15 位一样也算错。所以报告里「按值比」的 BIRD 分数会比官方高一点，
对外只说官方分。

**结果**（deepseek-flash，32 题 × 3 次，官方 EX）：

| | simple | moderate | challenging | 总 |
|---|--:|--:|--:|--:|
| 首个基线（交最后一条） | 11.1% | 19.7% | 0% | **14.6%** |
| 加提交轮 | 66.7% | 47.0% | 19.1% | **42.7%** |

参照：BIRD 官方给的 Mini-Dev PostgreSQL 基线 GPT-4 是 35.8%（500 题、11 个库，不完全可比）。
提交轮没改 Agent：主分数（回答那一轮）两次都是结果对 66~67%、回答对 73~75%。42.7% 和「按值比」的 52%
差的 9 次全是 float / Decimal 类型不同。没过的 46 次里，15 次是 5 道标注存疑题（官方只认错的原版，永远拿不到），
约 14 次是题意有歧义的题（账户所有人还是全部客户），剩下约 17 次才是要看的。

改了判分规则或补了标准答案，不用重跑模型：`python -m evals.run --regrade <运行目录>` 用存下的 SQL 和回答重判。

⚠️ BIRD 的标注错误率不低（社区统计 Mini-Dev 约一半的题有问题，比如 q94 的标准 SQL 就可疑）。只和自己的旧版本比，
不追榜；失败分析时先看标准答案对不对。

---

## 下一步扩展（插槽都留好了）

| 想加的东西 | 动哪里 | 大致做法 |
|---|---|---|
| **上下文压缩** | `core/context/` 写一个新的 `ContextEdit`，加进 `app.py` 的工序列表 | 两层都已实现：10 万时把较早的工具结果换成带线索的占位（`ClearOldToolResults`）；清理后还超 15 万，把较早的回合交给模型写成滚动摘要，保留最近约 2 万 token 原文（`CompactHistory`）。API 报上下文超长时强制整理一次再重试；`/compact` 手动压缩 |
| **大结果落盘（tool-results/）** | `core/tools.py` 的 `ToolOutput.capped()` | 通用兜底层：工具自己没缩小、结果还超上限时，不再截掉，而是把全文存进 `会话目录/tool-results/<调用id>.txt`，给模型开头一段 + 路径，配一个按位置读的工具（学 Claude Code / pi）。给**结果不能重拿**的工具用（网页、实时 API、Python 输出）；run_sql 能重查，在工具里自己处理。等第一个这类工具来了再做，会话目录已经有了（`Session.root`），放在它下面的 `tool-results/` |
| **长期记忆** | `app.py` 里的 `dynamic_context` 钩子 | 用户偏好、历史结论落盘，每轮检索相关片段拼进系统提示词 |
| **RAG** | 优先做成一个 `retrieve` 工具 | 让模型自己决定何时检索，比自动注入更灵活；向量可以直接存在这个 pgvector 库里 |
| **画图** | `tools/` 下开个 `chart/` 子包 | 查询结果交给 matplotlib，存图返回路径 |
| **流式输出** | `llm/` 各 provider 加 `stream_chat()` | `LLMResponse` 不变，只是分块 yield |
| **人工审批** | 已实现：`build_application(approval_hook=...)` | 传个函数，工具执行前弹确认 |
| **自定义结束条件** | 已实现：`build_application(finish_turn_hook=...)` | 见概念 8 |
| **Web 界面** | 换一个 `EventSink` | `core/` 一行不用动，这就是 `events.py` 存在的意义 |
| **多 Agent** | 把 `Agent` 包成一个 `Tool` | 子 Agent 就是一个工具，天然递归 |
| **持久化会话** | 已实现：`session/` | 每轮成功之后把新增的历史追加进 `session.jsonl`，`--resume` 读回来。以后要分支（从某一轮重来），学 pi 给每条记录加 id / parentId |

加**新工具**是最简单的扩展，三步：

1. 在 `tools/` 下写一个 `Tool` 子类（`name` / `description` / `Args` / `run`）。
   失败就抛异常（会变成 `is_error` 的结果喂回模型）；只读、没副作用的设 `rerunnable = True`，旧结果才会被上下文清理
2. 在 `app.py::build_application` 里注册
3. 想在终端里换个样子显示结果，在 `cli.py` 的 `RESULT_RENDERERS` 里按工具名登记一个函数（拿到的是 `details`）
4. 完事 —— 主循环一行不用动

---

## 踩过的坑

### 1. 造假数据：LATERAL 子查询只求值一次

`docker/initdb/02_seed.sql` 里最重要的一课：

```sql
CROSS JOIN LATERAL (SELECT ... ORDER BY random() LIMIT 1) AS x
```

如果 LATERAL 子查询**没有引用外层的任何列**，PG 会认为它是常量，整个查询只求值一次。
结果就是 200 个客户全在同一个城市、12000 个订单行全是同一个商品。
**而且它不报错**——我是靠 Agent 自己在分析时发现「region 只有一个值」才注意到的。

解决办法：用基于外层主键的哈希来选行，既保证按行求值，又可重现。

教训：造完假数据一定要 `count(DISTINCT)` 每一列验一遍。

### 2. 靠「导入顺序」活着的代码

```
settings ──► llm.base ──► core.messages
                              └─ 导入子模块会先执行 core/__init__
                                     └─ 里面 import agent
                                            └─ agent 又 import llm.base（还没初始化完）→ 炸
```

`import data_agent.settings` 作为第一个导入时直接 `ImportError`。之前一直没暴露，
纯粹因为 `app.py` 恰好先导入了 `core.agent`，顺序绕开了。

当时的解法是 `core/__init__.py` 改成惰性导出（`__getattr__`）—— 绕开了，但环还在。
后来根治：`LLMProvider` 接口挪进 `core/provider.py`，工具框架挪进 `core/tools.py`，
core 不再 import 包外任何东西，环没了，惰性导出也删了。`tests/test_imports.py` 里有一条测试守着这条规则。

**教训：能跑 ≠ 没问题。** 这类 bug 会在你加一个新入口脚本时突然出现。
**绕开不如拆掉**：循环导入说明两个包互相依赖，该问的是「接口归谁」。

### 3. 中立结构装不下厂商私有字段

`llm/openai_provider.py` 里曾写着「OpenAI 格式能从中立结构无损还原，不需要存原生内容」。
这句话是错的——DeepSeek 这类思考模型会返回 `reasoning_content` 并**要求原样回传**，
否则 400。当时没出事纯属运气：调工具时 `content` 恰好是空的，绕过了服务端校验。

解法：两家 provider 都把原生响应存进 `Message.raw`，回传时优先用它。见概念 4。

### 4. 工具调用会「泄漏」成文本

跑 `docs/why_tools_resent.py` 能看到：历史里有工具调用记录、但请求里**不带 `tools` 参数**时，
模型会认出那个工具并试图调用，而 API 没有 schema 可以把它解析成结构化的 `tool_calls`，
于是模型的内部格式直接泄漏成普通文本：

```
tool_calls : []
text       : '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="list_tables">...'
```

后果：`tool_calls` 是空的 → 循环判定「答完了」→ 把乱码当答案返回 → **全程不报错**。

所以**工具表在一个会话里必须保持一致**，不要中途增删。
「历史里留着调用痕迹 + 当前没有定义」正是触发这个问题的配方。
