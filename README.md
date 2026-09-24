# SQL 数据分析 Agent

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
│   ├── core/                   ★ Agent 运行时（不依赖任何厂商/工具/数据库）
│   │   ├── messages.py           统一消息结构 = 整个项目的「通用语」
│   │   ├── events.py             运行事件（解耦「运行」和「展示」）
│   │   ├── errors.py             运行时异常（截断 / 拒绝 / 未知停止原因）
│   │   ├── context.py            上下文管理（扩展点）
│   │   ├── tokens.py             上下文用量：锚点 + 增量估算（/context 命令看）
│   │   └── agent.py              主循环 ← 心脏（run() 约 50 行，
│   │                             其余是 stop_reason 分诊和两个钩子）
│   │
│   ├── llm/                    ★ 模型抽象层（换厂商只动这个包）
│   │   ├── base.py               LLMProvider 接口
│   │   ├── anthropic_provider.py
│   │   └── openai_provider.py    DeepSeek/千问/Kimi/vLLM 都走这个
│   │
│   ├── tools/                  ★ 工具框架
│   │   ├── base.py               Tool 基类
│   │   ├── registry.py           注册表
│   │   └── sql/                  具体工具按领域分包
│   │       ├── list_tables.py
│   │       ├── describe_table.py
│   │       └── run_sql.py
│   │
│   └── db/                     ★ 数据访问层
│       ├── connection.py         连接 + 只读保护
│       └── introspection.py      读 schema（Agent 的「眼睛」）
│
├── examples/
│   └── event_demo.py           事件/回调机制的最小演示
├── docs/                       可运行的「为什么这么写」说明（见上面的表）
└── tests/                      62 个用例，全部不需要 key 和数据库
    ├── test_agent_loop.py                    主循环行为
    ├── test_stop_reason_and_finish_turn.py   完成判定 + 结束钩子
    ├── test_provider_conversion.py           两家 provider 的格式转换
    └── test_sql_guard.py                     SQL 防线
```

### 依赖方向（只能单向，不能反过来）

```
cli ──┐
      ├──> app ──> core ──> { llm, tools }
tests ┘                          │
                              tools/sql ──> db ──> Postgres
```

**`core/` 不 import 任何具体的厂商、工具或数据库。** 这是整个结构的地基：
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
`core/context.py::TurnWindowContext` 演示了做法，`tests/` 里有对应的测试。

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

几个可以拿来练手的问题：

- `2025 年哪个大区销售额最高？给我前三名和占比`
- `哪些商品毛利率最低？是不是卖得越多亏得越多`
- `线上和线下渠道的客单价差多少`
- `退货率最高的是哪个品类`
- `有多少客户只下过一单`

---

## 下一步扩展（插槽都留好了）

| 想加的东西 | 动哪里 | 大致做法 |
|---|---|---|
| **上下文压缩** | `core/context.py` 派生新类 | 超阈值时把早期回合交给小模型做摘要，替换成一条 summary |
| **长期记忆** | `app.py` 里的 `dynamic_context` 钩子 | 用户偏好、历史结论落盘，每轮检索相关片段拼进系统提示词 |
| **RAG** | 优先做成一个 `retrieve` 工具 | 让模型自己决定何时检索，比自动注入更灵活；向量可以直接存在这个 pgvector 库里 |
| **画图** | `tools/` 下开个 `chart/` 子包 | 查询结果交给 matplotlib，存图返回路径 |
| **流式输出** | `llm/` 各 provider 加 `stream_chat()` | `LLMResponse` 不变，只是分块 yield |
| **人工审批** | 已实现：`build_application(approval_hook=...)` | 传个函数，工具执行前弹确认 |
| **自定义结束条件** | 已实现：`build_application(finish_turn_hook=...)` | 见概念 8 |
| **Web 界面** | 换一个 `EventSink` | `core/` 一行不用动，这就是 `events.py` 存在的意义 |
| **多 Agent** | 把 `Agent` 包成一个 `Tool` | 子 Agent 就是一个工具，天然递归 |
| **持久化会话** | 新增 `storage/` 包 | `Message` 是 dataclass，`asdict()` 直接落盘 |

加**新工具**是最简单的扩展，三步：

1. 在 `tools/` 下写一个 `Tool` 子类（`name` / `description` / `Args` / `run`）
2. 在 `app.py::build_application` 里注册
3. 完事 —— 主循环一行不用动

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

解法：`core/__init__.py` 改成惰性导出（`__getattr__`），和 `llm/__init__.py` 一样。

**教训：能跑 ≠ 没问题。** 这类 bug 会在你加一个新入口脚本时突然出现。

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
