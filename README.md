# FinHelm

金融垂直领域的通用 Agent（掌舵的 helm）：取数、分析、算指标、出报告，SQL 只是第一个工具。

从零手写的 Agent 骨架，不依赖 LangChain 之类的框架。目标是**每一行都看得懂**，
同时把后面要加的东西（上下文压缩、记忆、RAG）的插槽都预留好。

分层思路参考 [earendil-works/pi](https://github.com/earendil-works/pi)：
统一 LLM 抽象 → agent 运行时（工具调用 + 状态） → 持久化 → UI。

**评测成绩**（模型 DeepSeek V4.1 Flash，详见[评测](#评测)）：

| 评测 | 成绩 |
|:--|:--|
| [DABstep](https://huggingface.co/spaces/adyen/DABstep) 官方排行榜，450 题（支付数据多步分析） | **71.78%**（Hard 69.31%） |
| BIRD Mini-Dev financial，32 题（官方 EX） | 42.7% |
| 医学 meta 分析（上传杂乱 Excel，12 题 × 3 次） | 100% |
| 多轮会话：清理、压缩之后还答得对吗（9 段会话） | 100% |

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

**2. 建沙箱镜像**（`run_python` / `run_r` 在这两个容器里跑；没有 Docker 就在 `.env` 里设
`PYTHON_SANDBOX=false`、`R_SANDBOX=false`。R 镜像做统计分析（meta 分析）才用得到）

```bash
docker build -t finhelm-sandbox docker/sandbox
docker build -t finhelm-sandbox-r docker/sandbox-r
```

**3. 配 key**

```bash
copy .env.example .env
```

**4. 跑**

```bash
python run.py
```

或者用 Web 界面（第一次要先构建前端，需要 Node 18+）：

```bash
cd web && npm install && npm run build && cd ..
python run_web.py            # 打开 http://127.0.0.1:8765
```

改前端时用 `cd web && npm run dev`（5173 端口，`/api` 转给 `run_web.py`，改了立刻刷新）。

只有一个 Agent，不分场景。它能做什么只看环境：`.env` 里配了 `DATABASE_URL` 就有 SQL 工具，
开着沙箱就有 `run_python` / `run_r`，`DATA_DIR` 挂一个只读数据目录。数据是什么、业务口径怎么算，
写在项目目录（`PROJECT_DIR`，默认示例是 `examples/demo`）的 `AGENTS.md` 里，原样拼进系统提示词 ——
和 Codex、Claude Code 读项目说明文件是一个思路。换项目 = 换 `PROJECT_DIR`，代码不动。

**技能（Skills）**：换一份数据也成立的做法（比如 meta 分析的流程和默认口径）不写进 `AGENTS.md`，写成技能：一个目录一个 `SKILL.md`（[Agent Skills 规范](https://agentskills.io/specification)，学 pi）。系统提示词里只列名字和一句描述，模型做对应的任务前用 `load_skill` 读全文。内置的在 `src/data_agent/skills/builtin/`，项目自己的放 `<PROJECT_DIR>/.agents/skills/`（同名盖过内置的）；frontmatter 的 `tools: [run_r]` 写要用的工具，环境里没有就不列。`/skills` 看有哪些，`/skill:meta-analysis 要做的事` 直接指定。

**长期记忆**：跨会话记住你定过的口径（「我们说的活跃客户是……」）、个人偏好、对做法的纠正（学 Claude Code 的 auto memory）。一条记忆一个 Markdown 文件，用户级在 `~/.finhelm/memory/`，项目级在 `~/.finhelm/projects/<项目路径>/memory/`（不放进项目目录：`AGENTS.md` 是共享的项目说明，记忆是替你个人记的笔记）。会话开始时把每条的一行摘要拼进系统提示词，模型要细节再用 `read_memory` 读全文；写用 `remember`，只对这一轮有效的条件、查出来的数字不记。不用向量检索：几十条的量级，摘要目录放得下，模型自己判断读哪条。`/memory` 看有哪些，改、删直接动文件或者跟它说。`MEMORY_ENABLED=false` 关掉。

记忆之间、记忆和你这次说的不一样时这样定（回答里会说用了哪个）：你这一轮只说「这次按……看看」就照这次算、记忆不动，
说「以后都……」才更新；记忆盖过 `AGENTS.md`；项目级和用户级矛盾按项目级（更具体）；同层两条互相矛盾又影响结果，
就用 `ask_user` 问你以哪条为准，再删掉错的那条。防止矛盾出现的兜底在写入时：`remember` 新建一条时把已有的记忆列给模型，
换了个说法记同一件事（「KA」和「大客户」）它会看到、删掉旧的。评测（memory 题库 10 段会话 × 3）：回答对 100%，
记忆文件检查 45/45，两条矛盾时 3/3 问了用户、问完删掉了错的那条。对照（同样 4 段新场景，用加这些规则之前的提示词）：
也是 100%、矛盾时同样 3/3 问了（ask_user 的描述里本来就写了「和记忆矛盾就问」），「KA」也认出是大客户；
唯一的差别是项目级和用户级单位不同时，旧提示词有 1/3 去问用户，新的按「项目级更具体」直接定。
这个模型不靠这些规则也做对了，规则的作用是把「以哪个为准」定死、写清楚，换个弱一点的模型也照这样做。

每次对话有一个会话目录 `sessions/<会话ID>/`：对话日志 `session.jsonl`、查询结果 `results.jsonl`、
导出的 CSV `exports/`、沙箱的工作目录 `work/`（图在 `work/figures/`）。启动时会打印会话 ID，接着上次聊：

```bash
python run.py --resume
```

（不写 ID 就是最近一次，也可以 `--resume <会话ID>`。）

一轮跑到一半按了 Ctrl-C，或者 API 报错（限流、断网、余额不足），进度会保留：输入 `/continue` 从断的地方接着跑，
后面可以加一句话调整方向（`/continue 别按月拆了，直接算全年`）；直接问新问题就放弃这一轮。
进度存在会话目录的 `checkpoint.json`，程序被关掉了，`--resume` 之后照样能 `/continue`。

**中途问你**：口径有几种理解、算出来差很多（「大客户」按消费额还是按客户类型？门槛多少？），它会停下来问（`ask_user`，
学 Claude Code 的 AskUserQuestion），有选项的会列出来。直接打回答，或者输入编号选一个；同一轮接着做，不算新问题。
不想答就 `/continue`，让它按最合理的理解做、在回答里说明假设。停在问题上时关掉程序，`--resume` 回来还会再问一遍。
批处理这种没人回答的场合设 `ASK_USER=false`。

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
├── run_web.py                  Web 界面入口
├── web/                        Web 前端（React + Vite + TypeScript），npm run build 生成 web/dist
├── docker/
│   ├── docker-compose.yml      Postgres（端口 5433，避开常用的 5432）
│   ├── sandbox/Dockerfile      run_python 的沙箱镜像（pandas / scipy / statsmodels / matplotlib + 中文字体）
│   ├── sandbox-r/Dockerfile    run_r 的沙箱镜像（meta / metafor / robvis / readxl / ggplot2 + 中文字体）
│   └── initdb/                 容器首次启动自动执行
│       ├── 01_schema.sql       建表 + 表/列注释
│       ├── 02_seed.sql         造假数据
│       └── 03_readonly_role.sql 只读账号 agent_ro
├── src/data_agent/
│   ├── app.py                  ★ 组装层：唯一知道「零件怎么拼」的地方
│   ├── settings.py             配置（.env）
│   ├── prompts.py              系统提示词
│   ├── skills/                 ★ 技能目录（catalog.py）和内置技能（builtin/<名字>/SKILL.md）
│   ├── memory.py               ★ 长期记忆：两层目录、一条一个文件、索引
│   ├── cli.py                  终端界面（只管显示）
│   ├── web/                    Web 后端（FastAPI）：runner.py 一个会话一个后台线程，事件 SSE 推给浏览器
│   │
│   ├── core/                   ★ Agent 运行时（不 import 包外任何模块，tests/test_imports.py 守着）
│   │   ├── messages.py           统一消息结构 = 整个项目的「通用语」
│   │   ├── provider.py           LLMProvider 接口（实现在 llm/）
│   │   ├── tools.py              Tool / ToolOutput / ToolRegistry 工具框架（具体工具在 tools/）
│   │   ├── events.py             运行事件（解耦「运行」和「展示」）
│   │   ├── state.py              运行状态 AgentState：从事件折叠出来，界面读它
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
│   │   ├── sql/
│   │   │   ├── list_tables.py
│   │   │   ├── describe_table.py
│   │   │   ├── run_sql.py
│   │   │   ├── export_csv.py       用户要文件时按编号重跑导出
│   │   │   └── results.py          结果仓库：r1、r2… 只存一份，run_sql / 导出 / 界面 / 评测 / 沙箱共用
│   │   ├── sandbox.py          沙箱的宿主这头（两种语言共用）：起容器、收发消息、超时就杀掉重来
│   │   ├── python/
│   │   │   ├── run_python.py       在沙箱里跑 Python；load_result("r3") 直接拿 SQL 结果
│   │   │   └── kernel.py           容器里那头：常驻内核，变量跨调用保留（运行时只读挂进容器）
│   │   ├── r/
│   │   │   ├── run_r.py            在沙箱里跑 R
│   │   │   ├── kernel.R            R 版内核，协议和 Python 的一样
│   │   │   └── templates.R         meta 分析模板 fh_*：算法和版式对齐 RevMan 5（见「医学科研」一节）
│   │   ├── paths.py            模型写的路径（/data/…、/work/…、相对路径）→ 宿主机路径，只放行两个目录
│   │   ├── read_file.py        按行读文本文件、带行号（学 Claude Code 的 Read），一次能读完一份手册
│   │   ├── ask_user.py         中途问用户：抛 NeedsUserInput，Agent 暂停这一轮，回答作为调用结果接回来
│   │   ├── load_skill.py       读一个技能的全文
│   │   ├── memory.py           remember / read_memory
│   │   └── view_image.py       把一张图发给模型看（模型能看图时才注册）
│   │
│   ├── session/                ★ 会话落盘（学 pi / Claude Code 的 JSONL 日志）
│   │   ├── store.py              会话目录；每轮成功之后追加日志；读回历史（--resume）；没跑完那一轮的检查点
│   │   └── codec.py              消息 + 标记 ↔ JSON
│   │
│   └── db/                     ★ 数据访问层
│       ├── connection.py         连接 + 只读保护
│       └── introspection.py      读 schema（Agent 的「眼睛」）
│
├── examples/
│   ├── demo/AGENTS.md          默认项目（.env 的 PROJECT_DIR）：本地库里 shop、financial 两份数据的约定
│   └── event_demo.py           事件/回调机制的最小演示
├── docs/                       可运行的「为什么这么写」说明（见上面的表）
├── evals/                      评测：真模型跑标准题、自动判分（见下面「评测」一节）
│   ├── projects/<名字>/AGENTS.md 每套题库的项目约定（题库第一行 settings.project_dir 指过来）
│   ├── cases/shop.jsonl          20 道题，答案存标准 SQL
│   ├── runner.py / graders.py    跑一道题 / 判分（判分器有单元测试）
│   ├── report.py / run.py        报告 / 命令行入口
│   └── runs/                     每次运行的记录（不进 git）
└── tests/                      398 个用例，都不需要 key 和数据库（共用的假模型在 fakes.py；
                                沙箱隔离和 R 模板的用例要 Docker 镜像，没有就跳过）
    ├── test_agent_loop.py                    主循环行为
    ├── test_stop_reason_and_finish_turn.py   完成判定 + 结束钩子
    ├── test_provider_conversion.py           两家 provider 的格式转换
    └── test_sql_guard.py                     SQL 防线
```

### 依赖方向（只能单向，不能反过来）

```
cli ──┐
      ├──> app ──> { llm, tools/sql, tools/python, tools/r, db } ──> core
tests ┘                  tools/sql ──> db ──> Postgres
                         tools/python、tools/r ──> tools/sandbox.py ──> Docker 里的内核
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

多传、拼错的参数也要报错：pydantic 默认**静默丢掉**多余字段，模型把 `limit` 拼成 `limt`，
调用照样「成功」、带着默认值跑完，它永远不知道参数没生效。`Tool.execute()` 先查这一条，报错里列出可用的参数。

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

**步数用完不等于白做**（`Agent._wrap_up`）：以前直接返回「没做完」，前面几十步的结果全浪费。
现在追加一条收尾提示，让模型不再调工具、根据已有结果回答；又去调工具、被截断、API 报错才用兜底那句话。
收尾提示看配置 `WRAP_UP`：`report`（默认）确定的照实说、没做完的说清楚做到哪一步；
`best_guess` 按最合理的假设给出答案（只在 DABstep 题库打开：排行榜上弃权和答错一样算错）。

**原地打转**：同一轮里同一个工具、同样的参数调到第 3 次，就在结果后面提醒「再调也一样，换个思路或直接回答」。
只提醒不拦：第 2 次常常是正当的（旧结果被上下文清理后，占位就是叫它重调）。

**中途断了接着跑**（轮内检查点，`Agent.resume`）：一轮是事务，失败就回滚 —— 半截的一轮会毒化历史。
代价是前面的步数全丢：DABstep 补跑时 70 题被 429 限流打断，其中 18 题已经跑了 6 步以上、12 题跑了 20 多步，全部重来。
现在回滚照旧（正式历史只放完整的回合），但回滚之前把这一轮的进度另外存下来（`agent.interrupted`，
CLI 里每走一步还写一次 `checkpoint.json`），`resume()` 接回历史、从下一步接着跑，步数用剩下的。

- **工具执行到一半断的，不重做**：副作用可能已经发生（沙箱里的变量改了一半），给没执行完的调用补一条「中断，结果未知」
  的结果，形状照样合法（学 Claude Code 补 tool_result、pi 标「外部结果未知」）。
- **程序重启后接着跑**：沙箱内核是新的，这一轮用过沙箱的话补一句「之前的变量没了」，不重放代码（有副作用、耗时、结果不一定一样）。
- **评测**：出错后等 30 秒 / 2 分钟接着跑；有进展就重新计数，同一步上连续失败 3 次才放弃。
  `--inject-errors 0.1` 按概率注入 API 错误来验证，`--no-resume` 是以前整题作废的对照。

**验证**（dabstep_dev 10 题 × 3 次，每次请求 10% 概率注入错误）：注入 34 次，30 个 trial 里 20 个被打断过、都接着跑了。
回答对 **70%**、pass^3 70%，和不注入的那组逐题一模一样；被打断的 20 个里答对 13 个，错的 7 个全在三道本来就次次错的题上。
中断时已经走了 0–24 步，接着跑省下 229 次请求、452 万输入 token（占实际总输入的 70%）—— 整题重来至少要多花这么多，
还不算重来后可能走出另一条路。没有检查点的话，这 20 个 trial 都记为出错，分数大约只剩 8/30 ≈ 27%。
唯一放弃的一次：dab-2697 第 24 步上连着被注入 3 次，按规则（同一步连续失败 3 次）放弃了，这题本来就次次错。

**中途问用户**（`ask_user`，复用检查点）：以前模型想问只能把这一轮结束掉、把问题当回答，用户答了算新的一轮，
评测里还会被当成最终回答判错。现在问题是一次工具调用：

- 工具不等回答、不碰界面，只抛 `NeedsUserInput`（`Tool.execute` 不兜它）。Agent 把同一批的其它调用执行完，
  抛 `AwaitingUser`；和出错一样回滚、把进度存成检查点，只是提问那次调用的结果空着（`InterruptedTurn.pending`）。
- 界面拿到问题去问，回答用 `agent.resume(回答)` 接回来：回答就是那次调用的结果，步数接着数，还是同一轮。
  不回答（`resume("")`）就告诉它「用户没回答，按最合理的理解做，并在回答里说明假设」。
- 为什么不学 `approval_hook` 那样在工具里直接 `input()` 等：那样线程一直卡着，程序关了问题就没了；
  现在问题在 `checkpoint.json` 里，重启还能接着问，以后 Web 界面也是同一套（发问题、等下一个请求带回答来）。
- 同一批里问两个问题，第二个直接回绝（一次只问一个）。
- 评测里没有真人：题目写 `replies`，问了就按顺序拿来当回答；`ask: true / false` 检查该问的问了、不该问的没问。

**验证**（DeepSeek 官方端点 deepseek-flash，每题 3 次，`evals/cases/ask.jsonl`：3 道口径查不到的该问、3 道项目约定写清楚了的不该问）：

| | 回答对 | 该问的问了 | 不该问的问了 |
|:--|--:|--:|--:|
| 关掉 ask_user | 61% | —（只能把「你告诉我口径」当回答，这一轮就结束了） | — |
| **开着** | **83%** | 6 / 9 | 0 / 9 |

- 「大客户」「核心产品线」3/3 都问了，拿到回答同一轮算完，全对；关掉时模型也知道没定义，但只能停下来反问，0–1/3。
- 「活跃客户」一次都没问：它按行业惯例（近 90 天、以数据最后一天为准）算，列出 30/60/90/180 天的敏感性表、说可以换。
  原因在原则里那条「有歧义时……没有定义才按最常见的口径算」，和 ask_user 打架。注册了 ask_user 时改成
  「没有定义、几种理解算出来差很多就问，有常见口径也一样；差别不大才按常见口径算」（没注册时原文不变）。
  改完重跑 ask：**100%**，该问 9/9、不该问 0/9，「活跃客户」0/3 → 3/3（还会反问「现在」是数据截止日还是今天）。
  这条原则所有题库都有，其余题库没重跑（改动只在有歧义、结果差很多时才生效）。
- 其它题库 ask_user 开着重跑、对比上一次：shop 95%（98%，3 次都是答对了但表头写「万元」判分器不认）、shop_multi 100%（96%）、
  memory 100%、research 97%（100%）、BIRD 回答对 69% / 提交 47%（71% / 50%，逐题 ±1 的都是没问的题）、dabstep_dev 77%（70%）。
  一共只问了 6 次，都问在真有歧义的地方：BIRD 0094 / 0095 两个条件落在不同的人身上（0095 从 1/3 到 3/3），
  research rs-12 数据写成「151/5」（事件数超过人数）。

---

## 运行状态和流式输出

**运行状态**（`core/state.py`，学 pi 的 `AgentState`）：界面要知道「是不是在跑、在跑哪个工具、正在输出的半句话、
上一轮为什么断了、在等用户回答什么」。这些以前散在 Agent 的私有字段和局部变量里，界面只能自己从事件里再攒一遍。
现在状态是事件折叠出来的：`state = reduce(state, event)`，`reduce` 是纯函数。

- Agent 每个事件先 `reduce` 再交给订阅者（`Agent._emit`，同 pi 的 `processEvents`），订阅者收到事件时 `agent.state` 已经是新的。
- 前端拿到同一串事件，用同一个 `reduce` 折出来的状态和 `agent.state` 一模一样（测试里 `fold(events) == agent.state`）；
  中途连上的前端先要一份 `agent.state` 快照，再接着收事件。
- 为了能折出来，补了几个事件：`TurnStarted` / `TurnEnded`（一轮的开始和结束，结束带回答或断的原因）、
  `StepStarted`（开始一步）、`TextDelta`（流式的一小段）、`ConversationReset`；工具事件带上 `call_id`（一步里可能调两次同名工具）。
- 和 `docs/state_sketch.py` 的草图不同：状态**不用来恢复现场**。恢复靠检查点（`InterruptedTurn`），历史在 `Context` 里；
  状态只是给界面看的派生数据，丢了重新折一遍就有。

**流式输出**：`LLMProvider.stream(..., on_delta)` 边生成边交出文字，返回值和 `chat()` 一模一样（进历史的消息、用量、stop_reason），
所以主循环只在「怎么请求」这一处分叉（`Agent._call_llm`）。不支持流式的 provider 不用改：基类默认整段调 `chat`、回答一次给完。

- OpenAI 兼容：把分块拼回和非流式一样的原生 message —— 思考模型的 `reasoning_content` 回传时必须带上，工具参数是一段段 JSON 拼起来的。
  DeepSeek 的用量放在带 `finish_reason` 的最后一块，OpenAI 单独一块（`choices` 为空），两种都认。
- Anthropic：SDK 的 `messages.stream()` 自己拼好最终消息（`get_final_message()`），和 `chat` 走同一个转换。
- 终端：正文边来边打，`{{r3}}` 先原样出来，这一步说完补上表格；思考默认只显示「💭 思考中…」，`--verbose` 暗色原样打。
  最终回答改成在 `TurnEnded` 时打（流式打过就不再打），`_run` 不再管打印 —— 界面就是一个订阅事件的 sink。
- 配置 `STREAM`（默认开）。评测强制关：不需要看，也不想每题收几千个增量事件；请求本身和不流式一样，没重跑评测。

### Web 界面

`python run_web.py`。布局学 Claude desktop（左边会话列表、中间对话、右边结果面板），配色和标是自己的。
Web 界面就是换了一个事件订阅者，`core/` 没动。

- **后端**（FastAPI，`src/data_agent/web/`）：一个打开的会话 = 一套 `Application` + 一个跑回合的后台线程（Agent 是同步的）。
  事件从那个线程出来，经 `loop.call_soon_threadsafe` 交给各个页面的 asyncio 队列，SSE 推出去；发消息、回答、停止是普通 POST。
  页面连上时先拿一份快照（历史换成时间线条目 + `agent.state`），之后接着收事件。快照和推送用同一把锁：
  快照之前的事件已经在快照里，之后的一定进队列。
- **前端**（React + Vite + TS，`web/`）：`timeline.ts` 是 TypeScript 版的 reduce，(条目, 事件) → 新条目；
  状态栏（思考中 / 执行工具 / 等你回答、第几步、上下文用量）直接读服务器随事件带来的 `state`。
- **停止**：点停止后，下一个事件上抛 `Stopped`（和 Ctrl-C 一样是 `BaseException`，不会被「尽力而为」的 `except Exception` 吞掉），
  走原来的事务回滚 + 检查点，页面上是「已停止 · 做完了 n 步 · 继续」。模型每吐一个字就是一个事件，所以回答中途能马上停；
  工具正在跑的要等它跑完。
- **ask_user**：问题是一张带编号选项的卡片，点选项或者直接在输入框里写就是回答（`resume`），「让它自己判断」= 不回答。
- **MCP 审批**：外部工具第一次调用时，跑回合的线程等页面点「这次允许 / 本会话都允许 / 拒绝」（10 分钟没人点当拒绝）。
- **结果**：`{{r3}}` 在回答里是一张可展开的表格卡片，右边面板看全部行、看 SQL、下载 CSV（和 `/save` 同一个函数）；
  沙箱画的图显示在工具卡片里，回答里用相对路径引用的图（`figures/a.png`）换成会话 work 目录的地址。
- 只监听 127.0.0.1：没有登录，连上的人能让 Agent 跑 SQL、跑代码。文件接口只给会话 work 目录里的文件（防 `../`）。

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

### run_python / run_r：模型写的代码关在容器里

Python 和 R 能做的事比 SQL 多得多，所以不在代码层面拦（拦不住），而是让它**跑在哪都伤不到人**：

| 限制 | 怎么做（`tools/sandbox.py::Sandbox.docker`，两种语言一样） |
|---|---|
| 断网 | `--network none`：以后读研报、网页，里面的注入也没法把数据发出去 |
| 文件 | 根目录只读，只挂载会话的 `work/`；`/tmp` 是 256MB 的内存盘；非 root 用户，去掉所有 capabilities |
| 资源 | 内存 2G（超了被杀，内核自动重启）、2 核、最多 128 个进程 |
| 时间 | 两道超时：内核里的软超时打断代码、变量还在；卡在 C 代码里打断不了，宿主机再等 10 秒就杀掉整个内核 |
| 依赖 | 镜像里预装，运行时不能装包 |

数据只从两个口子进去：`load_result("r3")` 从结果仓库取 SQL 的完整结果，数字不经过模型的手；
用户 `/attach` 的文件复制进 `work/inputs/`（原件沙箱碰不到）。沙箱不能连数据库，取数只能走 `run_sql` 的三道防线。

### view_image：模型看自己画的图

文字重叠、标题被裁、单位写错，光看代码和输出是发现不了的。学 Claude Code / pi / Codex：
不另设审图员，主模型按需调 `view_image(path)` 把图「拉」进上下文，提示词叫它交付前看一眼（`fh_` 模板画的不用看）。

| 环节 | 怎么做 |
|---|---|
| 工具 | 只能看 `work/` 下的文件；长边缩到 1568 以内再发；PDF / SVG 看不了（叫模型另存 PNG）；只读，旧结果能被清理 |
| 消息 | `Message.images`。Anthropic 原生放进 `tool_result`；OpenAI 兼容的 tool 消息只能是文字，图片挪到这批工具结果后面的一条 user 消息里（学 pi） |
| 能力开关 | `OPENAI_VISION`：deepseek-flash 能看，deepseek-v4-pro 不能 —— 它收到图片不报错，只在回答里说 Unsupported Image，只能靠配置。关掉就不注册工具，提示词里那句也跟着没了；历史里已有的图发送时换成一句说明 |
| 上下文 | 按面积估 token（宽×高/750，1568×980 ≈ 2000；DeepSeek 实测约 1000，宁可高估）；清理旧结果时图片一起换成占位；写摘要前换成 `[图片]` |

### MCP：外部工具服务器

项目目录放一个 `.mcp.json`（格式和 Claude Code 一样），外部 MCP 服务器的工具就注册进来，名字是 `mcp__服务器__工具`：

```json
{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/data"],
                       "env": {"TOKEN": "${MY_TOKEN}"}, "autoApprove": ["read_text_file"]}}}
```

客户端是手写的（`src/data_agent/mcp/client.py`，只支持 stdio）：子进程 + JSON-RPC 2.0，每行一条消息，
`initialize` → `notifications/initialized` → `tools/list`（分页）→ `tools/call`。一个读线程按 id 把响应交给等它的调用，
服务器反过来发的 `ping` 就地回；stderr 另一个线程读掉（不读的话管道满了子进程会卡住），崩溃时报错带上最后几行。
没用官方 SDK：它是异步的（anyio），我们的 Agent 是同步的，包一层事件循环反而更绕。

外部服务器给的东西都当不可信输入：

| 环节 | 怎么做 |
|---|---|
| 工具描述 | 进模型的工具表 = 别人往我们的提示词里写字：标上「外部 MCP 服务器 X 的工具」，截到 1500 字 |
| 服务器说明 | 握手时给的 `instructions` 拼进系统提示词单独一节，说明是外部写的、只当用法参考 |
| 参数 | 按服务器给的 JSON Schema 校验（手写：类型、必填、enum、上下限、嵌套）；顶层多传的参数也拒掉 —— 拼错的可选参数服务器多半悄悄忽略 |
| 调用 | 第一次调用问用户：这次允许 / 本会话都允许 / 拒绝；`autoApprove` 里的不问；没人能问（评测、测试）的只放行 `autoApprove` |
| 上下文清理 | 服务器自称只读（`readOnlyHint`）的旧结果才能被清掉 |
| 连不上 | 程序照样起来，启动时和 `/mcp` 里提示原因 |

反过来，FinHelm 也是一个 MCP 服务器：`python -m data_agent.mcp.server --docs-dir <目录>` 把知识库的
`list_docs` / `search_docs` / `read_doc` 给任何 MCP 客户端用（Claude Code：`claude mcp add finhelm -- python -m data_agent.mcp.server --docs-dir ...`），
用法说明作为 `instructions` 发过去，和我们自己提示词里的是同一段话。

**验收**：FinanceBench 端到端的 `agentic_mcp` 关掉原生知识库工具、全走 MCP（4 路并发共用一个服务器进程），
和原生同一次运行对比（`python -m evals.financebench.e2e --modes agentic,agentic_mcp`）：

| | 答对 | 平均输入 token | 平均步数 | 平均秒数 | list / search / read_doc 调用 |
|:--|--:|--:|--:|--:|:--|
| 原生工具 | 48/50 = 96% | 24.8k | 4.0 | 15 | 70 / 97 / 12 |
| 走 MCP | 48/50 = 96% | 23.5k | 4.1 | 17 | 63 / 106 / 16 |

逐题对错完全相同（错的是同样两题：标准答案舍入、净利润口径），没有一次超时或协议错误；
慢 2 秒来自一个服务器进程按顺序处理 4 路的请求。协议这一层没丢东西。

Claude Code 也能直接用这个知识库（服务器由它按命令起，跑的是我们的代码和环境）。这台机器上要写全 Python 路径、设 PYTHONPATH
（环境里有个同名的无关 `data_agent` 包）：

```powershell
claude mcp add finhelm -s user -e PYTHONPATH=D:\AgentLearning\MyFirstAgent\src -- D:\anaconda3\envs\agentlearn\python.exe -X utf8 -m data_agent.mcp.server --docs-dir D:\AgentLearning\MyFirstAgent\data\financebench\pdfs
```

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

## 医学科研：系统评价 / meta 分析

还是同一个 Agent，不用切换什么：开着 R 沙箱就有 `run_r`，用户把 Excel 传上来，用一句话说要什么。
流程和默认口径（效应量、合并方法、报告格式）在内置技能 `meta-analysis` 里（`src/data_agent/skills/builtin/meta-analysis/SKILL.md`），模型做 meta 分析前先加载它：

```
你 > /attach D:\课题\纳入研究.xlsx
📎 纳入研究.xlsx → sessions\...\work\inputs\纳入研究.xlsx
你 > 帮我做 meta 分析，画 RevMan 格式的森林图（带偏倚风险），再画偏倚风险图，按手术类型做个亚组分析
```

**模型不自己写统计公式和画图代码**，调 `tools/r/templates.R` 里的模板（`fh_help()` 列出全部）：

| 模板 | 做什么 |
|---|---|
| `fh_meta_bin` / `fh_meta_cont` / `fh_meta_gen` | 二分类（RR/OR/RD，M-H / 倒方差 / Peto）、连续（MD/SMD）、文献直接给的效应量（HR 等）；可带亚组 |
| `fh_report` / `fh_methods` / `fh_export` | RevMan 格式的结果文字、一段中文方法学描述、结果表存 Excel |
| `fh_forest` | RevMan 5 版式的森林图，可带亚组、右侧附各领域偏倚风险；PNG 300dpi + PDF（可加 TIFF） |
| `fh_rob` | RoB 1 / RoB 2（用户指定）的汇总图和比例图；判定写成 Low / 低 / + 都认 |
| `fh_funnel` / `fh_sensitivity` | 漏斗图（10 项以上才做 Egger 检验）、逐一剔除的敏感性分析 |

为什么这样分工：统计代码错了往往不报错（statsmodels 收到不认识的方法名照样算出负权重），
R 包的参数名还在改，自己写的画图代码还得靠看图才能发现毛病。模板把「算得对、版式对、能复现」固定下来，
模型负责读懂杂乱的 Excel（标题行、「12/100」写在一格里）、把用户的话翻译成参数、解释结果。

**数字对齐 RevMan 5**：meta 包的 `settings.meta("RevMan5")`（τ² 用 DerSimonian-Laird、
异质性 Q 用 Mantel-Haenszel 的合并值、零事件按 RevMan 的规则加 0.5）。`tests/test_r_sandbox.py`
拿一份独立实现逐项对：固定效应对 metafor 的 `rma.mh`，随机效应按 RevMan 5 的公式手算（在有异质性的数据上，
metafor 的 DL 用的是倒方差的 Q，结果会不一样），SMD 按 Hedges' g 的 RevMan 公式手算，误差都在 1e-10 以内。

还没做：拿已发表的 Cochrane 系统评价的数据做评测题（像 BIRD 那样），网状 meta、meta 回归不在模板里。

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

**基线**（2026-09-25，6da696b，deepseek-v4.1-flash，3 段 × 2 次）：14 个判分轮全对，回忆 8 次全对（6 次重查）。
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

**第 5 步之后的基线**（2026-09-25，f203a3b，DeepSeek 官方 deepseek-flash，3 段 × 2 次）：
28 个判分轮全对，回忆 8/8（全部重查）；每段约 119 万输入 token、4.4 分钟，缓存命中 91%，
清理后那次调用 35%、压缩后 10%、写摘要 99%。

**加了查不到的回忆题**（2026-09-25，6fe7055，同上配置）：前面的回忆题 Agent 都靠重新查答对，
测不出摘要记没记住。B、C 两段各加一道：用户在开头口头给个目标（「比 2024 年增长 12%」「线上毛利率目标 35%」），
末尾让按这个目标算 —— 数据库里没有，只能靠上下文 / 摘要。4 次全对，都是压缩 1~2 次之后（滚动摘要也没丢）。
32 个判分轮全对，缓存命中 92%。报告新增「一轮里整理 ≥ 2 次」：0 / 110 轮。
连着几轮都清理（如第 13~16 轮各清一次）是填充轮每轮加 2 万多、门槛只有 3 万，每轮都该清，不是同一轮反复整理。

**大结果在源头处理**（2026-09-25，7b89220 / 933bf98 / b8f64d0）：一个工具结果两个读者（学 pi 的 content / details）——
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

### 医学科研：上传文件的题（research）

```bash
python -m evals.run --cases research --trials 3      # 12 道题，要 Docker 和 R 镜像
python evals/cases/gen_research.py                   # 改题之后重新生成 Excel、标准答案和题库
```

**数据是真的**：7 个 metafor / meta 包收录的已发表 meta 分析（BCG 疫苗、抗菌导管、卒中单元、被动吸烟、静脉镁剂…），
原样导出在 `evals/cases/research/source/`。**文件是乱的**：生成脚本把它们做成用户真会上传的那种 Excel ——
标题行、两行合并表头、「4/123」「55±47」「1.18 (0.90–1.54)」写在一格里、给的是标准误不是标准差、中间空行、表底合计行。

**标准答案不用模板算**：`research/gold.R` 直接调 meta 包的 RevMan 5 设置。`fh_` 模板是被考的对象，不能自己给自己判分。
BCG 那题的 RR 0.49（0.34–0.70）、I² 92% 和教科书一致。

| 题型 | 判分 |
|---|---|
| 合并效应、CI、I²、亚组、敏感性分析 | `gold_values`：每个数回答里都要说到。容差按回答写到几位算，不看正负号（「少住 14 天」算说到了 −13.98） |
| 该用的模板（通用倒方差、带偏倚风险的森林图） | `expect_code`：沙箱代码里要出现 `fh_meta_gen(`、`fh_forest(..., rob =` |
| 数据有错（Heard 1998 写成 151/5） | `expect_text`：回答里要点名这项研究、说出问题，不能自己改数 |
| 自己画图 | `expect_figure`；另外统计「自己画了图之后有没有 `view_image` 看一眼」，不计分 |

每个 trial 有自己的工作目录（`运行目录/work/<题>-<次>/`），图留着，失败时能直接看。步数耗尽、出错的不算对。

**首个基线**（2026-09-26，deepseek-flash，12 题 × 3 次，max_steps 20）：回答对 **94%**，pass^3 83%，
平均 12 步、53 秒、8.2 万输入 token（缓存 92%）。两次失败都是步数耗尽：一次反复修自己画的图；
一次已经指出了 Heard 1998 的错，又去手算 τ² = 0 时 M-H 固定效应和随机效应为什么不一样（RevMan 的随机效应用倒方差权重，本来就不一样）。
自己画图的 4 个 trial 里 3 个交付前看了图。标准误陷阱 3/3 都换算对了，还发现了源数据里一个方差异常小的研究。

**加了步数收尾之后**（2026-09-27，百炼端点）：回答对 **100%**、pass^3 100%。收尾触发 2 次、都答对了 ——
用的是默认的保守收尾（`report`），不引导模型去猜，它照样能根据已有结果把题答对。

### DABstep：支付数据多步分析（公开评测）

[DABstep](https://huggingface.co/spaces/adyen/DABstep) 是 Adyen 出的数据分析 Agent 评测：13.8 万笔支付交易、
1000 条手续费规则、商户资料和一份业务手册，考「读懂文档里的规则，再用代码逐笔算」—— SQL 做不了，考的是 Python 沙箱。

```bash
python -m evals.dabstep.prepare                         # 下载数据到 data/dabstep/（不进 git），生成两个题库
python -m evals.run --cases dabstep_dev --trials 3      # 10 道有答案的题，本地判分
python -m evals.run --cases dabstep --trials 1          # 450 道正式题，答案不公开
python -m evals.dabstep.submission <运行目录>            # 导出排行榜要的 submission.jsonl，去排行榜网页手动提交
```

- **项目 `evals/projects/payments`**：不连数据库，只有 Python 沙箱。数据文件只读挂进容器的 `/data/`（设置里的 `data_dir`），不往每个 trial 复制。
  约定里只写文件是什么、先读手册，不写任何题的口径。
- **答题格式**：题目原文后面附官方的格式要求，让模型最后一行写「最终答案：…」，评测只看这一行。
- **判分**：官方的 `question_scorer` 原样拷贝在 `evals/dabstep/scorer.py`（官方的测试也一起搬了过来，保证和排行榜一致）。
- 450 道正式题的答案不公开。数据集里有别人的提交和逐题得分，理论上能反推出答案，但那等于绕过它故意隐藏的测试集，不这么做。

**首个基线**（2026-09-26，deepseek-flash，dev 10 题 × 3 次，max_steps 25）：回答对 **30%**，pass^3 10%，
平均 16 步、97 秒、25 万输入 token（缓存 95%）。**30 次里 10 次步数耗尽**：手册有 2.2 万字符，工具结果上限 6000，
模型得分段读，每道题开头光读文档、逛数据就花掉约 10 步。答错的两类：算出了手册定义的指标却按常识下结论
（「欺诈最多」手册定义是金额占比，它按笔数选了）；手册里没有的概念（「高欺诈罚款」）不答 Not Applicable，自己编一个解释。

**加 `read_file` 之后**（同一天，同样配置）：回答对 **47%**、pass^3 30%，步数耗尽 10 → 2 次，平均 12.6 步。
同时试了 40 步：50%（多对 1 次，噪声范围内），输入 token 却多 60%，剩下的步数耗尽 40 步也做不完 —— 维持 25 步。

**之后两处改动**：错题分析发现规则匹配理解错了（规则里 null 或空列表表示「全部适用」；一笔交易要同时满足一条规则的全部条件，
不能按字段分开查），写进 payments 约定；另一类是算出了手册定义的指标、却按常识下结论，提示词从「歧义按最常见口径」
改成「有定义按定义」。再加上步数用完时的收尾：

| dev（10 题 × 3 次） | 回答对 | pass^3 | 步数耗尽 |
|:--|--:|--:|--:|
| 首个基线 | 30% | 10% | 10 |
| 加 `read_file` | 47% | 30% | 2 |
| 加规则约定、「有定义按定义」、收尾（2026-09-27，百炼端点） | **77%** | 70% | 0（收尾 4 次） |
| 同上 + 严格参数、原地打转、轮内检查点（2026-09-27，再跑一组） | **70%** | 70% | 0（收尾 2 次） |

最后两组相差 7 个点，全部来自 dab-49 一道题（2/3 → 0/3），其余 9 题每次结果都一样：这题问「欺诈最多的国家」，
按笔数和按手册的欺诈率定义答案不同，模型两种都会选。新加的三样在这 30 次里一次都没触发（没有参数错误、没有重复调用、
没有 API 报错），所以这是模型本身的摇摆，不是改动的影响。每题 3 次、一题摇摆就是 ±6.7 个点，dev 分数按 70%–77% 看。

剩下的错集中在三道题：dab-70 问手册里没有的「高欺诈罚款」（该答 Not Applicable），dab-2697 题意有分歧，都不追；dab-49 见上。

**正式集**（450 题，提交到[排行榜](https://huggingface.co/spaces/adyen/DABstep)）：**71.78%**，Easy 84.72%，Hard 69.31%。
构成要说清楚：首轮 25 步答出 349 题，答对 273（首轮单独算 60.7%）；没交出答案的 101 题用 40 步 + 收尾补跑
（`submission.py` 合并多个运行目录，只按「有没有写出答案」选，不看对错），答对 50；其中被强制收尾的 25 题答对 11 ——
交兜底那句话的话，这 11 分就没了。补跑的一部分走的是 DeepSeek 官方端点（同一个模型）。

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

**跑**：`python -m evals.run --cases bird_financial --trials 3`。题库第一行
`{"settings": {"project_dir": "evals/projects/financial", "db_schema": "financial", ...}}`
指定项目目录（读它的 `AGENTS.md`）和 schema，不用改 .env。

和 BIRD 官方判法的两处不同，报告里两种分数都有：
- **看哪条 SQL**：主分数看 Agent 跑过的任何一条；「只看最后一条 SQL」和「提交轮」两行是 BIRD 的判法。
- **去重**：BIRD 比的是 `set(结果)`，所以题目用 `distinct` 模式（去重后比集合）。有 3 道题的标准 SQL 查出大量重复行
  （比如 461 行全是 `DISPONENT`），按我们原来的 `set`（重复行要一一对上）写了 DISTINCT 的 Agent 会被判错。

**提交轮**：Agent 回答真人时会多给几列上下文（名字、笔数、最大最小值）、把数 ROUND 好看，
对人是更好的回答，按 BIRD「结果集合完全一样」的规则却是错的 —— 首个基线里 29% 的作答是
「值对了、格式不对」。所以题库第一行带了 `submit`：每题答完，评测再追问一句「交一条只返回所问列、
不 ROUND 的 SQL」，BIRD 判法看这一条。这是评测的输出格式，放在评测里（`evals/bird/prepare.py` 的
`SUBMIT`），不写进 AGENTS.md，Agent 对人的回答不受影响；提交轮的步数和 token 单独统计，不算进主分数。

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

**「有定义按定义」的代价**（2026-09-27 回归，百炼端点）：回答对 74%（持平），提交轮按值比 52% → 46%。
少对的 6 次集中在两道题：financial 约定写着「问账户的所有人要限定 `OWNER`」，改成「有定义按定义」之后，
模型把「account holders」「customers among the account opened」都只算了 `OWNER`，标准答案算的是全部客户。
这条原则在 DABstep 上提分，BIRD 只做回归、不为它回退。

改了判分规则或补了标准答案，不用重跑模型：`python -m evals.run --regrade <运行目录>` 用存下的 SQL 和回答重判。

⚠️ BIRD 的标注错误率不低（社区统计 Mini-Dev 约一半的题有问题，比如 q94 的标准 SQL 就可疑）。只和自己的旧版本比，
不追榜；失败分析时先看标准答案对不对。

---

### FinanceBench：财报文档检索（RAG）

[FinanceBench](https://github.com/patronus-ai/financebench) 开源的 150 题：美股公司的 10-K / 10-Q / 8-K，
每题标了证据在哪份文档的哪一页。先只测检索（证据页有没有被搜出来），不调大模型、不花钱；
端到端（Agent 用检索工具答题）是下一步。代码在 `src/data_agent/rag/`，评测在 `evals/financebench/`。

每一环都可配置，配置分两类：**建索引时**（解析器、分片、嵌入模型）改了要重建，按指纹各存一个目录；
**查询时**（召回路数、RRF、重排、返回几条）改了立刻生效。

- **解析**（学 Unstructured / Docling 的元素模型）：PDF → 一串有类型的元素（标题、段落、表格、页眉页脚），
  带页码和章节路径（`PART II > Item 8 > Consolidated Statement of Cash Flows`）。手写的版面分析：
  同一水平线的文字拼成一行、空隙大的分列（直接 `get_text()` 会把表格一格一行打散，科目和数字对不上）；
  在很多页边缘重复出现的是页眉页脚；数字右对齐，按右边缘定列（某年没数不会错位）。368 份 PDF 两分钟。
- **分片**：`fixed`（朴素滑窗）/ `page`（一页一片）/ `structure`（学 Docling 的 HybridChunker：同章节合并到
  512 token，表格按行拆、每块带表头，标题跟着内容走）。每片前面加「文档名 > 章节路径」（简化版 Contextual Retrieval）。
- **召回**：BM25（手写倒排，分词学 Lucene 英文分析器）+ bge-m3 向量，RRF 融合；按文档过滤；
  bge-reranker-v2-m3 重排。模型本地跑（RTX 5070，15.8 万片编码 23 分钟）。
- **增量更新**：索引是文件夹里 PDF 的镜像，`Index.sync` 只处理新增、改过、删掉的文档（按文件内容哈希判断；
  大小和修改时间没变就不重新算哈希，学 git 的 index），每处理完一份更新清单，断了能接着做。三层缓存：
  解析按文件内容（同名换了内容会重新解析，改名不用）、嵌入按「模型 + 片文本」（改分片或解析规则时，文字没变的片不重新编码）、
  BM25 不存盘，加载时现建（全局 IDF 总是最新）。改了解析规则全量重建 368 份：23 分钟 → 24 秒（向量全部命中缓存），
  检索结果逐题和原来一样。
- **接进 Agent（知识库）**：一个文档目录就是一个知识库（学 Bedrock Knowledge Bases / Dify：知识库单独存在，项目去挂载），
  `DOCS_DIRS` 挂几个目录就有几个库，配了才注册 `list_docs`（先定到是哪份文档）/ `search_docs`（在那份里搜片段）/
  `read_doc`（读整页，表格看全）。索引跟着目录走、不跟着项目走：放在 `<RAG_DIR>/collections/<目录>/`，几个项目挂同一个目录
  共用一份；解析、嵌入缓存所有库共用。目录下的 `metadata.jsonl` 给每份文档配公司、期间，拼进片的前缀。
  检索做成工具让模型自己决定搜什么、搜几次（Agentic RAG）：失败分析剩下的「用词对不上」「派生指标页上没这个词」
  都写进了工具说明和提示词（换说法再搜、搜组成项）。上面的检索评测和 Agent 用的是同一份索引。

结果（证据页在前 5 片里的比例，hit@5；「本文档」= 只在题目问的那份里找，「全部」= 368 份混在一起）：

| 做法 | 本文档 hit@5 | 本文档 hit@10 | 全部 hit@5 |
|:--|--:|--:|--:|
| 朴素滑窗 + BM25（最初的分词） | 30% | 39% | 10% |
| 同上，分词改成 Lucene 的（停用词、字母数字拆开、词形还原） | 38% | 49% | 13% |
| 按结构分片 + BM25 | 47% | 59% | 15% |
| + bge-m3 向量（只用向量） | 55% | 71% | 34% |
| BM25 + 向量，RRF 融合 | 65% | 75% | 34% |
| **+ 重排** | **76%** | **85%** | **54%** |

- BM25 败在用词不匹配：问题说「capital expenditure」，财报写「Purchases of property, plant and equipment」。
  向量补上了这块，分析师口吻的题（domain-relevant）从 24% 到 64%（加重排后）。
- 融合比任何单独一路都好：两路找到的是不同的页。重排是提升最大的一步，代价是每题 0.3 秒 → 0.9 秒。
- 全部混在一起时最好也只有 54%，第一片来自正确文档的 71%：先按公司、年份定到那份文档最值钱。

```bash
python -m evals.financebench.retrieval --embedder BAAI/bge-m3 --search bm25 --search dense --search bm25+dense --search "bm25+dense>"
```

**端到端**（`evals/financebench/e2e.py`）：同一批题、同一个模型（deepseek-flash）、同一种判法，比六种给模型看文档的做法。
题目是 metrics-generated 那 50 道（答案都是一个数，都来自 10-K）；只判最后一行「Final answer」，
容差 max(1%, 标准答案末位的一半)。

| 做法 | 告诉是哪份文档 | 答对 | 证据页全拿到 | 平均输入 token | 调用次数 |
|:--|:-:|--:|--:|--:|--:|
| 整份 10-K 放进上下文（平均 12 万、最大 30 万 token） | 是 | 48/50 = 96% | — | 141k | 1 |
| 直接给证据页（检索满分的上限） | 是 | 48/50 = 96% | 50 | 1.3k | 1 |
| 传统 RAG：题目原文检索一次，前 5 片 | 是 | 40/50 = 80% | 27 | 3.0k | 1 |
| 传统 RAG，368 份全库检索 | 否 | 38/50 = 76% | 22 | 3.0k | 1 |
| **查询改写**：先让模型出检索计划（公司年份 → 元数据过滤；要几个数拆几条、用报表措辞），再检索一次 | 否 | **48/50 = 96%** | 47 | **3.7k** | 2 |
| **Agentic RAG**：自己 list_docs 找文档、搜几次、要时读整页、run_python 算 | 否 | **48/50 = 96%** | — | 24.9k | 3.9 |

- 原问题直接检索，证据页全拿到的只有 22~27 题：这类指标要两张表（固定资产周转率 = 利润表的收入 ÷ 资产负债表的固定资产），
  一条查询前 5 片常常只命中一张。错的题都是证据没搜全，模型要么凭残缺的数硬算、要么答「无法确定」。
- **查询改写**补上了几乎全部：抽出公司、年份做元数据过滤（50 题全部定到了对的文档，不告诉文档反而比告诉了还好），
  再按需要的科目拆成 2~3 条查询、用报表里的说法写（「Consolidated Balance Sheets Property and equipment, net」），证据页全拿到 22 → 47。
  固定两次调用、平均 3.7k token，是 Agentic 的 1/7。
- Agentic 同样 96%，贵在能看了结果再搜：改写是搜之前一次想好，搜空了没有第二次（General Mills 那题改写的计划是对的，
  但检索没拿到资产负债表，只能答「无法确定」）。这 50 题的难度用不上多轮，所以两者打平；多轮的价值要在更难的题上看。
- 四种做法都错的 2 题不是检索的问题：一题标准答案舍入有误（股息 389 百万美元 = 0.389 十亿，标准答案写 0.40）；
  一题是净利润的口径（标准答案用归属母公司的；证据拿全的几种里，模型多用了合并口径）。
- 顺带查出一个**解析器的 bug**（已修，解析器 v3）：加粗的列名行（「May 31, 2020 ⎮ May 26, 2019」「Level 1 Level 2 Level 3」）
  被当成了标题，表格的章节路径被它顶掉，「Consolidated Balance Sheets」挂不到表上，按报表名就搜不到。上面 General Mills 那题就是这么丢的。
  现在只有日期、年份、Level 的行当表头归进表里。重建 368 份用了 5 分钟，15.7 万片里 8.9k 片的文字变了、要重新编码，其余走嵌入缓存。
  原问题检索的分数不变（它不按报表名搜）；把上面 50 条改写计划在新索引上重放（不调模型）：证据页全拿到 47 → 48，
  General Mills 从一页没有到全拿到，AES 反而从全拿到退成一部分（查询里写了「December 31 2022」，
  以前这串日期在表的章节名里，现在不在了，被另一张章节名带日期的补充表抢了前面）。
  v3 上重跑三种检索做法：rag 80 → 82%、rag_all 76 → 72%、rag_rewrite 96 → 96%。逐题拆开看，**证据因修复而变的题只升不降**
  （General Mills 在改写和全库两种里、Best Buy 在全库里，都从一页证据没有、答错变成全拿到、答对，共 +3 题）；
  但**证据完全没变、对错却翻了的有 8 题**（全库那一路 6 题，5 题对变错）：同样几片内容，心算结果每次不同，
  证据不全时有时硬算、有时答「无法确定」。50 题跑一次的波动（±4 题）比这次修复的效果还大，
  判断小改动要看确定性的检索指标（证据页拿全没有）、逐题归因，或者每题多跑几次。

```bash
python -m evals.financebench.e2e --label 首轮                                             # 前四种，4 路并发约 6 分钟
python -m evals.financebench.e2e --resume evals/runs/<目录> --modes rag_all,rag_rewrite   # 往同一份报告里追加做法
```

## 下一步扩展（插槽都留好了）

| 想加的东西 | 动哪里 | 大致做法 |
|---|---|---|
| **上下文压缩** | `core/context/` 写一个新的 `ContextEdit`，加进 `app.py` 的工序列表 | 两层都已实现：10 万时把较早的工具结果换成带线索的占位（`ClearOldToolResults`）；清理后还超 15 万，把较早的回合交给模型写成滚动摘要，保留最近约 2 万 token 原文（`CompactHistory`）。API 报上下文超长时强制整理一次再重试（写摘要的请求自己也超长，就丢掉最老的一半回合再写，最多 3 次）；自动压缩失败不中断这一轮，连续失败 3 次熔断（只清理不压缩，`/compact` 成功后恢复）；`/compact` 手动压缩 |
| **大结果落盘（tool-results/）** | `core/tools.py` 的 `ToolOutput.capped()` | 通用兜底层：工具自己没缩小、结果还超上限时，不再截掉，而是把全文存进 `会话目录/tool-results/<调用id>.txt`，给模型开头一段 + 路径，用 `read_file` 按行号分页读（学 Claude Code / pi）。给**结果不能重拿**的工具用（网页、实时 API、Python 输出）；run_sql 能重查，在工具里自己处理。等第一个这类工具来了再做，会话目录已经有了（`Session.root`），放在它下面的 `tool-results/` |
| **长期记忆** | 已实现：`memory.py` + `remember` / `read_memory` | 记忆多到索引放不下时，再加 Claude Code 那种「每轮用小模型按摘要挑几条」；自动从对话里提取（Codex 的后台合并）等评测证明漏记再做 |
| **RAG** | 已实现：`rag/` + `tools/docs.py`（见「FinanceBench」一节） | 端到端评测扩到文字题（大模型判分）和不告诉文档的设定；MinerU 作为对照解析器；图检索先留接口 |
| **画图、统计** | 已实现：`tools/python/`、`tools/r/` | `run_python` / `run_r` 在沙箱里跑，图存进 `work/figures/`；meta 分析有 RevMan 5 模板 |
| **沙箱表格编号** | 已实现：`tools/sql/results.py` + 两个内核的 `save_result()` | Python / R 里 `save_result(df, "标题")` 把表发给宿主，存进同一个结果仓库、接着 r 号往下编；回答里 `{{r5}}` 引用、`/save r5` 导出（没有 SQL 可重跑，直接写存下的行）、`load_result("r5")` 取回 |
| **流式输出** | 已实现：`LLMProvider.stream()` + `TextDelta` 事件 | 见「运行状态和流式输出」一节。工具参数的流式（边生成边显示 SQL）没做 |
| **人工审批** | 已实现：`build_application(approval_hook=...)` | 传个函数，工具执行前弹确认 |
| **自定义结束条件** | 已实现：`build_application(finish_turn_hook=...)` | 见概念 8 |
| **Web 界面** | 已实现：`web/`（后端 `src/data_agent/web/`） | 见「Web 界面」一节。没做：多用户和登录、工具执行到一半的取消、会话改名和删除 |
| **多 Agent** | 把 `Agent` 包成一个 `Tool` | 子 Agent 就是一个工具，天然递归 |
| **持久化会话** | 已实现：`session/` | 每轮成功之后把新增的历史追加进 `session.jsonl`，`--resume` 读回来。以后要分支（从某一轮重来），学 pi 给每条记录加 id / parentId |
| **中途问用户** | 已实现：`tools/ask_user.py`，`AwaitingUser` + `Agent.resume(回答)` | 以后 Web 界面：拿到 `AwaitingUser` 就把问题发给前端，下一个请求带着回答调 `resume` |
| **轮内检查点** | 已实现：`Agent.resume` + `session/store.py` | 回滚照旧，进度另外存（`checkpoint.json`），`/continue` 从断的地方接着跑。以后要「整个评测跑到一半接着跑」（跳过做完的 trial），是另一件事 |
| **Skills** | 已实现：`skills/` + `load_skill` | 系统提示词里只放目录（名字 + 一句话），模型需要时再读全文；以后要带脚本、参考文件，挂进沙箱的 /skills/ |
| **MCP** | 已实现：`mcp/`（见「MCP：外部工具服务器」一节） | HTTP 传输（Streamable HTTP）；服务器发来的「工具列表变了」通知；resources / prompts |
| **统一运行状态** | 已实现：`core/state.py` | 程序重启后从 `checkpoint.json` 恢复的那一轮不在状态里（没有事件），Web 界面时再补 |

接下来的顺序：~~轮内检查点~~ → ~~Skills~~ → ~~长期记忆~~ → ~~RAG（配 FinanceBench）~~ → ~~MCP~~ → ~~统一运行状态 + 流式输出~~ → ~~Web 界面~~。
每一样都要有评测证明它有用。

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

---

## 许可证

MIT，见 [LICENSE](LICENSE)。DABstep 的判分代码和测试（`evals/dabstep/scorer.py`、`tests/test_dabstep_scorer.py`）是 CC BY 4.0，
meta 分析的原始数据来自 metafor / meta 包收录的已发表研究，出处见 `evals/cases/gen_research.py`。
