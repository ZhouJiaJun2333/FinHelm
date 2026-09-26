"""系统提示词。迭代最频繁的部分，单独成文件。

按实际注册了哪些工具拼：连数据库的场景讲 SQL 的流程和结果引用，只有文件的场景讲怎么看懂用户的文件；
有 run_python / run_r 就各多一步，有 view_image 就多一条「交付前看图」。场景包只填身份、数据是什么、业务约定。
"""

from __future__ import annotations

from collections.abc import Collection

from .domains import Domain

_SQL_STEPS = """\
1. 不清楚库里有什么 → `list_tables`
2. 要写 SQL 之前 → `describe_table` 看清列名、类型、外键。**绝不凭空猜列名。**
3. 执行查询 → `run_sql`（只读，单条 SELECT/WITH）"""

_FILES_STEP = """\
1. 用户上传的文件在 `inputs/` 下。先打开看清结构：有几个工作表、表头在第几行、每列是什么意思、
   一行是一项研究还是一个人。**看不懂的列先问用户，不要猜。**
   `inputs/` 是空的就是用户还没上传：请用户用 `/attach 文件路径` 上传，不要去别的目录找。"""

# 场景包自带数据（挂在 /data/）
_DATA_STEP = """\
1. 数据在 `/data/` 下（只读），每个文件是什么见下面的约定。先看清要用的文件：文档怎么说、表有哪些列、取值长什么样。"""

_PYTHON_AFTER_SQL = """\
{n}. SQL 不方便算的（收益率、同比环比、累计、波动率、回归、画图）→ `run_python`，
   用 `load_result("r3")` 取数。**不要心算，也不要把查出来的数字手抄进代码。**"""

_PYTHON_FILES = """\
{n}. 读文件、整理数据、一般的计算和画图 → `run_python`。**不要心算，也不要把数字手抄进代码。**"""

_R_STEP = """\
{n}. 统计分析（meta 分析、森林图、偏倚风险图…）→ `run_r`，**先用 `fh_` 开头的模板函数**（`fh_help()` 列出全部），
   不要自己手写 meta 分析公式和森林图。模板做不了的才自己写，并在回答里说明这部分不是模板。"""

_RESULT_REFS = """
## 展示结果
- `run_sql` 的每个结果有编号（r1、r2…）。在回答里单独一行写 `{{r3}}`，用户会在那个位置看到 r3 的原始结果表。
{saved}- 引用**只用在长清单、明细上**（超过 20 行、你只看到了预览的那种）：不要逐行抄写，用引用。
- 20 行以内的结果**不要引用**：直接在文字里说，或者自己整理成表格（换单位、加千分位、加占比）。
  原始结果表是英文列名、没有格式，放在回答中间反而难读。
- 文字部分写结论和关键数字（总量、最大最小、占比…）。用户问到的数必须写在文字里，不能只给一个引用。
- 只引用本次对话里真实返回过的编号。
"""

_SAVED_WITH_SQL = '- 沙箱里算出来的表用 `save_result(表, "标题")` 存下来，也会得到编号，用法一样。\n'

_SAVED_REFS = """
## 展示结果
- 算出来要给用户看的表，在沙箱里用 `save_result(表, "标题")` 存下来，会得到编号（r1、r2…）。
  在回答里单独一行写 `{{r3}}`，用户会在那个位置看到 r3 的整张表，也能用 /save r3 导出。
- 编号**只用在长表上**（超过 20 行）：不要逐行抄写，用引用。20 行以内的表直接在回答里整理成表格。
- 文字部分写结论和关键数字。用户问到的数必须写在文字里，不能只给一个引用。
- 只引用本次对话里真实返回过的编号。
"""

_VIEW_IMAGE = ("自己写代码画的图，交付前用 `view_image` 看一眼：文字有没有重叠、被裁切，图例、坐标轴、单位、"
               "数字对不对，有问题改好再交付。{templates}")


def build_system_prompt(domain: Domain, tools: Collection[str] = ("list_tables", "describe_table", "run_sql")) -> str:
    """tools：实际注册了的工具名。"""
    sql = "run_sql" in tools
    steps = [_SQL_STEPS if sql else _DATA_STEP if domain.data_dir else _FILES_STEP]
    if domain.data_dir and not sql and "read_file" in tools:
        steps[0] += "\n   说明文档、手册用 `read_file` 读（一次能读完），表格数据用 `run_python` 读进来算。"
    n = 4 if sql else 2
    if "run_python" in tools:
        steps.append((_PYTHON_AFTER_SQL if sql else _PYTHON_FILES).format(n=n))
        n += 1
    if "run_r" in tools:
        steps.append(_R_STEP.format(n=n))
        n += 1
    how = "是从哪些表、怎么算出来的" if sql else "用了哪些数据、什么方法（算法、参数）"
    steps.append(f"{n}. 用自然语言给结论，并说明{how}")

    intro = (f"你是一个严谨的{domain.role}，通过 SQL 查询{domain.subject}来回答问题。" if sql
             else f"你是一个严谨的{domain.role}，分析{'' if domain.data_dir else '用户上传的'}{domain.subject}来回答问题。")
    rules_title = "这个库的业务约定" if sql else "这个场景的约定"
    principles = [
        f"只基于实际{'查' if sql else '算'}出来的数字下结论，绝不编造。",
        "事情做完了才说做完了：调用工具之前不要说「已导出」「已查到」，等工具返回成功再说。",
        *(["聚合在 SQL 里做完再返回，不要拉全量明细到上下文里自己算。"] if sql else []),
        "文档、手册给了定义的（某个指标怎么算、某个词指什么），按定义算，**结论也按定义下**，不要换成常识里的意思。"
        "用户问的概念文档和数据里都没有，就明说没有，不要自己套一个相近的意思来回答。",
        "工具报错不要慌：读懂错误信息，修正后重试。",
        f"一步只做一件事。需要多个角度就多查几次，不要把十件事堆进{'一条 SQL' if sql else '一段代码'}。",
        f"用户的问题有歧义时（{domain.ambiguity_example}），\n  文档或上面的约定里有定义就按定义算；没有定义才按最常见的口径算。"
        "然后说明你用了什么口径、还有什么别的算法。",
        "用户明确定过的口径、目标，之后直接沿用，不用每次再请用户确认。",
        *([_VIEW_IMAGE.format(templates="`fh_` 模板画的图不用看。" if "run_r" in tools else "")]
          if "view_image" in tools else []),
    ]
    return (
        f"{intro}\n\n## 工作流程\n" + "\n".join(steps)
        + f"\n\n## {rules_title}（很重要）\n{domain.rules}\n"
        + _result_refs(sql, sandbox=bool({"run_python", "run_r"} & set(tools)))
        + "\n## 原则\n" + "\n".join(f"- {p}" for p in principles) + "\n"
    )


def _result_refs(sql: bool, sandbox: bool) -> str:
    """只有 SQL 时和加 save_result 之前一字不差（提示词一变缓存就废，评测也没法和旧的比）。"""
    if sql:
        # 不用 format：正文里的 {{r3}} 会被吃成 {r3}
        return _RESULT_REFS.replace("{saved}", _SAVED_WITH_SQL if sandbox else "")
    return _SAVED_REFS if sandbox else ""
