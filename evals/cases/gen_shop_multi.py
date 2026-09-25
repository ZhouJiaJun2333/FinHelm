"""生成 evals/cases/shop_multi.jsonl（多轮会话题库）。改题改这里，然后在项目根目录跑：

    python evals/cases/gen_shop_multi.py

SQL 片段拼出来比手写一整行 JSON 好改。生成的 jsonl 也提交，评测只读 jsonl。
"""
import json

AMT = "oi.quantity * oi.unit_price * (1 - oi.discount)"
LIST = "oi.quantity * oi.unit_price"
FROM = "FROM shop.orders o JOIN shop.order_items oi ON oi.order_id = o.order_id"
JC = " JOIN shop.customers c ON c.customer_id = o.customer_id"
JP = " JOIN shop.products p ON p.product_id = oi.product_id"
DONE = "o.status = 'completed'"
Y24 = "o.order_date >= DATE '2024-01-01' AND o.order_date < DATE '2025-01-01'"
Y25 = "o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'"
Q123_25 = "o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2025-10-01'"
SEP25 = "o.order_date >= DATE '2025-09-01' AND o.order_date < DATE '2025-10-01'"


# 填充题：每条 50~130 行。起初工具结果接近 6000 字的上限；0013876 之后模型只看前 10 行，
# 回答里用 {{r3}} 引用，每轮只涨 0.7~1k
F = {
    "华东9月订单": "把 2025 年 9 月华东区所有已完成的订单列出来：订单号、下单日期、客户名、实付金额，按日期排。",
    "华北客户": "列出所有华北的客户：客户名、城市、客户分层、注册日期。",
    "24年12月线上订单": "列出 2024 年 12 月线上渠道的全部已完成订单：订单号、日期、客户名、实付金额。",
    "25年退货订单": "列出 2025 年所有退货的订单：订单号、日期、渠道、客户名。",
    "25年8月线上订单": "列出 2025 年 8 月线上渠道的全部订单：订单号、日期、状态、客户名、城市。",
    "华东客户": "列出所有华东的客户：客户名、城市、客户分层、注册日期。",
    "24年12月明细": "把 2024 年 12 月已完成订单的明细列出来：订单号、商品名、数量、成交单价、折扣。",
    "25年7月订单": "列出 2025 年 7 月的全部已完成订单：订单号、日期、渠道、客户名。",
    "西南华南客户": "列出所有西南和华南的客户：客户名、城市、区域、客户分层。",
    "25年取消订单": "列出 2025 年所有被取消的订单：订单号、日期、渠道、客户名。",
    "24年6月明细": "把 2024 年 6 月的订单明细列出来：订单号、商品名、数量、成交单价、折扣。",
    "全部商品": "列出全部商品：商品编号、商品名、品类、标准售价、成本价。",
    "政府客户": "列出所有政府客户：客户名、城市、区域、注册日期。",
    "24年3月线下订单": "列出 2024 年 3 月线下渠道的全部已完成订单：订单号、日期、客户名、城市。",
    "25年5月分销订单": "列出 2025 年 5 月分销渠道的全部订单：订单号、日期、状态、客户名。",
}


def fill(*names):
    return [{"question": F[n], "filler": True} for n in names]


# 门槛按真实配置的比例缩小（真实：清理 10 万 / 保留 3 条 / 至少 1 万；压缩 15 万 / 保留 2 万）。
# 缩多少看单个工具结果多大：门槛太低会变成每来一条结果就清一次，缓存数字失真。
#   · 起初结果最多约 4k token（6000 字上限），门槛 3 万 / 4.5 万（1 万时一轮里清了 5 次）
#   · run_sql 改成只给模型预览后（0013876），结果最多约 1.5k，每轮只涨 0.7~1k，
#     3 万再也碰不到 → 降到 1 万 / 1.5 万试了一次：都触发了但太晚（B、C 最后一轮才压缩，
#     回忆轮大多在压缩之前），再降到 8000 / 1.2 万
CLEAR = {"context_clear_trigger_tokens": 8000, "context_keep_tool_results": 3,
         "context_clear_at_least": 2000}
COMPACT = {"context_compact_trigger_tokens": 12000, "context_compact_keep_recent_tokens": 2500}
NO_CLEAR = {"context_clear_trigger_tokens": 1000000}
NO_COMPACT = {"context_compact_trigger_tokens": 1000000}


def by_region(expr, when):
    base = f"SELECT c.region, {expr} {FROM}{JC} WHERE {DONE} AND {when}"
    return [f"{base} GROUP BY c.region", f"{base} AND c.region IS NOT NULL GROUP BY c.region"]


A = {
    "id": "multi-001",
    "note": "只开清理：大结果把早先的工具结果挤出去之后，回忆第一轮的数、再拿它算占比",
    "tags": ["清理"],
    "settings": {**CLEAR, **NO_COMPACT},
    "turns": [
        {"question": "2024 年各区域的销售额是多少？",
         "gold_sql": by_region(f"SUM({AMT})", Y24), "tags": ["分组"]},
        {"question": "那 2025 年前三季度呢？",
         "gold_sql": by_region(f"SUM({AMT})", Q123_25), "tags": ["指代"]},
        *fill("华东9月订单"),
        {"question": "这些订单里，消费最多的客户是谁？一共花了多少？", "match": "top",
         "gold_sql": [
             f"SELECT c.customer_name, SUM({AMT}) AS s {FROM}{JC} WHERE {DONE} AND {SEP25} AND c.region = '华东' "
             "GROUP BY c.customer_id, c.customer_name ORDER BY s DESC LIMIT 1"],
         "tags": ["指代"]},
        *fill("华北客户", "24年12月线上订单", "全部商品"),
        {"question": "2024 年各区域的客单价是多少？",
         "gold_sql": by_region(f"SUM({AMT}) / COUNT(DISTINCT o.order_id)", Y24), "tags": ["分组"]},
        # 第一次校准时清理到第 13 轮才发生，回忆轮在它前面 —— 多塞 4 条，保证回忆时第 1 轮的结果已经清掉
        *fill("政府客户", "24年3月线下订单", "华东客户", "25年退货订单", "25年8月线上订单",
              "24年12月明细", "25年取消订单"),
        {"question": "回到第一个问题：2024 年华东的销售额是多少？", "match": "answer",
         "answer_sql": f"SELECT SUM({AMT}) {FROM}{JC} WHERE {DONE} AND {Y24} AND c.region = '华东'",
         "tags": ["回忆"]},
        {"question": "华东 2024 年的销售额占全国的百分之多少？", "match": "answer",
         "answer_sql": f"SELECT SUM({AMT}) FILTER (WHERE c.region = '华东') / SUM({AMT}) {FROM}{JC} "
                       f"WHERE {DONE} AND {Y24}",
         "tags": ["计算"],
         "note": "全国含区域为空的客户"},
    ],
}

RETURN_RATE_ALL = "SELECT channel, COUNT(*) FILTER (WHERE status = 'returned')::numeric / COUNT(*) FROM shop.orders GROUP BY channel"
RET_OFFLINE_25 = f"{FROM}{JP} WHERE o.channel = '线下' AND o.status = 'returned' AND {Y25}"
B = {
    "id": "multi-002",
    "note": "清理 + 压缩都开：大结果一路塞，先清理、清理跟不上再压缩；之后回忆算过的比例，再追问依赖它的结论",
    "tags": ["清理", "压缩"],
    "settings": {**CLEAR, **COMPACT},
    "turns": [
        # 「增长 12%」只在对话里，库里查不到 —— 最后一轮考摘要有没有记住它
        {"question": "我们给 2025 年定的销售目标是比 2024 年增长 12%。先看看 2024 年各渠道的销售额是多少？",
         "gold_sql": f"SELECT o.channel, SUM({AMT}) {FROM} WHERE {DONE} AND {Y24} GROUP BY o.channel",
         "tags": ["分组"]},
        # 写明「全部年份」：上一轮问的是 2024 年，不写的话 Agent 延续 2024 也说得通
        # （2026-09-25 发现：第 19 轮时对时错，就是两种理解各占一半）
        {"question": "各渠道的退货率是多少？看全部年份，退货订单数占全部订单数的比例。",
         "gold_sql": [RETURN_RATE_ALL,
                      "SELECT channel, COUNT(*) FILTER (WHERE status = 'returned'), COUNT(*) FROM shop.orders GROUP BY channel"],
         "answer_sql": RETURN_RATE_ALL, "tags": ["比例"],
         "note": "查出退货数和总数、在回答里自己除，取数也算对；回答里必须有比例"},
        *fill("25年退货订单", "25年8月线上订单", "华东客户", "24年12月明细", "25年7月订单",
              "西南华南客户", "25年取消订单", "24年6月明细", "全部商品", "政府客户",
              "24年3月线下订单", "25年5月分销订单",
              # 基线（df78559）里 2 次只压了 1 次：清理把上下文一直压在 4.5 万以下。多塞 3 条保证压缩
              "华东9月订单", "华北客户", "24年12月线上订单"),
        {"question": "刚才算的线上渠道退货率是多少？", "match": "answer",
         "answer_sql": "SELECT COUNT(*) FILTER (WHERE status = 'returned')::numeric / COUNT(*) FROM shop.orders WHERE channel = '线上'",
         "tags": ["回忆"]},
        {"question": "退货率最高的那个渠道，2025 年退货的商品里，哪个品类的件数最多？", "match": "top",
         "gold_sql": f"SELECT p.category, SUM(oi.quantity) AS n {RET_OFFLINE_25} GROUP BY p.category ORDER BY n DESC LIMIT 1",
         "tags": ["指代"],
         "note": "退货率最高的是线下（第 2 轮算过）。按件数问：按订单数、明细行数，笔记本和耳机并列"},
        {"question": "按我一开始说的增长目标，2025 年全年的销售额目标应该是多少？", "match": "answer",
         "answer_sql": f"SELECT SUM({AMT}) * 1.12 {FROM} WHERE {DONE} AND {Y24}",
         "tags": ["回忆:查不到"],
         "note": "12% 是用户在第 1 轮顺口说的，库里没有，只能靠摘要记住；2024 年的销售额可以重查"},
    ],
}

C = {
    "id": "multi-003",
    "note": "只开压缩：第一轮用户改了销售额口径（不扣折扣），压缩之后还守不守",
    "tags": ["压缩", "用户约定"],
    "settings": {**NO_CLEAR, **COMPACT},
    "turns": [
        {"question": "先说好：接下来我说的「销售额」都按成交价算，不扣折扣，也就是 数量 × 成交单价，"
                     "其他口径照旧。2024 年的销售额是多少？", "match": "contains",
         "gold_sql": f"SELECT SUM({LIST}) {FROM} WHERE {DONE} AND {Y24}", "tags": ["用户约定"]},
        # 只在对话里的数：不判分，最后一轮要用
        {"question": "再记一下：我们给线上渠道定的毛利率目标是 35%，后面算目标会用到。", "filler": True},
        *fill("25年7月订单", "西南华南客户", "25年取消订单", "24年6月明细", "24年12月线上订单",
              "华北客户", "政府客户",
              # 基线（df78559）里 2 次有 1 次没到 4.5 万、压根没压缩。多塞 3 条保证压缩
              "全部商品", "25年5月分销订单", "24年3月线下订单",
              # 输出上限调到 32768 之后（aae61d1），第 5 步完成那次 2 次里有 1 次没到 4.5 万
              "25年退货订单", "25年8月线上订单"),
        {"question": "2025 年线上渠道的销售额是多少？", "match": "contains",
         "gold_sql": f"SELECT SUM({LIST}) {FROM} WHERE {DONE} AND {Y25} AND o.channel = '线上'",
         "tags": ["用户约定"], "note": "按第一轮的约定不扣折扣"},
        {"question": "2025 年各品类的销售额从高到低排一下。", "match": "ordered",
         "gold_sql": f"SELECT p.category, SUM({LIST}) AS s {FROM}{JP} WHERE {DONE} AND {Y25} GROUP BY p.category ORDER BY s DESC",
         "tags": ["用户约定"]},
        {"question": "第一个问题的答案是多少来着？", "match": "answer",
         "answer_sql": f"SELECT SUM({LIST}) {FROM} WHERE {DONE} AND {Y24}", "tags": ["回忆"]},
        {"question": "2025 年线上渠道的销售额，按我说的毛利率目标，对应的毛利目标是多少？", "match": "answer",
         "answer_sql": f"SELECT SUM({LIST}) * 0.35 {FROM} WHERE {DONE} AND {Y25} AND o.channel = '线上'",
         "tags": ["回忆:查不到", "用户约定"],
         "note": "35% 只在第 2 轮的对话里；销售额还得按第 1 轮的约定不扣折扣"},
    ],
}

with open("evals/cases/shop_multi.jsonl", "w", encoding="utf-8") as f:
    for s in (A, B, C):
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
