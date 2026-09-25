"""场景包：假电商库（docker/initdb 造的数据）。"""

from . import Domain

SHOP = Domain(
    name="shop",
    schema="shop",
    subject="一个电商业务库",
    rules="""\
- `orders.status` 有三种值：`completed` / `cancelled` / `returned`。
  **算销售额、营收类指标时通常只算 `completed`**，除非用户明确说要含取消和退货。
  你做了哪种选择，必须在回答里讲清楚。
- 实付金额不是 `unit_price * quantity`，而是
  `quantity * unit_price * (1 - discount)`，别漏掉折扣。
- 毛利 = 实付金额 - `products.unit_cost` * quantity。
- `customers.region` 有少量 NULL。做分区域统计时要么剔除、要么单列一类，
  两种做法都行，但要在回答里说明，并给出这部分的规模。""",
)
