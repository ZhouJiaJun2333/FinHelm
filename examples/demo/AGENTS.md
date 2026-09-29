- 数据库里有两份不相干的数据，表名要带 schema 前缀写（`shop.orders`、`financial.account`）：
  - `shop`：一个电商业务库（客户、商品、订单、订单明细）
  - `financial`：一家捷克银行的业务库（账户、客户、贷款、交易、信用卡）

## shop
- `shop.orders.status` 有三种值：`completed` / `cancelled` / `returned`。
  **算销售额、营收类指标时通常只算 `completed`**，除非用户明确说要含取消和退货。
  你做了哪种选择，必须在回答里讲清楚。
- 实付金额不是 `unit_price * quantity`，而是
  `quantity * unit_price * (1 - discount)`，别漏掉折扣。
- 毛利 = 实付金额 - `products.unit_cost` * quantity。
- `shop.customers.region` 有少量 NULL。做分区域统计时要么剔除、要么单列一类，
  两种做法都行，但要在回答里说明，并给出这部分的规模。

## financial
- 编码都是捷克语（比如 `POPLATEK MESICNE`、`VYDAJ`），含义写在列注释里 —— `describe_table` 能看到。
  用户说「月度出账」「取款」这类意思时，先查列注释对上编码，别猜。
- `district` 表的列叫 `a2`~`a16`，含义（区名、所属地区、人口、平均工资、失业率…）同样在列注释里。
- 客户和账户是多对多，通过 `disp` 连：`disp.type` 区分账户所有人（`OWNER`）和授权使用人（`DISPONENT`）。
  问「账户的所有人」要限定 `OWNER`。信用卡 `card` 挂在 `disp` 上，不直接挂在账户上。
- `loan`、`order`、`trans` 都挂在账户（`account_id`）上。`order` 是 SQL 关键字，写成 `financial."order"`。
