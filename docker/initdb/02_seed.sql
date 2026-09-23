-- 造假数据。全部用 SQL 生成，容器首次启动时自动跑，不需要额外的 Python 脚本。
--
-- ⚠️⚠️ 这个文件里最重要的一课（我连踩两次）：
--
--    CROSS JOIN LATERAL (SELECT ... ORDER BY random() LIMIT 1) AS x
--
--    如果 LATERAL 子查询里**没有引用外层的任何列**，PG 就认为它是常量，
--    整个查询只求值一次 —— 结果 200 个客户全在同一个城市、
--    12000 个订单行全是同一个商品。而且它不报错，你只能靠查 count(DISTINCT) 发现。
--
--    解决办法：用一个**基于外层主键的哈希**来选行，这样它必然是相关子查询，
--    按行求值，而且结果可重现（不像 random() 每次重建数据都变）。
--    下面的 pick_index() 就是干这个的。

SET search_path TO shop, public;
SELECT setseed(0.42);

-- 把任意文本映射成 [0, modulus) 的整数。
-- 'x' || 7位十六进制 -> bit(28) -> int 是 PG 里把 md5 转成整数的常用写法。
CREATE FUNCTION pg_temp.pick_index(seed TEXT, modulus INT) RETURNS INT AS $$
    SELECT (('x' || substr(md5(seed), 1, 7))::bit(28)::int % modulus);
$$ LANGUAGE SQL IMMUTABLE;


-- ---------------------------------------------------------------- customers
INSERT INTO customers (customer_id, customer_name, city, region, segment, signup_date)
SELECT
    i,
    '客户' || lpad(i::text, 4, '0'),
    picked.city,
    -- 故意留 ~5% 的 region 为空，让 Agent 有机会发现数据质量问题
    CASE WHEN pg_temp.pick_index('region-null-' || i, 20) = 0 THEN NULL ELSE picked.region END,
    (ARRAY['个人', '企业', '政府'])[1 + pg_temp.pick_index('seg-' || i, 3)],
    DATE '2023-01-01' + pg_temp.pick_index('signup-' || i, 700)
FROM generate_series(1, 200) AS i
CROSS JOIN LATERAL (
    SELECT c.city, c.region
    FROM (VALUES
        ('上海', '华东'), ('杭州', '华东'), ('南京', '华东'), ('苏州', '华东'),
        ('北京', '华北'), ('天津', '华北'), ('青岛', '华北'),
        ('广州', '华南'), ('深圳', '华南'), ('厦门', '华南'),
        ('成都', '西南'), ('重庆', '西南'), ('昆明', '西南')
    ) AS c(city, region)
    -- 注意这里引用了外层的 i —— 这才让它成为按行求值的相关子查询
    OFFSET pg_temp.pick_index('city-' || i, 13) LIMIT 1
) AS picked;


-- ----------------------------------------------------------------- products
INSERT INTO products (product_id, product_name, category, unit_price, unit_cost)
SELECT
    i,
    picked.cat || '-' || (ARRAY['入门款', '标准款', '专业款', '旗舰款'])[1 + (i % 4)],
    picked.cat,
    picked.price,
    -- 毛利率 35%~60% 之间。
    -- random() 返回 double precision，会把 numeric 也带成 double，
    -- 而 PG 没有 round(double, int) 这个重载，所以必须显式 ::numeric。
    round((picked.price * (0.40 + random() * 0.25))::numeric, 2)
FROM generate_series(1, 40) AS i
CROSS JOIN LATERAL (
    SELECT p.cat, p.price
    FROM (VALUES
        ('笔记本',  6999.00), ('显示器', 1899.00), ('键盘',  399.00),
        ('耳机',     899.00), ('平板',   3299.00), ('打印机', 1299.00),
        ('摄像头',   599.00), ('扩展坞',  749.00)
    ) AS p(cat, price)
    OFFSET (i % 8) LIMIT 1
) AS picked;


-- ------------------------------------------------------------------- orders
INSERT INTO orders (order_id, customer_id, order_date, channel, status)
SELECT
    i,
    1 + pg_temp.pick_index('cust-' || i, 200),
    DATE '2024-01-01' + pg_temp.pick_index('date-' || i, 640),
    (ARRAY['线上', '线上', '线上', '线下', '线下', '分销'])[
        1 + pg_temp.pick_index('chan-' || i, 6)
    ],
    -- 约 85% 完成、10% 取消、5% 退货
    CASE
        WHEN pg_temp.pick_index('status-' || i, 100) < 85 THEN 'completed'
        WHEN pg_temp.pick_index('status-' || i, 100) < 95 THEN 'cancelled'
        ELSE 'returned'
    END
FROM generate_series(1, 5000) AS i;


-- -------------------------------------------------------------- order_items
-- 每单 1~4 个商品行。行数和商品都由 order_id 的哈希决定 —— 必须相关，理由见文件头。
INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price, discount)
SELECT
    row_number() OVER (ORDER BY o.order_id, n)                   AS order_item_id,
    o.order_id,
    p.product_id,
    1 + pg_temp.pick_index('qty-' || o.order_id || '-' || n, 5)  AS quantity,
    -- 实际成交价在标准价 ±8% 浮动
    round((p.unit_price * (0.92 + random() * 0.16))::numeric, 2) AS unit_price,
    round((pg_temp.pick_index('disc-' || o.order_id || '-' || n, 301) / 1000.0)::numeric, 3) AS discount
FROM orders AS o
CROSS JOIN LATERAL generate_series(
    1, 1 + pg_temp.pick_index('nitems-' || o.order_id, 4)
) AS n
-- 商品 id 和价格必须取自**同一行**，否则价格对不上商品
CROSS JOIN LATERAL (
    SELECT pr.product_id, pr.unit_price
    FROM products AS pr
    WHERE pr.product_id = 1 + pg_temp.pick_index('prod-' || o.order_id || '-' || n, 40)
) AS p;

ANALYZE;
