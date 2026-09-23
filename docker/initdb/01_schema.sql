-- 电商订单库：四张表，足够练各种 JOIN / 聚合 / 窗口函数
-- 表注释和列注释都写上 —— Agent 靠 COMMENT 理解业务含义，这是最廉价的「喂上下文」

CREATE SCHEMA IF NOT EXISTS shop;
SET search_path TO shop, public;

-- ---------------------------------------------------------------- customers
CREATE TABLE customers (
    customer_id   INT PRIMARY KEY,
    customer_name TEXT        NOT NULL,
    city          TEXT,
    region        TEXT,
    segment       TEXT,
    signup_date   DATE        NOT NULL
);
COMMENT ON TABLE  customers               IS '客户主表，一行一个客户';
COMMENT ON COLUMN customers.region        IS '大区：华东/华北/华南/西南，少量历史数据为空';
COMMENT ON COLUMN customers.segment       IS '客户分层：个人/企业/政府';
COMMENT ON COLUMN customers.signup_date   IS '注册日期';

-- ----------------------------------------------------------------- products
CREATE TABLE products (
    product_id   INT PRIMARY KEY,
    product_name TEXT           NOT NULL,
    category     TEXT           NOT NULL,
    unit_price   NUMERIC(10, 2) NOT NULL,
    unit_cost    NUMERIC(10, 2) NOT NULL
);
COMMENT ON TABLE  products             IS '商品主表';
COMMENT ON COLUMN products.unit_price  IS '标准售价（元），实际成交价见 order_items.unit_price';
COMMENT ON COLUMN products.unit_cost   IS '成本价（元），算毛利用这个';

-- ------------------------------------------------------------------- orders
CREATE TABLE orders (
    order_id    INT PRIMARY KEY,
    customer_id INT         NOT NULL REFERENCES customers (customer_id),
    order_date  DATE        NOT NULL,
    channel     TEXT        NOT NULL,
    status      TEXT        NOT NULL
);
COMMENT ON TABLE  orders            IS '订单头表，一行一单';
COMMENT ON COLUMN orders.channel    IS '渠道：线上/线下/分销';
COMMENT ON COLUMN orders.status     IS '状态：completed 已完成 / cancelled 已取消 / returned 已退货。算销售额时通常只算 completed';

-- -------------------------------------------------------------- order_items
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id      INT            NOT NULL REFERENCES orders (order_id),
    product_id    INT            NOT NULL REFERENCES products (product_id),
    quantity      INT            NOT NULL,
    unit_price    NUMERIC(10, 2) NOT NULL,
    discount      NUMERIC(4, 3)  NOT NULL DEFAULT 0
);
COMMENT ON TABLE  order_items            IS '订单明细表，一行一个商品行；一个订单可能有多行';
COMMENT ON COLUMN order_items.unit_price IS '实际成交单价（元）';
COMMENT ON COLUMN order_items.discount   IS '折扣率，0~0.3；实付金额 = quantity * unit_price * (1 - discount)';

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date     ON orders (order_date);
CREATE INDEX idx_items_order     ON order_items (order_id);
CREATE INDEX idx_items_product   ON order_items (product_id);
