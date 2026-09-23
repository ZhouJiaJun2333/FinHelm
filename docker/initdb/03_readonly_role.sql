-- Agent 专用的只读账号。
--
-- 为什么不让 Agent 直接用 postgres 超级用户？
--   run_sql 工具里那层「只允许 SELECT」的关键字检查是可以被绕过的（注释、大小写、
--   奇怪的语法…）。真正可靠的防线在数据库这一侧：**给它一个物理上就没有写权限的账号**。
--   这叫纵深防御 —— 应用层拦一道，数据库层再拦一道，不指望任何单点。

CREATE ROLE agent_ro WITH LOGIN PASSWORD 'agent_ro_pwd';

-- 只给连接权和读权，不给任何写权
GRANT CONNECT ON DATABASE analytics TO agent_ro;
GRANT USAGE  ON SCHEMA shop         TO agent_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA shop TO agent_ro;

-- 以后新建的表也自动带上只读权限
ALTER DEFAULT PRIVILEGES IN SCHEMA shop GRANT SELECT ON TABLES TO agent_ro;

-- 不允许在 public 里建表（默认权限在 PG15+ 已经收紧，这里再确认一次）
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- 单条语句最多跑 30 秒，防止一个笛卡尔积把库拖死
ALTER ROLE agent_ro SET statement_timeout = '30s';
-- 连上来就默认只读事务
ALTER ROLE agent_ro SET default_transaction_read_only = on;
-- 让它能看到 shop schema，写 SQL 时不用每次都加前缀
ALTER ROLE agent_ro SET search_path = shop, public;
