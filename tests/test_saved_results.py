"""沙箱里的表也编 r 号：save_result() 存进结果仓库，回答里 {{r5}} 引用、/save 导出、load_result 取回。

Python 用本机进程（Sandbox.local），不需要 Docker；R 的在最后，没有镜像就跳过。
"""

from __future__ import annotations

import csv
import shutil
import subprocess

import pytest

from data_agent.db.connection import QueryResult
from data_agent.domains import get_domain
from data_agent.prompts import build_system_prompt
from data_agent.tools.python import PYTHON_KERNEL
from data_agent.tools.r import R_KERNEL
from data_agent.tools.sandbox import Sandbox
from data_agent.tools.sql.export_csv import export_result
from data_agent.tools.sql.results import ResultStore, result_resolver, result_saver


@pytest.fixture(scope="module")
def py(tmp_path_factory):
    results = ResultStore()
    results.add("SELECT 1", QueryResult(["x"], [(1,)], False, 1))                  # r1 是 SQL 的
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path_factory.mktemp("work"), result_resolver(results),
                            save=result_saver(results, "python"))
    sandbox.results = results
    yield sandbox
    sandbox.close()


# ================================================================ Python
def test_存一张分组结果_接着SQL的编号往下编(py):
    ex = py.run("df = pd.DataFrame({'区域': ['华东', '华北', '华东'], 'gmv': [1.5, None, 2.0]})\n"
                "save_result(df.groupby('区域', sort=False).gmv.sum(min_count=1), '各区域 GMV')")
    assert ex.error is None, ex.error
    assert "已存为结果 r2：2 行 × 2 列" in ex.output and "{{r2}}" in ex.output
    table = py.results.get("r2")
    assert (table.source, table.title, table.sql) == ("python", "各区域 GMV", "")
    assert table.result.columns == ["区域", "gmv"], "分组列在索引里，要放回来"
    assert table.result.rows == [("华东", 3.5), ("华北", None)], "NaN 存成空值"


def test_存下的表能load_result取回(py):
    ex = py.run("save_result(pd.DataFrame({'d': pd.to_datetime(['2024-01-02']), 'n': [3]}), '日期')\n"
                "back = load_result('r3')\nprint(back.n.sum(), back.d[0][:10])")
    assert ex.error is None, ex.error
    assert ex.output.strip().endswith("3 2024-01-02")


def test_空表不存_报错告诉模型(py):
    ex = py.run("save_result(pd.DataFrame({'a': []}))")
    assert ex.error and "表是空的" in ex.error


def test_宿主没接存储时报错而不是卡住(tmp_path):
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path, lambda ref: {"error": "no"})
    try:
        ex = sandbox.run("save_result(pd.DataFrame({'a': [1]}))")
        assert ex.error and "不能存结果编号" in ex.error
    finally:
        sandbox.close()


# ================================================================ 仓库、导出
def test_来源和标题跟着日志存盘读回(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultStore(path)
    store.add("SELECT 1", QueryResult(["x"], [(1,)], False, 1))
    result_saver(store, "r")({"title": "亚组", "columns": ["亚组", "RR"], "rows": [["随机", 0.37]]})
    again = ResultStore(path)
    assert (again.get("r1").source, again.get("r2").source, again.get("r2").title) == ("sql", "r", "亚组")
    assert "source" not in path.read_text(encoding="utf-8").splitlines()[0], "SQL 结果的日志行和以前一样"
    store.add("SELECT 2", QueryResult(["x"], [(2,)], False, 1))
    assert again.refs() == ["r1", "r2"] and ResultStore(path).refs()[-1] == "r3"


def test_太长的表只存前一万行():
    store = ResultStore()
    reply = result_saver(store, "python")({"columns": ["i"], "rows": [[i] for i in range(10_005)]})
    assert reply == {"ref": "r1", "rows": 10_000, "truncated": True}


def test_沙箱的表直接写CSV_SQL的没有库就说清楚(tmp_path):
    store = ResultStore()
    result_saver(store, "python")({"title": "t", "columns": ["区域", "gmv"], "rows": [["华东", 3.5]]})
    done = export_result(None, store.get("r1"), tmp_path, "")
    with open(done.path, encoding="utf-8-sig") as f:
        assert list(csv.reader(f)) == [["区域", "gmv"], ["华东", "3.5"]]
    assert "重新查询" not in done.describe("r1")
    store.add("SELECT 1", QueryResult(["x"], [(1,)], False, 1))
    with pytest.raises(ValueError, match="没有连数据库"):
        export_result(None, store.get("r2"), tmp_path, "x.csv")


# ================================================================ 提示词
def test_提示词_有沙箱才讲save_result():
    assert "save_result" not in build_system_prompt(["run_sql"]), "只有 SQL 的提示词不变"
    assert "save_result" in build_system_prompt(["run_sql", "run_python"])
    research = build_system_prompt(["run_python", "run_r"])
    assert "## 展示结果" in research and "save_result" in research and "{{r3}}" in research
    assert "## 展示结果" not in build_system_prompt([])


# ================================================================ R
def _r_image_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", "finhelm-sandbox-r"],
                              capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _r_image_ready(), reason="没有 Docker 或 finhelm-sandbox-r 镜像")
def test_R也能存表_行名和因子都处理(tmp_path):
    store = ResultStore()
    sandbox = Sandbox.docker("finhelm-sandbox-r", R_KERNEL, tmp_path, result_resolver(store),
                             save=result_saver(store, "r"), timeout_s=120)
    try:
        ex = sandbox.run('m <- data.frame(RR = c(0.37, NA), d = as.Date(c("2024-01-02", NA)), '
                         'g = factor(c("a", "b")), row.names = c("随机", "交替"))\n'
                         'ref <- save_result(m, "亚组")\ncat(ref, nrow(load_result(ref)), "\\n")')
        assert ex.error is None, ex.error
        assert "已存为结果 r1：2 行 × 4 列" in ex.output and ex.output.strip().endswith("r1 2")
        table = store.get("r1")
        assert table.result.columns == ["项目", "RR", "d", "g"]
        assert table.result.rows == [("随机", 0.37, "2024-01-02", "a"), ("交替", None, None, "b")]
        assert (table.source, table.title) == ("r", "亚组")
    finally:
        sandbox.close()

