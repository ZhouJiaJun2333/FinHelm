"""run_python 和它的沙箱。

大部分用 Sandbox.local（本机进程，协议和容器里一样），不需要 Docker；
隔离本身（断网、只读、内存上限）只有容器里才有，放在最后，没有 Docker 或镜像就跳过。
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from data_agent.app import build_application
from data_agent.db.connection import QueryResult
from data_agent.tools.python import PYTHON_KERNEL, RunPythonTool
from data_agent.tools.sandbox import Execution, Sandbox, SandboxUnavailable
from data_agent.tools.sql.results import ResultStore, encode_result, result_resolver

from fakes import ScriptedProvider, eval_settings

MONTHLY = QueryResult(
    ["month", "gmv", "note"],
    [(dt.date(2024, m, 1), Decimal(f"{1000 + m}.25"), None if m % 2 else "促销") for m in range(1, 13)],
    False, 3,
)


def store_with_monthly() -> ResultStore:
    results = ResultStore()
    results.add("SELECT month, gmv, note FROM monthly", MONTHLY)      # r1
    return results


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    """几个测试共用一个内核：启动要 import pandas，一秒左右。"""
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path_factory.mktemp("work"), result_resolver(store_with_monthly()))
    yield sandbox
    sandbox.close()


# ---------------------------------------------------------------- 内核
def test_变量在调用之间保留_最后一行是表达式就显示它的值(kernel):
    first = kernel.run("x = 40\nprint('算好了')")
    assert first.output == "算好了\n" and first.value is None
    assert kernel.run("x + 2").value == "42"


def test_load_result拿到完整结果_类型也对(kernel):
    ex = kernel.run("df = load_result('r1')\nprint(len(df), df.gmv.dtype, df.month.dtype.kind)\n"
                    "df.gmv.sum()")
    assert ex.error is None, ex.error
    assert ex.output.split() == ["12", "float64", "M"]     # Decimal → float，日期 → datetime64
    assert float(ex.value) == pytest.approx(sum(1000 + m + 0.25 for m in range(1, 13)))


def test_load_result每次给副本_改了不影响下次取(kernel):
    kernel.run("a = load_result('r1')\na['gmv'] = 0")
    assert kernel.run("load_result('r1').gmv.iloc[0]").value == "1001.25"


def test_没有这个编号时告诉模型有哪些(kernel):
    ex = kernel.run("load_result('r9')")
    assert "LookupError" in ex.error and "r1" in ex.error


def test_报错只留用户代码的帧(kernel):
    ex = kernel.run("df = load_result('r1')\ndf['不存在的列']")
    assert "KeyError" in ex.error and "line 2" in ex.error
    assert "site-packages" not in ex.error and "pandas" not in ex.error


def test_语法错误(kernel):
    ex = kernel.run("for i in range(3)\n    print(i)")
    assert "SyntaxError" in ex.error


def test_用户代码退出不了内核_也读不到协议(kernel):
    assert "SystemExit" in kernel.run("import sys\nsys.exit(1)").error
    assert "input()" in kernel.run("input('你好？')").error
    # 直接写 fd 1 的内容被转去 stderr，协议没乱
    assert kernel.run("import os\nos.write(1, b'garbage\\n')\n1 + 1").value == "2"
    assert kernel.run("x").value == "40"


def test_硬超时_杀掉重来(tmp_path):
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path, result_resolver(ResultStore()), timeout_s=0.5, grace_s=0.5)
    try:
        sandbox.run("y = 1")
        # 屏蔽软超时，模拟卡在 C 代码里
        ex = sandbox.run("import signal, time\n"
                         "if hasattr(signal, 'SIGALRM'): signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
                         "time.sleep(30)")
        assert ex.restarted and "强制重启" in ex.error
        assert "NameError" in sandbox.run("y").error, "新内核是空的"
    finally:
        sandbox.close()


@pytest.mark.skipif(sys.platform == "win32", reason="软超时靠 SIGALRM，Windows 上没有")
def test_软超时_打断代码但变量还在(tmp_path):
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path, result_resolver(ResultStore()), timeout_s=0.5, grace_s=5)
    try:
        sandbox.run("y = 1")
        ex = sandbox.run("while True: pass")
        assert not ex.restarted and "被中断" in ex.error
        assert sandbox.run("y").value == "1"
    finally:
        sandbox.close()


def test_内核崩了_下次自动起一个新的(tmp_path):
    sandbox = Sandbox.local(PYTHON_KERNEL, tmp_path, result_resolver(ResultStore()))
    try:
        ex = sandbox.run("import os\nos._exit(3)")
        assert ex.restarted and "意外退出" in ex.error
        assert sandbox.run("1 + 1").value == "2"
    finally:
        sandbox.close()


def test_起不来就报SandboxUnavailable(tmp_path):
    missing = Sandbox(["finhelm-no-such-binary"], tmp_path, result_resolver(ResultStore()))
    with pytest.raises(SandboxUnavailable, match="找不到"):
        missing.run("1")
    dies = Sandbox([sys.executable, "-c", "import sys; sys.stderr.write('No such image: finhelm-sandbox'); sys.exit(1)"],
                   tmp_path, result_resolver(ResultStore()))
    with pytest.raises(SandboxUnavailable, match="No such image"):
        dies.run("1")


# ---------------------------------------------------------------- 结果怎么传
def test_encode_result_转成能过JSON的样子():
    payload = encode_result(MONTHLY)
    assert payload["dates"] == ["month"], "全是日期（或空）的列才转回日期"
    assert payload["rows"][0] == ["2024-01-01", 1001.25, None]
    assert payload["rows"][1][2] == "促销"


def test_resolver_不存在的编号():
    assert "还没有" in result_resolver(ResultStore())("r1")["error"]


# ---------------------------------------------------------------- 工具
class FakeSandbox:
    def __init__(self, execution: Execution) -> None:
        self.execution = execution
        self.code: list[str] = []

    def run(self, code: str) -> Execution:
        self.code.append(code)
        return self.execution


def run_tool(execution: Execution):
    return RunPythonTool(FakeSandbox(execution)).execute({"code": "..."})


def test_工具输出_打印_值_图():
    out = run_tool(Execution(output="共 12 个月\n", value="0.23", figures=[Path("work/figures/fig-1.png")]))
    assert not out.is_error
    assert out.content.startswith("共 12 个月\n\n0.23")
    assert "fig-1.png" in out.content and "用户能看到" in out.content
    assert out.summary == "输出 2 行，1 张图"


def test_工具输出_报错时带上已经打印的内容():
    out = run_tool(Execution(output="第一步好了\n", error="Traceback…\nZeroDivisionError: division by zero"))
    assert out.is_error and "第一步好了" in out.content and "ZeroDivisionError" in out.content
    assert out.summary == "报错：ZeroDivisionError: division by zero"


def test_工具输出_重启和NameError的提示():
    assert "变量都没了" in run_tool(Execution(error="超时", restarted=True)).content
    assert "可能重启过" in run_tool(Execution(error="NameError: name 'df' is not defined")).content


def test_什么都没输出时提醒怎么看结果():
    assert "打印出来" in run_tool(Execution()).content


def test_终端显示_输出长了会折叠_图的路径一定列出来():
    from data_agent.cli import _show_execution

    shown = _show_execution(Execution(output="x" * 5000, value="42", figures=[Path("work/figures/fig-1.png")]))
    assert "已折叠" in shown
    assert shown.splitlines()[-1].endswith("fig-1.png")


def test_run_python不参与清理():
    assert not RunPythonTool.rerunnable, "有状态：重跑会改变内核里的变量"


# ---------------------------------------------------------------- 组装
def test_开沙箱时注册工具_提示词里多一步_容器不急着启动():
    app = build_application(eval_settings("shop", python_sandbox=True), llm=ScriptedProvider())
    try:
        assert "run_python" in app.tools and "run_r" not in app.tools, "shop 场景包不要 R"
        assert "load_result" in app.agent.system_prompt
        assert not app.sandboxes["python"].running, "第一次 run_python 才启动"
    finally:
        app.close()


def test_关掉沙箱时只有SQL工具_提示词不讲沙箱():
    app = build_application(eval_settings("shop", python_sandbox=False), llm=ScriptedProvider())
    assert "run_python" not in app.tools and app.sandboxes == {}
    prompt = app.agent.system_prompt
    assert "4. 用自然语言给结论" in prompt
    assert "inputs/" not in prompt and "save_result" not in prompt


# ---------------------------------------------------------------- 真的容器
def _docker_image_ready(image: str = "finhelm-sandbox") -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


docker_only = pytest.mark.skipif(not _docker_image_ready(),
                                 reason="没有 Docker 或还没 docker build -t finhelm-sandbox docker/sandbox")


@docker_only
def test_容器里_断网_只读_能画图_关了就删(tmp_path):
    sandbox = Sandbox.docker("finhelm-sandbox", PYTHON_KERNEL, tmp_path, result_resolver(store_with_monthly()), timeout_s=20)
    try:
        net = sandbox.run("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)")
        assert net.error and "OSError" in net.error, net.error
        assert "Read-only file system" in sandbox.run("open('/etc/x', 'w')").error
        assert "root" not in sandbox.run("import getpass\ngetpass.getuser()").value

        fig = sandbox.run("df = load_result('r1')\nplt.plot(df.month, df.gmv)\nplt.title('月度 GMV')")
        assert fig.error is None and len(fig.figures) == 1 and fig.figures[0].exists()
        assert fig.figures[0].parent == tmp_path.resolve() / "figures"

        # 模型习惯性地把字体改成 DejaVu Sans，中文照样有字形（逐字回退到中文字体）
        cjk = sandbox.run('import matplotlib\nmatplotlib.rcParams["font.family"] = "DejaVu Sans"\n'
                          'plt.bar(["一月", "二月"], [1, 2])\nplt.gcf().canvas.draw()')
        assert "missing from font" not in cjk.output, cjk.output
    finally:
        name = sandbox.kill_command[-1]
        sandbox.close()
    listed = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"],
                            capture_output=True, text=True).stdout
    assert listed.strip() == "", "--rm：关掉之后容器也没了"


@docker_only
def test_容器里_内存超限被杀_自动重启(tmp_path):
    sandbox = Sandbox.docker("finhelm-sandbox", PYTHON_KERNEL, tmp_path, result_resolver(ResultStore()),
                             timeout_s=30, memory="256m")
    try:
        ex = sandbox.run("big = b'x' * 1024 ** 3")   # 真写满，calloc 的零页不算数
        assert ex.restarted
        assert sandbox.run("1 + 1").value == "2"
    finally:
        sandbox.close()


@docker_only
def test_自己savefig的图不再自动存一份_没存的照样自动存(tmp_path):
    sandbox = Sandbox.docker("finhelm-sandbox", PYTHON_KERNEL, tmp_path, result_resolver(store_with_monthly()), timeout_s=30)
    try:
        ex = sandbox.run("plt.figure(); plt.plot([1, 2])\nplt.savefig('figures/mine.png')\n"
                         "plt.figure(); plt.plot([3, 4])")
        assert ex.error is None, ex.error
        names = [f.name for f in ex.figures]
        assert len(names) == 2 and "mine.png" in names and any(n.startswith("fig-") for n in names), names
        # 存到 figures/ 以外的不算「自己存进交付目录」，照样自动存一份
        other = sandbox.run("plt.plot([1]); plt.savefig('/tmp/x.png')")
        assert [f.name for f in other.figures][0].startswith("fig-")
    finally:
        sandbox.close()
