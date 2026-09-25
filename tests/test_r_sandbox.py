"""run_r、R 内核和 meta 分析模板。R 只在容器里有：没有 Docker 或 finhelm-sandbox-r 镜像就跳过。

模板的数字和一份独立的实现逐项对：固定效应对 metafor，随机效应按 RevMan 5 的公式手算
（RevMan 的 τ² 用 Mantel-Haenszel 的 Q，metafor 的 DL 用倒方差的 Q，异质性不为 0 时两者不同）。
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from decimal import Decimal

import pandas as pd
import pytest

from data_agent.db.connection import QueryResult
from data_agent.tools.r import R_KERNEL, RunRTool
from data_agent.tools.sandbox import Execution, Sandbox
from data_agent.tools.sql.results import ResultStore, result_resolver


def _image_ready(image: str = "finhelm-sandbox-r") -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _image_ready(), reason="没有 Docker 或还没 docker build -t finhelm-sandbox-r docker/sandbox-r")


@pytest.fixture(scope="module")
def r(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    results = ResultStore()
    results.add("SELECT ...", QueryResult(["day", "amount", "tag"],
                                          [(dt.date(2024, 1, d), Decimal(f"{d}.5"), None if d % 2 else "x") for d in (1, 2, 3)],
                                          False, 1))
    sandbox = Sandbox.docker("finhelm-sandbox-r", R_KERNEL, work, result_resolver(results), timeout_s=60)
    yield sandbox
    sandbox.close()


def ok(r: Sandbox, code: str) -> Execution:
    ex = r.run(code)
    assert ex.error is None, f"{ex.error}\n{ex.output}"
    return ex


# ---------------------------------------------------------------- 内核
def test_变量保留_可见值打印_不可见的不打印(r):
    assert ok(r, "x <- 40\ninvisible(7)").output == ""
    assert ok(r, "x + 2").output.strip() == "[1] 42"


def test_报错带行号_不带内核自己的eval(r):
    ex = r.run("a <- 1\nb <- 2\nnot_defined + 1")
    assert "第 3 行" in ex.error and "not_defined" in ex.error and "eval(exprs" not in ex.error
    assert "代码解析出错" in r.run("f <- function( {").error


def test_退出不了_warning收进输出(r):
    assert "不能退出" in r.run("q()").error
    assert "警告：" in ok(r, 'as.numeric("abc")').output
    assert ok(r, "x").output.strip() == "[1] 40"


def test_load_result_类型和空值(r):
    ok(r, 'd <- load_result("r1")')
    out = ok(r, 'cat(class(d$day), class(d$amount), sum(d$amount), sum(is.na(d$tag)), "\\n")').output
    assert out.split() == ["Date", "numeric", "7.5", "2"]
    assert "没有编号为 r9" in r.run('load_result("r9")').error


def test_图自动保存_自己存的文件也报(r):
    auto = ok(r, 'plot(1:10, main = "中文")')
    assert len(auto.figures) == 1 and auto.figures[0].name.startswith("fig-") and auto.figures[0].exists()
    own = ok(r, 'cairo_pdf("figures/mine.pdf"); plot(1); invisible(dev.off())')
    assert [f.name for f in own.figures] == ["mine.pdf"]


def test_先打印再ggsave_同一张图只存一份(r):
    """模型常常 print(p) 看一眼再 ggsave(p)：自动设备上那一页不再另存 fig-N.png。"""
    ex = ok(r, 'p <- ggplot(data.frame(x = 1:3, y = 3:1), aes(x, y)) + geom_point()\n'
               'print(p)\nggsave("figures/p.png", p, width = 4, height = 3)')
    assert [f.name for f in ex.figures] == ["p.png"]
    assert not list((r.work_dir / "figures").glob(".r-*.png")), "临时页删掉了"
    # 只存了表格（不是图）时，自动出的图照样保存
    table = ok(r, 'plot(1:3)\nwritexl::write_xlsx(data.frame(a = 1), "figures/t.xlsx")')
    assert sorted(f.suffix for f in table.figures) == [".png", ".xlsx"]


def test_工具输出里的重启提示认得R的报错():
    tool = RunRTool(sandbox=None)  # type: ignore[arg-type]
    assert "可能重启过" in tool.render(Execution(error="第 1 行出错：object 'm' not found"))


# ---------------------------------------------------------------- 数字：和独立实现对
STUDIES = """
d <- data.frame(study = c("S1", "S2", "S3", "S4", "S5", "S6"),
                ee = c(12, 20, 8, 30, 15, 40), ne = c(100, 150, 80, 210, 120, 160),
                ec = c(20, 28, 15, 41, 18, 20), nc = c(98, 148, 82, 205, 118, 158))
"""


def test_二分类_固定效应MH_和metafor一样(r):
    ok(r, STUDIES)
    ok(r, """
m <- fh_meta_bin(d, "study", "ee", "ne", "ec", "nc", sm = "RR", model = "fixed")
ref <- metafor::rma.mh(ai = ee, n1i = ne, ci = ec, n2i = nc, data = d, measure = "RR")
stopifnot(abs(m$TE.common - ref$beta) < 1e-10, abs(m$lower.common - ref$ci.lb) < 1e-10)
m2 <- fh_meta_bin(d, "study", "ee", "ne", "ec", "nc", sm = "OR", model = "fixed")
ref2 <- metafor::rma.mh(ai = ee, n1i = ne, ci = ec, n2i = nc, data = d, measure = "OR")
stopifnot(abs(m2$TE.common - ref2$beta) < 1e-10)
""")


def test_二分类_随机效应按RevMan5的公式(r):
    """RevMan 5：Q 用 M-H 合并值算，τ² = DerSimonian-Laird，随机效应用倒方差权重 1/(v+τ²)。"""
    ok(r, STUDIES)
    ex = ok(r, """
m <- fh_meta_bin(d, "study", "ee", "ne", "ec", "nc")
y <- log((d$ee / d$ne) / (d$ec / d$nc)); v <- 1/d$ee - 1/d$ne + 1/d$ec - 1/d$nc; w <- 1 / v
mh <- metafor::rma.mh(ai = ee, n1i = ne, ci = ec, n2i = nc, data = d, measure = "RR")$beta[1]
Q <- sum(w * (y - mh)^2); k <- nrow(d)
tau2 <- max(0, (Q - (k - 1)) / (sum(w) - sum(w^2) / sum(w)))
wr <- 1 / (v + tau2); est <- sum(wr * y) / sum(wr); se <- sqrt(1 / sum(wr))
stopifnot(tau2 > 0)                                   # 这组数据有异质性，公式的差别才测得出来
stopifnot(abs(m$Q - Q) < 1e-8, abs(m$tau2 - tau2) < 1e-10, abs(m$TE.random - est) < 1e-10,
          abs(m$lower.random - (est - qnorm(0.975) * se)) < 1e-10, abs(m$I2 - (Q - (k - 1)) / Q) < 1e-10)
""")
    first = ex.output.splitlines()[0]
    assert first.startswith("随机效应模型（Mantel-Haenszel 法，τ² 用 DerSimonian-Laird 法估计）：RR ")
    assert "异质性：Tau² = " in ex.output and "I² = " in ex.output


def test_单个零事件照常合并_双零研究不参与(r):
    ex = ok(r, """
z <- data.frame(study = c("A", "B", "C", "D"), ee = c(0, 3, 0, 5), ne = c(50, 60, 40, 80),
                ec = c(4, 6, 0, 9), nc = c(50, 58, 41, 79))
mz <- fh_meta_bin(z, "study", "ee", "ne", "ec", "nc")
stopifnot(mz$k == 3)
""")
    assert "不参与合并" in ex.output and "C" in ex.output.split("不参与合并")[1]


def test_连续变量_MD和SMD(r):
    ok(r, """
c <- data.frame(study = paste0("C", 1:5), me = c(5.1, 4.8, 6.0, 5.5, 4.9), se_ = c(1.2, 1.5, 1.1, 1.3, 1.6),
                ne = c(40, 55, 30, 62, 48), mc = c(6.0, 5.9, 6.4, 6.8, 5.2), sc = c(1.3, 1.4, 1.2, 1.5, 1.5),
                nc = c(42, 50, 33, 60, 45))
md <- fh_meta_cont(c, "study", "me", "se_", "ne", "mc", "sc", "nc", sm = "MD", model = "fixed")
ref <- metafor::rma(measure = "MD", m1i = me, sd1i = se_, n1i = ne, m2i = mc, sd2i = sc, n2i = nc, data = c, method = "FE")
stopifnot(abs(md$TE.common - ref$beta) < 1e-10)
# RevMan 5 的 SMD：Hedges' g，校正系数 1 - 3/(4N - 9)
smd <- fh_meta_cont(c, "study", "me", "se_", "ne", "mc", "sc", "nc", sm = "SMD", model = "fixed")
N <- c$ne + c$nc
sp <- sqrt(((c$ne - 1) * c$se_^2 + (c$nc - 1) * c$sc^2) / (N - 2))
g <- (c$me - c$mc) / sp * (1 - 3 / (4 * N - 9))
vg <- N / (c$ne * c$nc) + g^2 / (2 * (N - 3.94))
stopifnot(max(abs(smd$TE - g)) < 1e-10, abs(smd$TE.common - sum(g / vg) / sum(1 / vg)) < 1e-10)
""")


def test_文献给的HR和置信区间_在对数尺度上合并(r):
    ok(r, """
h <- data.frame(study = c("H1", "H2", "H3"), hr = c(0.8, 0.7, 0.9), lo = c(0.6, 0.5, 0.7), hi = c(1.07, 0.98, 1.16))
mh <- fh_meta_gen(h, "study", "hr", "lo", "hi", sm = "HR")
se <- (log(h$hi) - log(h$lo)) / (2 * qnorm(0.975))
ref <- metafor::rma(yi = log(h$hr), sei = se, method = "DL")
stopifnot(abs(mh$TE.random - ref$beta) < 1e-10)
""")


def test_数据有问题时说清楚(r):
    assert "没有列「总数」" in r.run('fh_meta_bin(d, "study", "ee", "总数", "ec", "nc")').error
    assert "事件数不在 0 到总人数之间" in r.run(
        'bad <- d; bad$ee[2] <- 999; fh_meta_bin(bad, "study", "ee", "ne", "ec", "nc")').error
    assert "不在置信区间里" in r.run(
        'fh_meta_gen(data.frame(s = "x", e = 2, l = 0.5, u = 1.5), "s", "e", "l", "u")').error


# ---------------------------------------------------------------- 偏倚风险、图
def test_RoB_判定的各种写法都认_认不出就报错(r):
    ok(r, STUDIES)
    ex = ok(r, """
rb <- data.frame(study = d$study, D1 = c("低", "Low", "+", "low risk", "不清楚", "高"), D2 = "?", D3 = "High",
                 D4 = "Low", D5 = "Low", D6 = "Low", D7 = c("Unclear risk of bias", "-", "L", "H", "U", "unclear"))
r1 <- fh_rob(rb, tool = "rob1")
stopifnot(identical(r1$D1, c(rep("Low risk of bias", 4), "Unclear risk of bias", "High risk of bias")),
          attr(r1, "labels")[1] == "Random sequence generation (selection bias)")
""")
    assert "RoB1，6 项研究" in ex.output
    # 每张图是什么写进输出：模型转述时不会把汇总图和比例图说反
    assert "rob_summary.png" in ex.output.split("Risk of bias summary")[1].split("\n")[0]
    assert "rob_graph.png" in ex.output.split("Risk of bias graph")[1].split("\n")[0]
    assert {f.name for f in ex.figures} >= {"rob_summary.png", "rob_summary.pdf", "rob_graph.png", "rob_graph.pdf"}
    err = r.run('rb2 <- rb; rb2$D2[1] <- "maybe"; fh_rob(rb2, tool = "RoB1")').error
    assert "认不出" in err and "maybe" in err


def test_RoB2_中文领域名原样用_带overall(r):
    ok(r, """
r2 <- fh_rob(data.frame(研究 = d$study, 随机化 = "Low", 偏离干预 = "Some concerns", 缺失 = "Low",
                        测量 = "Low", 选择性报告 = "High", 总体 = "High", check.names = FALSE),
             tool = "RoB2", study = "研究", overall = "总体")
stopifnot(identical(attr(r2, "labels"), c("随机化", "偏离干预", "缺失", "测量", "选择性报告")),
          all(r2$overall == "High risk of bias"))
""")


def test_森林图_带RoB列_研究名对不上就报错(r):
    ok(r, STUDIES)
    ex = ok(r, """
m <- fh_meta_bin(d, "study", "ee", "ne", "ec", "nc")
rb <- fh_rob(data.frame(study = d$study, A = "Low", B = "Low", C = "Unclear", D = "Low", E = "High", F = "Low", G = "Low"),
             tool = "RoB1", file = "rob_for_forest")
fh_forest(m, rob = rb, file = "forest_rob", formats = c("png", "pdf", "tiff"))
""")
    names = {f.name for f in ex.figures}
    assert {"forest_rob.png", "forest_rob.pdf", "forest_rob.tiff"} <= names
    assert "找不到这些研究：S6" in r.run('rb_bad <- rb[1:5, ]; attributes(rb_bad)[c("tool", "domains", "labels")] <- '
                                          'attributes(rb)[c("tool", "domains", "labels")]; fh_forest(m, rob = rb_bad)').error


def test_漏斗图_敏感性分析_导出(r):
    ok(r, STUDIES)
    ex = ok(r, 'm <- fh_meta_bin(d, "study", "ee", "ne", "ec", "nc")\nfh_funnel(m)\nfh_sensitivity(m)\nfh_export(m)')
    assert "少于 10 项" in ex.output and "去掉的研究" in ex.output
    names = {f.name for f in ex.figures}
    assert {"funnel.png", "leave_one_out.png", "meta_results.xlsx"} <= names
    sheet = pd.read_excel(next(f for f in ex.figures if f.name == "meta_results.xlsx"), sheet_name="各研究")
    assert list(sheet["研究"]) == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert sheet["权重 %"].sum() == pytest.approx(100, abs=0.2)


def test_fh_help列出全部模板(r):
    out = ok(r, "fh_help()").output
    for name in ("fh_meta_bin", "fh_meta_cont", "fh_meta_gen", "fh_forest", "fh_rob", "fh_funnel",
                 "fh_sensitivity", "fh_report", "fh_methods", "fh_export"):
        assert name in out
