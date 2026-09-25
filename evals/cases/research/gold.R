# research 题库的标准答案。gen_research.py 把 source/ 和这个脚本放进 R 沙箱跑，打印一行 JSON。
#
# 直接用 meta 包 + RevMan 5 的设置算，不经过 fh_ 模板：模板是被考的对象，不能自己给自己判分。
# （模板本身在 tests/test_r_sandbox.py 里对过独立实现。）
# 数据都是 metafor / meta 包收录的已发表 meta 分析，出处见 gen_research.py。

suppressPackageStartupMessages(library(meta))
settings.meta("RevMan5")
rd <- function(name) read.csv(file.path("source", paste0(name, ".csv")), stringsAsFactors = FALSE)

# 合并效应量、95% CI（比值类回到原尺度）、I²
pooled <- function(m, random = TRUE) {
  te <- if (random) c(m$TE.random, m$lower.random, m$upper.random) else c(m$TE.common, m$lower.common, m$upper.common)
  c(if (m$sm %in% c("RR", "OR")) exp(te) else te, m$I2)
}

bcg <- rd("bcg")
bin_bcg <- function(sm = "RR", random = TRUE, ...) {
  metabin(tpos, tpos + tneg, cpos, cpos + cneg, data = bcg, studlab = paste(author, year),
          sm = sm, method = "MH", common = !random, random = random, ...)
}

gold <- list()
gold[["rs-01"]] <- pooled(bin_bcg())
gold[["rs-02"]] <- pooled(bin_bcg("OR", random = FALSE), random = FALSE)

sub <- bin_bcg(subgroup = bcg$alloc)
gold[["rs-03"]] <- c(rbind(exp(sub$TE.random.w), exp(sub$lower.random.w), exp(sub$upper.random.w)))
names(gold[["rs-03"]]) <- NULL
gold[["rs-03-labels"]] <- sub$subgroup.levels
gold[["rs-03-p"]] <- sub$pval.Q.b.random

nw <- rd("nielweise2007")
m <- metabin(ai, n1i, ci, n2i, data = nw, studlab = paste(author, year), sm = "RR", method = "MH")
gold[["rs-04"]] <- pooled(m)
gold[["rs-04-k"]] <- m$k                      # 两组都是零事件的 Yucel 2004 不参与合并

nm <- rd("normand1999")
gold[["rs-05"]] <- pooled(metacont(n1i, m1i, sd1i, n2i, m2i, sd2i, data = nm, studlab = source, sm = "MD"))

fl <- rd("fleiss1993cont")
gold[["rs-06"]] <- pooled(metacont(n.psyc, mean.psyc, sd.psyc, n.cont, mean.cont, sd.cont, data = fl,
                                   studlab = paste(study, year), sm = "SMD"))

am <- rd("amlodipine")
gold[["rs-07"]] <- pooled(metacont(n.amlo, mean.amlo, sqrt(var.amlo), n.plac, mean.plac, sqrt(var.plac),
                                   data = am, studlab = study, sm = "MD"))

hk <- rd("hackshaw1998")
gold[["rs-08"]] <- pooled(metagen(log(or), lower = log(or.lb), upper = log(or.ub), data = hk,
                                  studlab = paste(author, year), sm = "OR"))

eg <- rd("egger2001")
mg <- function(d) metabin(ai, n1i, ci, n2i, data = d, studlab = paste(study, year), sm = "RR", method = "MH")
gold[["rs-09"]] <- c(pooled(mg(eg))[1:3], pooled(mg(eg[eg$study != "ISIS-4", ]))[1:3])

gold[["rs-10"]] <- gold[["rs-01"]]

num <- function(x) if (is.numeric(x)) sprintf("[%s]", paste(sprintf("%.6f", x), collapse = ", ")) else
  sprintf("[%s]", paste(sprintf("\"%s\"", x), collapse = ", "))
cat("GOLD_JSON: {", paste(sprintf("\"%s\": %s", names(gold), vapply(gold, num, "")), collapse = ", "), "}\n", sep = "")
