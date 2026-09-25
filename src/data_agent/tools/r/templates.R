# FinHelm 的 meta 分析模板：算法和版式对齐 RevMan 5。内核启动时 source 进来。
#
# 模型调这些函数，不自己写 meta 分析公式和画图代码：数字全部来自 meta 包（settings.meta("RevMan5")），
# 模板只负责取列、查数据、画图存图、把结果写成 RevMan 的格式。fh_help() 列出全部。

suppressPackageStartupMessages({
  library(meta); library(metafor); library(robvis)
  library(readxl); library(writexl); library(ggplot2)
})
invisible(settings.meta("RevMan5"))

RATIO_SM <- c("RR", "OR", "HR", "IRR")

# ================================================================ 取列、查数据
.col <- function(data, name, what) {
  if (is.null(name)) stop(sprintf("没有指定%s的列名。", what), call. = FALSE)
  if (!name %in% names(data)) {
    stop(sprintf("数据里没有列「%s」（%s）。现有的列：%s", name, what, paste(names(data), collapse = "、")), call. = FALSE)
  }
  data[[name]]
}

.num <- function(data, name, what) {
  x <- .col(data, name, what)
  v <- suppressWarnings(as.numeric(x))
  bad <- which(is.na(v) & !is.na(x))
  if (length(bad)) {
    stop(sprintf("列「%s」（%s）第 %s 行不是数字：%s", name, what, paste(bad, collapse = "、"),
                 paste(x[bad], collapse = "、")), call. = FALSE)
  }
  v
}

.model <- function(model) {
  model <- match.arg(model, c("random", "fixed"))
  list(common = model == "fixed", random = model == "random")
}

.subgroup <- function(data, subgroup) if (is.null(subgroup)) NULL else as.character(.col(data, subgroup, "亚组"))

# ================================================================ 合并效应量
fh_meta_bin <- function(data, study, event_e, n_e, event_c, n_c, sm = "RR", method = "MH",
                        model = "random", label_e = "试验组", label_c = "对照组",
                        subgroup = NULL, outcome = "") {
  sm <- match.arg(sm, c("RR", "OR", "RD"))
  method <- match.arg(method, c("MH", "Inverse", "Peto"))
  if (method == "Peto" && sm != "OR") stop("Peto 法只能算 OR。", call. = FALSE)
  ee <- .num(data, event_e, "试验组事件数"); ne <- .num(data, n_e, "试验组总人数")
  ec <- .num(data, event_c, "对照组事件数"); nc <- .num(data, n_c, "对照组总人数")
  bad <- which(ee > ne | ec > nc | ee < 0 | ec < 0)
  if (length(bad)) stop(sprintf("第 %s 行的事件数不在 0 到总人数之间，先核对数据。", paste(bad, collapse = "、")), call. = FALSE)
  mdl <- .model(model)
  m <- metabin(ee, ne, ec, nc, studlab = as.character(.col(data, study, "研究")), sm = sm, method = method,
               common = mdl$common, random = mdl$random, label.e = label_e, label.c = label_c,
               subgroup = .subgroup(data, subgroup), title = outcome)
  fh_report(m)
  invisible(m)
}

fh_meta_cont <- function(data, study, mean_e, sd_e, n_e, mean_c, sd_c, n_c, sm = "MD",
                         model = "random", label_e = "试验组", label_c = "对照组",
                         subgroup = NULL, outcome = "") {
  sm <- match.arg(sm, c("MD", "SMD"))
  mdl <- .model(model)
  m <- metacont(.num(data, n_e, "试验组人数"), .num(data, mean_e, "试验组均数"), .num(data, sd_e, "试验组标准差"),
                .num(data, n_c, "对照组人数"), .num(data, mean_c, "对照组均数"), .num(data, sd_c, "对照组标准差"),
                studlab = as.character(.col(data, study, "研究")), sm = sm,
                common = mdl$common, random = mdl$random, label.e = label_e, label.c = label_c,
                subgroup = .subgroup(data, subgroup), title = outcome)
  fh_report(m)
  invisible(m)
}

# 文献里直接报的效应量和置信区间（HR、校正后的 OR…）：通用倒方差法。比值类在对数尺度上合并
fh_meta_gen <- function(data, study, estimate, lower, upper, sm = "HR", model = "random",
                        label_e = "试验组", label_c = "对照组", subgroup = NULL, outcome = "") {
  est <- .num(data, estimate, "效应量"); lo <- .num(data, lower, "置信区间下限"); hi <- .num(data, upper, "置信区间上限")
  bad <- which(!(lo <= est & est <= hi))
  if (length(bad)) stop(sprintf("第 %s 行的效应量不在置信区间里，先核对数据。", paste(bad, collapse = "、")), call. = FALSE)
  ratio <- sm %in% RATIO_SM
  if (ratio && any(lo <= 0)) stop(sprintf("%s 是比值，置信区间下限必须大于 0。", sm), call. = FALSE)
  z <- qnorm(0.975)
  te <- if (ratio) log(est) else est
  se <- if (ratio) (log(hi) - log(lo)) / (2 * z) else (hi - lo) / (2 * z)
  mdl <- .model(model)
  m <- metagen(te, se, studlab = as.character(.col(data, study, "研究")), sm = sm,
               common = mdl$common, random = mdl$random, label.e = label_e, label.c = label_c,
               subgroup = .subgroup(data, subgroup), title = outcome)
  fh_report(m)
  invisible(m)
}

# ================================================================ 结果文字
.bt <- function(m, x) if (m$sm %in% RATIO_SM) exp(x) else x
.f2 <- function(x) formatC(x, format = "f", digits = 2)
.p <- function(p) if (is.na(p)) "NA" else if (p < 0.001) "P < 0.001" else paste0("P = ", formatC(p, format = "f", digits = 3))
.random <- function(m) isTRUE(m$random)

.method_text <- function(m) {
  pooling <- switch(m$method, MH = "Mantel-Haenszel 法", Inverse = "倒方差法", Peto = "Peto 法", "倒方差法")
  if (.random(m)) paste0("随机效应模型（", pooling, "，τ² 用 DerSimonian-Laird 法估计）")
  else paste0("固定效应模型（", pooling, "）")
}

fh_report <- function(m, quiet = FALSE) {
  r <- .random(m)
  te <- if (r) m$TE.random else m$TE.common
  lo <- if (r) m$lower.random else m$lower.common
  hi <- if (r) m$upper.random else m$upper.common
  z <- if (r) m$statistic.random else m$statistic.common
  p <- if (r) m$pval.random else m$pval.common
  lines <- c(
    sprintf("%s：%s %s（95%% CI %s–%s），Z = %s，%s", .method_text(m), m$sm, .f2(.bt(m, te)),
            .f2(.bt(m, lo)), .f2(.bt(m, hi)), .f2(abs(z)), .p(p)),
    sprintf("纳入 %d 项研究（参与合并 %d 项）%s", length(m$studlab), m$k,
            if (!is.null(m$n.e)) sprintf("，%s %d 人、%s %d 人", m$label.e, sum(m$n.e, na.rm = TRUE),
                                         m$label.c, sum(m$n.c, na.rm = TRUE)) else ""),
    sprintf("异质性：Tau² = %s；Chi² = %s，df = %d（%s）；I² = %d%%", formatC(m$tau2, format = "f", digits = 3),
            .f2(m$Q), as.integer(m$df.Q), .p(m$pval.Q), as.integer(round(100 * m$I2)))
  )
  if (!is.null(m$subgroup)) {
    lines <- c(lines, "亚组：")
    for (i in seq_along(m$subgroup.levels)) {
      lines <- c(lines, sprintf("  %s：%s %s（95%% CI %s–%s），%d 项研究，I² = %d%%", m$subgroup.levels[i], m$sm,
                                .f2(.bt(m, if (r) m$TE.random.w[i] else m$TE.common.w[i])),
                                .f2(.bt(m, if (r) m$lower.random.w[i] else m$lower.common.w[i])),
                                .f2(.bt(m, if (r) m$upper.random.w[i] else m$upper.common.w[i])),
                                as.integer(m$k.w[i]), as.integer(round(100 * m$I2.w[i]))))
    }
    qb <- if (r) m$Q.b.random else m$Q.b.common
    pb <- if (r) m$pval.Q.b.random else m$pval.Q.b.common
    lines <- c(lines, sprintf("亚组间差异检验：Chi² = %s，df = %d（%s）", .f2(qb), as.integer(m$df.Q.b), .p(pb)))
  }
  w <- if (r) m$w.random else m$w.common
  excluded <- m$studlab[is.na(w) | w == 0]
  if (length(excluded)) lines <- c(lines, sprintf("不参与合并（两组都是零事件等，RevMan 标为 Not estimable）：%s",
                                                  paste(excluded, collapse = "、")))
  if (!quiet) cat(paste(lines, collapse = "\n"), "\n")
  invisible(list(sm = m$sm, estimate = .bt(m, te), lower = .bt(m, lo), upper = .bt(m, hi), z = abs(z), p = p,
                 tau2 = m$tau2, Q = m$Q, df = m$df.Q, p_Q = m$pval.Q, I2 = m$I2, k = m$k))
}

fh_methods <- function(m) {
  measure <- switch(m$sm, RR = "相对危险度（RR）", OR = "比值比（OR）", RD = "危险差（RD）", MD = "均数差（MD）",
                    SMD = "标准化均数差（SMD，Hedges' g）", HR = "风险比（HR）", m$sm)
  text <- sprintf(paste0(
    "采用 R %s.%s 的 meta 包（%s 版）进行 meta 分析，参数设置与 RevMan 5 一致。效应量为%s及其 95%% 置信区间，",
    "采用%s合并。异质性用 Cochran Q 检验（Chi²）和 I² 统计量评价。"),
    R.version$major, R.version$minor, as.character(packageVersion("meta")), measure, .method_text(m))
  cat(text, "\n")
  invisible(text)
}

# ================================================================ 存图
# what：这张图是什么。写进输出，模型转述时才不会把两张图说反
.save <- function(draw, file, width, height, formats, what) {
  paths <- character(0)
  for (fmt in formats) {
    path <- file.path("figures", paste0(file, ".", fmt))
    switch(fmt,
      png = png(path, width = width, height = height, units = "in", res = 300, type = "cairo"),
      pdf = cairo_pdf(path, width = width, height = height),          # 普通 pdf() 画不出中文
      tiff = tiff(path, width = width, height = height, units = "in", res = 300, type = "cairo", compression = "lzw"),
      stop("格式只能是 png / pdf / tiff。", call. = FALSE))
    tryCatch(draw(), finally = dev.off())
    paths <- c(paths, path)
  }
  cat(sprintf("已保存%s：%s\n", what, paste(paths, collapse = "、")))
  invisible(paths)
}

# meta 的森林图只算得出高度：先画在很宽的画布上，量出内容实际多宽，再按这个宽度出图
.forest_size <- function(draw) {
  pdf(NULL)
  dims <- tryCatch(draw(), finally = dev.off())
  height <- dims$figheight$total_height + 0.3
  tmp <- tempfile(fileext = ".png")
  png(tmp, width = 30, height = height, units = "in", res = 100, type = "cairo")
  tryCatch(draw(), finally = dev.off())
  img <- png::readPNG(tmp)
  ink <- which(colSums(img[, , 1] < 0.95 | img[, , 2] < 0.95 | img[, , 3] < 0.95) > 0)
  c(width = (max(ink) - min(ink)) / 100 + 0.4, height = height)
}

fh_forest <- function(m, file = "forest", rob = NULL, label_left = NULL, label_right = NULL,
                      formats = c("png", "pdf"), ...) {
  args <- list(m, label.left = label_left %||% paste0("利于", m$label.e),
               label.right = label_right %||% paste0("利于", m$label.c),
               print.subgroup.name = FALSE,                       # RevMan 的亚组标题只写组名
               text.common.w = "Subtotal (95% CI)", text.random.w = "Subtotal (95% CI)")
  if (!is.null(rob)) {
    args[[1]] <- .attach_rob(m, rob)
    args$rob.only <- TRUE        # meta 默认在 RoB 列前面把效应量再列一遍；RevMan 右边只有 RoB
  }
  args <- c(args, list(...))
  draw <- function() do.call(forest, args)
  size <- .forest_size(draw)
  .save(draw, file, size[["width"]], size[["height"]], formats,
        paste0("森林图（RevMan 5 版式", if (!is.null(m$subgroup)) "，含亚组" else "",
               if (!is.null(rob)) "，右侧附各领域偏倚风险" else "", "）"))
}

# ================================================================ 偏倚风险
.ROB_CATEGORIES <- list(
  RoB1 = c("Low risk of bias", "Unclear risk of bias", "High risk of bias"),
  RoB2 = c("Low risk of bias", "Some concerns", "High risk of bias")
)
.ROB_DOMAINS <- list(
  RoB1 = c("Random sequence generation (selection bias)", "Allocation concealment (selection bias)",
           "Blinding of participants and personnel (performance bias)",
           "Blinding of outcome assessment (detection bias)", "Incomplete outcome data (attrition bias)",
           "Selective reporting (reporting bias)", "Other bias"),
  RoB2 = c("Bias arising from the randomization process", "Bias due to deviations from intended interventions",
           "Bias due to missing outcome data", "Bias in measurement of the outcome",
           "Bias in selection of the reported result")
)
.ROB_COLOURS <- c("#02C100", "#E2DF07", "#BF0000")          # Cochrane 的绿、黄、红
.ROB_SYMBOLS <- c("+", "?", "−")

.rob_tool <- function(tool) {
  key <- toupper(gsub("[^0-9A-Za-z]", "", tool))
  if (key %in% c("ROB1", "ROB")) return("RoB1")
  if (key == "ROB2") return("RoB2")
  stop("tool 只能是 RoB1 或 RoB2。", call. = FALSE)
}

.rob_values <- function(x, tool, column) {
  cats <- .ROB_CATEGORIES[[tool]]
  key <- tolower(trimws(as.character(x)))
  low <- c("low", "low risk", "low risk of bias", "+", "l", "低", "低风险", "低偏倚风险")
  high <- c("high", "high risk", "high risk of bias", "-", "−", "h", "高", "高风险", "高偏倚风险")
  mid <- c("unclear", "unclear risk", "unclear risk of bias", "some concerns", "some concern", "some", "?", "u",
           "不清楚", "不确定", "未知", "风险不清楚",
           "一些担忧", "有一定风险", "部分担忧")
  out <- ifelse(key %in% low, cats[1], ifelse(key %in% mid, cats[2], ifelse(key %in% high, cats[3], NA)))
  bad <- unique(as.character(x)[is.na(out)])
  if (length(bad)) {
    stop(sprintf("列「%s」里有认不出的判定：%s。能认的写法：Low / %s / High，或者 低 / 不清楚 / 高，或者 + / ? / -。",
                 column, paste(ifelse(is.na(bad), "（空）", bad), collapse = "、"), cats[2]), call. = FALSE)
  }
  other <- if (tool == "RoB1") "concern" else "unclear"
  if (any(grepl(other, key))) {
    warning(sprintf("列「%s」里有 %s 的写法，这是 %s 的类别，确认一下工具选对了没有。", column, other,
                    if (tool == "RoB1") "RoB 2" else "RoB 1"), call. = FALSE)
  }
  out
}

fh_rob <- function(data, tool, study = "study", domains = NULL, overall = NULL, labels = NULL,
                   file = "rob", formats = c("png", "pdf")) {
  tool <- .rob_tool(tool)
  studies <- as.character(.col(data, study, "研究"))
  domains <- domains %||% setdiff(names(data), c(study, overall))
  for (d in c(domains, overall)) .col(data, d, "偏倚风险领域")
  # 列名是 D1、A、领域1 这种代号，而且个数对得上标准领域：换成标准领域名
  coded <- all(grepl("^(d|domain|领域)?\\s*[0-9a-g]$", tolower(domains)))
  labels <- labels %||% (if (coded && length(domains) == length(.ROB_DOMAINS[[tool]])) .ROB_DOMAINS[[tool]] else domains)
  if (length(labels) != length(domains)) stop("labels 的个数要和领域列一样多。", call. = FALSE)
  out <- data.frame(study = studies, check.names = FALSE, stringsAsFactors = FALSE)
  for (d in domains) out[[d]] <- .rob_values(data[[d]], tool, d)
  if (!is.null(overall)) out$overall <- .rob_values(data[[overall]], tool, overall)
  attr(out, "tool") <- tool
  attr(out, "domains") <- domains
  attr(out, "labels") <- labels

  cols <- c(domains, if (!is.null(overall)) "overall")
  labs <- c(labels, if (!is.null(overall)) "Overall")
  cats <- .ROB_CATEGORIES[[tool]]
  counts <- t(sapply(cols, function(cl) table(factor(out[[cl]], levels = cats))))
  dimnames(counts) <- list(labs, c("Low", if (tool == "RoB1") "Unclear" else "Some concerns", "High"))
  cat(sprintf("%s，%d 项研究，各领域的判定数：\n", tool, nrow(out)))
  print(counts)

  n <- nrow(out)
  label_width <- max(nchar(labs, type = "width"))
  .save(function() print(.rob_summary_plot(out, cols, labs, cats)), paste0(file, "_summary"),
        width = 1.2 + max(nchar(studies, type = "width")) * 0.09 + 0.5 * length(cols),
        height = 1 + 0.4 * n + label_width * 0.075, formats,
        "偏倚风险汇总图（每项研究 × 每个领域一个 + ? - 圆点，RevMan 的 Risk of bias summary）")
  .save(function() print(.rob_graph_plot(out, cols, labs, cats)), paste0(file, "_graph"),
        width = 4.5 + label_width * 0.075, height = 1.2 + 0.4 * length(cols), formats,
        "偏倚风险比例图（每个领域各类判定占多少比例的条图，RevMan 的 Risk of bias graph）")
  invisible(out)
}

# RevMan 5 的 Risk of bias summary：每项研究 × 每个领域一个带符号的圆
.rob_summary_plot <- function(r, cols, labs, cats) {
  long <- data.frame(study = rep(r$study, times = length(cols)), domain = rep(labs, each = nrow(r)),
                     value = unlist(r[cols], use.names = FALSE))
  long$study <- factor(long$study, levels = rev(r$study))
  long$domain <- factor(long$domain, levels = labs)
  long$value <- factor(long$value, levels = cats)
  ggplot(long, aes(domain, study)) +
    geom_point(aes(fill = value), shape = 21, size = 8, colour = "grey25") +
    geom_text(aes(label = .ROB_SYMBOLS[as.integer(value)]), size = 5.5, fontface = "bold") +
    scale_fill_manual(values = setNames(.ROB_COLOURS, cats), drop = FALSE, name = NULL) +
    scale_x_discrete(position = "top") +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x.top = element_text(angle = 90, hjust = 0, vjust = 0.5),
          panel.grid = element_blank(), legend.position = "bottom")
}

# RevMan 5 的 Risk of bias graph：每个领域各类判定占多少比例（不按权重）
.rob_graph_plot <- function(r, cols, labs, cats) {
  long <- do.call(rbind, lapply(seq_along(cols), function(i) {
    share <- as.numeric(table(factor(r[[cols[i]]], levels = cats))) / nrow(r)
    data.frame(domain = labs[i], category = cats, share = share)
  }))
  long$domain <- factor(long$domain, levels = rev(labs))
  long$category <- factor(long$category, levels = rev(cats))
  ggplot(long, aes(share, domain, fill = category)) +
    geom_col(width = 0.7, colour = "grey25", linewidth = 0.3) +
    scale_x_continuous(labels = function(x) paste0(x * 100, "%"), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
    scale_fill_manual(values = setNames(.ROB_COLOURS, cats), breaks = cats, name = NULL) +
    labs(x = NULL, y = NULL) +
    theme_classic(base_size = 11) +
    theme(legend.position = "bottom", plot.margin = margin(6, 18, 6, 6))   # 右边留出「100%」
}

# 森林图右边加上每个领域的 + ? -（RevMan 5 的版式）
.attach_rob <- function(m, rob) {
  if (is.null(attr(rob, "tool"))) stop("rob 要传 fh_rob() 的返回值。", call. = FALSE)
  idx <- match(m$studlab, rob$study)
  if (anyNA(idx)) {
    stop(sprintf("偏倚风险表里找不到这些研究：%s（研究名要和 meta 分析里的一字不差）",
                 paste(m$studlab[is.na(idx)], collapse = "、")), call. = FALSE)
  }
  domains <- attr(rob, "domains")
  items <- setNames(lapply(domains, function(d) rob[[d]][idx]), paste0("item", seq_along(domains)))
  args <- c(items, list(data = m, tool = attr(rob, "tool"), domains = attr(rob, "labels")))
  if (!is.null(rob$overall)) args$overall <- rob$overall[idx]
  do.call(meta::rob, args)
}

# ================================================================ 发表偏倚、敏感性分析、导出
fh_funnel <- function(m, file = "funnel", formats = c("png", "pdf")) {
  .save(function() funnel(m), file, 6, 5, formats, "漏斗图")
  if (m$k >= 10) {
    b <- metabias(m, method.bias = "Egger", k.min = 10)
    cat(sprintf("Egger 检验：t = %s，df = %d，%s\n", .f2(b$statistic), as.integer(b$df), .p(b$pval)))
    invisible(b)
  } else {
    cat(sprintf("只有 %d 项研究：少于 10 项时 Egger 检验的检验效能太低（Cochrane 手册的建议），只画漏斗图，不做检验。\n", m$k))
    invisible(NULL)
  }
}

fh_sensitivity <- function(m, file = "leave_one_out", formats = c("png", "pdf")) {
  mi <- metainf(m, pooled = if (.random(m)) "random" else "common")
  n <- length(m$studlab)
  p <- mi$pval[1:n]
  res <- data.frame(`去掉的研究` = m$studlab, check.names = FALSE,
                    effect = round(.bt(m, mi$TE[1:n]), 2), lower = round(.bt(m, mi$lower[1:n]), 2),
                    upper = round(.bt(m, mi$upper[1:n]), 2), I2 = paste0(round(100 * mi$I2[1:n]), "%"),
                    P = vapply(p, .p, character(1)))
  names(res)[2:5] <- c(m$sm, "下限", "上限", "I²")
  pooled_p <- if (.random(m)) m$pval.random else m$pval.common
  flips <- m$studlab[(p < 0.05) != (pooled_p < 0.05)]
  print(res, row.names = FALSE)
  cat(if (length(flips)) sprintf("去掉 %s 后，合并结果的统计学显著性发生改变。\n", paste(flips, collapse = "、"))
      else "逐一去掉每项研究，合并结果的统计学显著性都不变。\n")
  draw <- function() forest(mi)
  size <- .forest_size(draw)
  .save(draw, file, size[["width"]], size[["height"]], formats, "逐一剔除的敏感性分析图")
  invisible(res)
}

fh_export <- function(m, file = "meta_results") {
  w <- if (.random(m)) m$w.random else m$w.common
  studies <- data.frame(`研究` = m$studlab, check.names = FALSE)
  studies[[m$sm]] <- round(.bt(m, m$TE), 2)
  studies[["95% CI 下限"]] <- round(.bt(m, m$lower), 2)
  studies[["95% CI 上限"]] <- round(.bt(m, m$upper), 2)
  studies[["权重 %"]] <- round(100 * w / sum(w, na.rm = TRUE), 1)
  s <- fh_report(m, quiet = TRUE)
  pooled <-data.frame(`项目` = c(.method_text(m), "效应量", "95% CI 下限", "95% CI 上限", "Z", "P",
                                  "Tau²", "Chi²", "df", "异质性 P", "I² %"),
                       `数值` = c("", round(s$estimate, 2), round(s$lower, 2), round(s$upper, 2), round(s$z, 2),
                                signif(s$p, 3), round(s$tau2, 3), round(s$Q, 2), s$df, signif(s$p_Q, 3),
                                round(100 * s$I2)), check.names = FALSE)
  path <- file.path("figures", paste0(file, ".xlsx"))
  write_xlsx(list(`各研究` = studies, `合并结果` = pooled), path)
  cat(sprintf("已保存结果表（工作表「各研究」「合并结果」）：%s\n", path))
  invisible(path)
}

# ================================================================ 目录
fh_help <- function() {
  cat(paste(c(
    "FinHelm meta 分析模板（算法和版式对齐 RevMan 5，数字来自 meta 包）。列名都用字符串传。",
    "",
    "合并（返回 meta 对象 m，同时打印 RevMan 格式的结果）：",
    "  fh_meta_bin(data, study, event_e, n_e, event_c, n_c, sm = \"RR\"|\"OR\"|\"RD\", method = \"MH\"|\"Inverse\"|\"Peto\",",
    "              model = \"random\"|\"fixed\", label_e = \"试验组\", label_c = \"对照组\", subgroup = NULL, outcome = \"\")",
    "  fh_meta_cont(data, study, mean_e, sd_e, n_e, mean_c, sd_c, n_c, sm = \"MD\"|\"SMD\", model = ..., subgroup = NULL)",
    "  fh_meta_gen(data, study, estimate, lower, upper, sm = \"HR\" 等, model = ...)   # 文献直接给的效应量和 95% CI",
    "",
    "结果：",
    "  fh_report(m)       RevMan 格式的合并结果、异质性、亚组",
    "  fh_methods(m)      一段方法学描述（中文）",
    "  fh_export(m)       各研究结果和合并结果存成 figures/meta_results.xlsx",
    "",
    "图（存到 figures/，默认 PNG 300dpi + PDF，formats = c(\"png\", \"pdf\", \"tiff\")）：",
    "  fh_forest(m, file = \"forest\", rob = NULL, label_left = \"利于试验组\", label_right = \"利于对照组\")",
    "      RevMan 5 版式的森林图；rob 传 fh_rob() 的返回值，右边加上各领域的 + ? -",
    "  fh_rob(data, tool = \"RoB1\"|\"RoB2\", study = \"study\", domains = NULL, overall = NULL, labels = NULL)",
    "      偏倚风险汇总图（rob_summary）和比例条图（rob_graph）；domains 默认除研究、overall 外的所有列；",
    "      判定可以写 Low/Unclear/Some concerns/High、低/不清楚/高、+/?/-；RoB 2 一般有 overall 列",
    "  fh_funnel(m)       漏斗图；10 项研究以上才做 Egger 检验",
    "  fh_sensitivity(m)  逐一剔除的敏感性分析（表 + 图）",
    "",
    "读 Excel：read_excel(\"inputs/文件.xlsx\", sheet = \"工作表\", skip = 表头上面空几行)"
  ), collapse = "\n"), "\n")
}
