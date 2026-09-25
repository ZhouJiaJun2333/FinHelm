# 沙箱里的 R 内核：一个常驻进程，变量在调用之间保留。协议和 tools/python/kernel.py 一样：
#     宿主 → 内核（stdin）      {"op": "exec", "code": ..., "timeout": 秒}   {"op": "data", ...}   {"op": "saved", ...}
#     内核 → 宿主（/dev/stdout） {"op": "ready"}  {"op": "need", "ref": "r3"}  {"op": "save", ...}  {"op": "done", ...}
# 用户代码的输出靠 sink 收进来，协议单独开一个连接写，互不干扰。
# 和 R 控制台一样，每个顶层表达式的值只要可见就打印（ggplot 对象打印出来就是画图）。

suppressPackageStartupMessages(library(jsonlite))

PROTO_IN <- file("stdin", open = "r", encoding = "UTF-8")
PROTO_OUT <- file("/dev/stdout", open = "w")
MAX_OUTPUT_CHARS <- 20000L

send <- function(msg) {
  writeLines(as.character(toJSON(msg, auto_unbox = TRUE, null = "null", na = "null", digits = NA)), PROTO_OUT)
  flush(PROTO_OUT)
}

receive <- function() {
  line <- readLines(PROTO_IN, n = 1L)
  if (length(line) == 0L) base::quit(save = "no", status = 0L)
  fromJSON(line, simplifyVector = FALSE)
}

# ---------------------------------------------------------------- 结果仓库（r 编号）
.results <- new.env()

load_result <- function(ref) {
  ref <- trimws(as.character(ref))
  if (is.null(.results[[ref]])) {
    send(list(op = "need", ref = ref))
    reply <- receive()
    if (!is.null(reply$error)) stop(reply$error, call. = FALSE)
    cols <- lapply(seq_along(reply$columns), function(j) {
      unlist(lapply(reply$rows, function(row) if (is.null(row[[j]])) NA else row[[j]]))
    })
    df <- as.data.frame(setNames(cols, unlist(reply$columns)), check.names = FALSE, stringsAsFactors = FALSE)
    for (col in unlist(reply$dates)) df[[col]] <- as.Date(substr(df[[col]], 1L, 10L))
    if (isTRUE(reply$truncated)) cat(sprintf("注意：%s 超过 %d 行被截断了，这里只有前 %d 行。\n", ref, nrow(df), nrow(df)))
    assign(ref, df, envir = .results)
  }
  get(ref, envir = .results)
}

# 把一张表存进宿主的结果仓库、编上号（比如 r5）：回答里写 {{r5}} 用户就能看到整张表，/save r5 能导出
save_result <- function(table, title = "") {
  df <- as.data.frame(table, check.names = FALSE, stringsAsFactors = FALSE)
  rn <- rownames(df)
  if (!identical(rn, as.character(seq_len(nrow(df))))) df <- cbind(data.frame(`项目` = rn, check.names = FALSE), df)
  for (col in names(df)) if (is.factor(df[[col]]) || inherits(df[[col]], "Date")) df[[col]] <- as.character(df[[col]])
  rows <- lapply(seq_len(nrow(df)), function(i) unname(lapply(df, function(col) col[[i]])))
  send(list(op = "save", title = as.character(title), columns = names(df), rows = rows))
  reply <- receive()
  if (!is.null(reply$error)) stop(reply$error, call. = FALSE)
  cut <- if (isTRUE(reply$truncated)) sprintf("（超过上限，只存了前 %d 行）", reply$rows) else ""
  cat(sprintf("已存为结果 %s：%d 行 × %d 列%s。回答里单独一行写 {{%s}}，用户会在那里看到整张表。\n",
              reply$ref, reply$rows, ncol(df), cut, reply$ref))
  invisible(reply$ref)
}

# ---------------------------------------------------------------- 执行
run <- function(code, timeout) {
  out <- textConnection("captured", open = "w", local = TRUE)
  sink(out)
  sink(out, type = "message")
  error <- NULL
  at <- NA_integer_
  plots_before <- .plot_files()
  files_before <- .figure_files()
  png(file.path("figures", ".r-%03d.png"), width = 1600, height = 1000, res = 150, type = "cairo")
  setTimeLimit(elapsed = timeout, transient = TRUE)
  tryCatch({
    exprs <- parse(text = code, keep.source = TRUE)
    lines <- vapply(attr(exprs, "srcref"), function(s) s[[1L]], integer(1))
    for (i in seq_along(exprs)) {
      at <- lines[[i]]
      withCallingHandlers({
        res <- withVisible(eval(exprs[[i]], envir = globalenv()))
        if (res$visible) print(res$value)
      }, warning = function(w) {
        cat("警告：", conditionMessage(w), "\n", sep = "")
        invokeRestart("muffleWarning")
      })
    }
  }, error = function(e) {
    msg <- conditionMessage(e)
    call <- conditionCall(e)
    if (grepl("elapsed time limit", msg)) {
      msg <- sprintf("执行超过 %g 秒，被中断了。已经算完的变量还在。", timeout)
    } else {
      shown <- if (is.null(call)) "" else deparse(call, nlines = 1L)
      if (nzchar(shown) && !startsWith(shown, "eval(exprs")) msg <- paste0(shown, "：", msg)   # 内核自己的 eval 不算
      msg <- if (is.na(at)) paste0("代码解析出错：", msg) else sprintf("第 %d 行出错：%s", at, msg)
    }
    error <<- msg
  })
  setTimeLimit()
  while (dev.cur() > 1L) dev.off()
  sink(type = "message")
  sink()
  close(out)
  text <- paste(captured, collapse = "\n")
  if (nchar(text) > MAX_OUTPUT_CHARS) {
    text <- paste0(substr(text, 1L, MAX_OUTPUT_CHARS), sprintf("\n…（输出太长，只保留了前 %d 个字符）", MAX_OUTPUT_CHARS))
  }
  written <- .changed_files(files_before)
  # 自己存了图（ggsave、png()…）就不再自动存：模型常常先 print(p) 看一眼再 ggsave(p)，
  # 自动设备上那一页和存下的文件是同一张图。对不上是哪一页，只能整次调用一起算
  own <- any(grepl("\\.(png|pdf|tiff?|svg|jpe?g|eps)$", unlist(written), ignore.case = TRUE))
  list(output = if (nzchar(text)) paste0(text, "\n") else "", value = NULL, error = error,
       figures = c(written, .collect_plots(plots_before, keep = !own)))
}

.plot_files <- function() list.files("figures", pattern = "^\\.r-\\d+\\.png$", full.names = TRUE, all.files = TRUE)

# figures/ 下的文件和修改时间（不含自动出图的临时文件）：模板和用户代码自己存的图也要报给宿主
.figure_files <- function() {
  files <- setdiff(list.files("figures", full.names = TRUE), .plot_files())
  setNames(file.mtime(files), files)
}

.changed_files <- function(before) {
  now <- .figure_files()
  changed <- names(now)[is.na(before[names(now)]) | now > before[names(now)]]
  as.list(changed)
}

.collect_plots <- function(before, keep = TRUE) {
  # 自动开的 png 设备：每画一页一个临时文件，改名成 fig-N.png（和 Python 共用编号）；keep = FALSE 就删掉
  saved <- character(0)
  for (tmp in setdiff(.plot_files(), before)) {
    if (!keep) {
      file.remove(tmp)
      next
    }
    n <- 1L
    while (file.exists(path <- sprintf("figures/fig-%d.png", n))) n <- n + 1L
    file.rename(tmp, path)
    saved <- c(saved, path)
  }
  as.list(saved)
}

# 用户代码退出不了内核
q <- quit <- function(...) stop("沙箱里不能退出 R。", call. = FALSE)

main <- function() {
  dir.create("figures", showWarnings = FALSE)
  # 模板库和内核挂在同一个目录
  here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
  templates <- file.path(here, "templates.R")
  if (file.exists(templates)) sys.source(templates, envir = globalenv())
  send(list(op = "ready", r = paste(R.version$major, R.version$minor, sep = ".")))
  repeat {
    msg <- receive()
    if (identical(msg$op, "exec")) {
      send(c(list(op = "done"), run(msg$code, as.numeric(if (is.null(msg$timeout)) 60 else msg$timeout))))
    }
  }
}

main()
