# run_r 的沙箱镜像：meta 分析和偏倚风险图要用的 R 包。和 Python 沙箱一样运行时断网、不许装包。
# 内核代码（src/data_agent/tools/r/kernel.R）运行时只读挂进来，改了不用重建镜像。
#
#   docker build -t finhelm-sandbox-r docker/sandbox-r
#
# rocker 默认从 Posit 的二进制源装 CRAN 包（按 R 版本锁日期），不用现场编译
FROM rocker/r-ver:4.4

# 中文字体 + cairo（png(type = "cairo") 靠 fontconfig 逐字回退，中文不会变方块）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-wqy-microhei libcairo2 libfontconfig1 libfreetype6 libxt6 \
    && rm -rf /var/lib/apt/lists/*

# 国内网络下载偶尔会断：拉长超时，失败了重试（--skipinstalled 让重试从断的地方接着装）
RUN echo 'options(timeout = 600)' >> /usr/local/lib/R/etc/Rprofile.site \
    && for i in 1 2 3; do \
         install2.r --error --skipinstalled jsonlite meta metafor robvis readxl writexl ggplot2 && break; \
         [ "$i" = 3 ] && exit 1; sleep 10; \
       done \
    && rm -rf /tmp/downloaded_packages

# 模板裁森林图的白边要读 PNG 像素（meta 只算得出图的高度，算不出宽度）
RUN for i in 1 2 3; do install2.r --error --skipinstalled png && break; [ "$i" = 3 ] && exit 1; sleep 10; done

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /work
