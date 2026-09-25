# run_python 的沙箱镜像：只装分析要用的库，运行时断网、不许装包。
# 内核代码（src/data_agent/tools/python/kernel.py）运行时只读挂进来，改了不用重建镜像。
#
#   docker build -t finhelm-sandbox docker/sandbox
#   国内网络慢就换源：--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

# 中文字体：不装的话图里的中文是方块
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
        "pandas>=2.2" "numpy>=2.0" "scipy>=1.13" "statsmodels>=0.14" \
        "matplotlib>=3.9" "openpyxl>=3.1" "pdfplumber>=0.11"

# 非 root 用户跑；uid 和 sandbox.py 里的 --user 对上
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /work
