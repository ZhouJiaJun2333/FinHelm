"""评测：用一批标准题，让真实的 Agent 跑一遍，自动判分，全过程存档。

    python -m evals.run --cases shop --trials 3

    cases/     题库（jsonl，答案存标准 SQL）
    cases.py   读题库
    runner.py  一道题跑一次：全新的 Agent，不开界面，收集事件，判分
    graders.py 判分（纯函数，tests/test_eval_graders.py 测它）
    report.py  汇总、按标签统计、和上一次逐题对比
    run.py     命令行入口
    runs/      每次运行的记录（不进 git）

和 tests/ 的区别：tests 用假模型，几秒跑完，每次改代码都跑；
这里用真模型，要花时间和钱，改提示词 / 换模型 / 改策略时手动跑。
"""

import sys
from pathlib import Path

# 和根目录的 run.py 一样：让 `python -m evals.run` 不用安装也能找到 src/data_agent
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
