"""Web 界面入口：python run_web.py（默认 http://127.0.0.1:8765）

和 run.py 一样把 src 放到最前面：环境里可能装着同名的无关包 data_agent。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_agent.web.server import main  # noqa: E402

if __name__ == "__main__":
    main()
