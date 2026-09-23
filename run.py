"""项目入口：python run.py

放在根目录只是为了少打几个字。真正的逻辑在 src/data_agent/cli.py。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
