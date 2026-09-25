"""session —— 会话落盘：会话目录、对话日志、恢复会话。

    store.py   Session：目录布局、每轮成功后追加日志、读回历史
    codec.py   历史条目（消息 + 标记）↔ JSON
"""

from .store import Session

__all__ = ["Session"]
