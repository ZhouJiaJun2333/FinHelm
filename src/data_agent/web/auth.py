"""账号和登录。没有建账号 = 单用户（只给本机用）；建了账号就必须登录，每个人的会话、记忆分开放。

    python run_web.py adduser 名字      # 建账号 / 改密码（交互输入密码）
    python run_web.py deluser 名字
    python run_web.py users

账号在 <WEB_DIR>/users.json（密码只存 scrypt 哈希），登录凭证是签了名的 cookie（名字 + 过期时间 + HMAC），
签名密钥在 <WEB_DIR>/secret。服务器重启不用重新登录；删掉账号，它的 cookie 马上失效。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

COOKIE = "finhelm_login"
TTL_S = 30 * 86400
# 名字会拼进目录名，只收安全的字符
NAME = re.compile(r"[A-Za-z0-9_-]{1,32}")
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)


class Users:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "users.json"

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _save(self, users: dict[str, dict[str, str]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def names(self) -> list[str]:
        return sorted(self._load())

    def add(self, name: str, password: str) -> None:
        if not NAME.fullmatch(name):
            raise ValueError("名字只能用字母、数字、_ 和 -，最多 32 个字符")
        if len(password) < 8:
            raise ValueError("密码至少 8 位")
        salt = secrets.token_bytes(16)
        users = self._load()
        users[name] = {"salt": salt.hex(), "hash": _hash(password, salt).hex()}
        self._save(users)

    def remove(self, name: str) -> bool:
        users = self._load()
        if users.pop(name, None) is None:
            return False
        self._save(users)
        return True

    def verify(self, name: str, password: str) -> bool:
        user = self._load().get(name)
        if user is None:
            _hash(password, b"x" * 16)          # 不存在的名字也花同样的时间，别让人试出有哪些账号
            return False
        return hmac.compare_digest(_hash(password, bytes.fromhex(user["salt"])), bytes.fromhex(user["hash"]))

    # ------------------------------------------------------------ 登录凭证
    def _secret(self) -> bytes:
        path = self.root / "secret"
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(secrets.token_bytes(32))
        return path.read_bytes()

    def token(self, name: str, now: float | None = None) -> str:
        payload = f"{name}.{int((time.time() if now is None else now) + TTL_S)}"
        sig = hmac.new(self._secret(), payload.encode(), hashlib.sha256).digest()
        return f"{payload}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"

    def check(self, token: str | None, now: float | None = None) -> str | None:
        """合法、没过期、账号还在 → 名字；否则 None。"""
        if not token or token.count(".") != 2:
            return None
        name, expires, sig = token.split(".")
        payload = f"{name}.{expires}"
        want = base64.urlsafe_b64encode(hmac.new(self._secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(want.decode().rstrip("="), sig):
            return None
        if not expires.isdigit() or int(expires) < (time.time() if now is None else now) or name not in self._load():
            return None
        return name
