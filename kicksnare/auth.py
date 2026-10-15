"""Contas, senhas e sessões.

- Senha: scrypt (stdlib) com sal aleatório.
- Sessão: token aleatório em cookie HttpOnly; no banco guarda-se só o SHA-256 do token.
- Limite de tentativas em memória (por IP+usuário) para o login e o cadastro.
"""
import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque

SESSION_COOKIE = "ks_session"
SESSION_DAYS = 30
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
MIN_PASSWORD = 8
MAX_PASSWORD = 200

_N, _R, _P = 2 ** 14, 8, 1


def normalize_username(name: str) -> str:
    return (name or "").strip().lower()


def validate_username(name: str) -> str | None:
    """None se válido; senão a mensagem de erro."""
    if not USERNAME_RE.match(name):
        return "usuário: 3 a 32 caracteres (letras minúsculas, números, '.', '_' ou '-')"
    return None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_NAME = 60


def validate_profile(first_name: str, last_name: str, email: str) -> str | None:
    """Dados pessoais do cadastro (já sem espaços nas pontas). None se válidos."""
    if not first_name or not last_name:
        return "informe nome e sobrenome"
    if len(first_name) > MAX_NAME or len(last_name) > MAX_NAME:
        return f"nome e sobrenome: até {MAX_NAME} caracteres cada"
    if len(email) > 254 or not EMAIL_RE.match(email):
        return "e-mail inválido"
    return None


def validate_password(pw: str) -> str | None:
    if len(pw) < MIN_PASSWORD:
        return f"senha: mínimo de {MIN_PASSWORD} caracteres"
    if len(pw) > MAX_PASSWORD:
        return "senha longa demais"
    return None


def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, bytes.fromhex(dk))
    except (ValueError, TypeError):
        return False


# hash falso para gastar o mesmo tempo quando o usuário não existe (evita revelar contas pelo tempo)
_DUMMY = hash_password("kicksnare-dummy-password")


def burn_time(pw: str) -> None:
    verify_password(pw, _DUMMY)


def new_token() -> tuple[str, str]:
    """(token para o cookie, hash para o banco)."""
    tok = secrets.token_urlsafe(32)
    return tok, token_hash(tok)


def token_hash(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


class Throttle:
    """Máx. `limit` eventos por `window` s para uma chave. Em memória (zera ao reiniciar)."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits: dict = defaultdict(deque)

    def _trim(self, key: str, now: float) -> deque:
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        return len(self._trim(key, time.time())) >= self.limit

    def hit(self, key: str) -> None:
        now = time.time()
        self._trim(key, now).append(now)

    def clear(self, key: str) -> None:
        self.hits.pop(key, None)


login_failures = Throttle(limit=6, window=600)    # por IP+usuário
signups = Throttle(limit=5, window=3600)          # contas criadas por IP
