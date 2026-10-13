"""Fila de jobs em SQLite. Sem Redis: 1 worker, 1 música por vez (VPS de 4 GB)."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .ingest import DATA_DIR

DB_PATH = DATA_DIR / "kicksnare.db"

SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    pw_hash  TEXT NOT NULL,
    created  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created    REAL NOT NULL,
    expires    REAL NOT NULL
)
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id        TEXT PRIMARY KEY,
    title     TEXT,
    source    TEXT NOT NULL,          -- URL ou caminho do upload
    status    TEXT NOT NULL,          -- queued | running | done | error
    stage     TEXT,                   -- ingest | separate | transcribe | finalize
    message   TEXT,
    error     TEXT,
    created   REAL NOT NULL,
    started   REAL,
    finished  REAL,
    stats     TEXT
)
"""


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(SCHEMA)
    for stmt in SCHEMA_USERS.split(";"):
        con.execute(stmt)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(songs)")}
    if "owner_id" not in cols:      # migração: músicas existentes ficam sem dono até alguém assumir
        con.execute("ALTER TABLE songs ADD COLUMN owner_id INTEGER")
    for col, ddl in (("duration", "REAL"), ("stage_started", "REAL"), ("timings", "TEXT"), ("progress", "REAL")):
        if col not in cols:         # estimativa de tempo: duração do áudio, início da etapa, tempos e progresso
            con.execute(f"ALTER TABLE songs ADD COLUMN {col} {ddl}")
    ucols = {r["name"] for r in con.execute("PRAGMA table_info(users)")}
    if "is_admin" not in ucols:
        con.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
    if "status" not in ucols:       # contas anteriores à aprovação continuam ativas
        con.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'")
    try:
        yield con
    finally:
        con.close()


def _row(r) -> dict:
    d = dict(r)
    d["stats"] = json.loads(d["stats"]) if d.get("stats") else None
    d["timings"] = json.loads(d["timings"]) if d.get("timings") else {}
    return d


def enqueue(song_id: str, source: str, title: str | None = None, owner_id: int | None = None) -> dict:
    """Cria a música na fila. Se já existe e está pronta/em andamento, não duplica."""
    with connect() as con:
        cur = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        if cur and cur["status"] in ("queued", "running", "done"):
            return _row(cur)
        now = time.time()
        con.execute(
            "INSERT OR REPLACE INTO songs(id,title,source,status,stage,message,created,owner_id) VALUES(?,?,?,?,?,?,?,?)",
            (song_id, title, source, "queued", None, "Na fila", now, owner_id),
        )
        return _row(con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone())


def claim_next() -> dict | None:
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT * FROM songs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not r:
            con.execute("COMMIT")
            return None
        con.execute(
            "UPDATE songs SET status='running', started=?, message='Iniciando' WHERE id=?",
            (time.time(), r["id"]),
        )
        con.execute("COMMIT")
        return _row(r)


def update(song_id: str, **fields) -> None:
    if "stats" in fields and fields["stats"] is not None:
        fields["stats"] = json.dumps(fields["stats"])
    if "timings" in fields and fields["timings"] is not None:
        fields["timings"] = json.dumps(fields["timings"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE songs SET {cols} WHERE id=?", (*fields.values(), song_id))


def get(song_id: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        return _row(r) if r else None


def list_songs(owner_id: int | None = None) -> list[dict]:
    """Do dono pedido; com owner_id=None, de todos (visão do admin). Inclui o nome do dono em `owner`."""
    q = ("SELECT songs.*, users.username AS owner FROM songs "
         "LEFT JOIN users ON users.id = songs.owner_id")
    with connect() as con:
        if owner_id is None:
            rows = con.execute(q + " ORDER BY songs.created DESC")
        else:
            rows = con.execute(q + " WHERE songs.owner_id=? ORDER BY songs.created DESC", (owner_id,))
        return [_row(r) for r in rows]


def song_counts(owner_id: int) -> tuple[int, int]:
    """(total de músicas, jobs ativos = na fila ou rodando) do usuário."""
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM songs WHERE owner_id=?", (owner_id,)).fetchone()[0]
        active = con.execute(
            "SELECT COUNT(*) FROM songs WHERE owner_id=? AND status IN ('queued','running')", (owner_id,)
        ).fetchone()[0]
        return total, active


def active_total() -> int:
    """Músicas na fila ou processando, de todos os usuários."""
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM songs WHERE status IN ('queued','running')").fetchone()[0]


def delete_song(song_id: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM songs WHERE id=?", (song_id,))


def assign_orphans(owner_id: int) -> int:
    """Músicas criadas antes das contas passam para esse usuário."""
    with connect() as con:
        return con.execute("UPDATE songs SET owner_id=? WHERE owner_id IS NULL", (owner_id,)).rowcount


# ---------- usuários e sessões ----------
def set_admin(user_id: int, is_admin: bool = True) -> None:
    with connect() as con:
        con.execute("UPDATE users SET is_admin=? WHERE id=?", (1 if is_admin else 0, user_id))


USER_STATUSES = ("pending", "approved", "rejected", "blocked")


def create_user(username: str, pw_hash: str, is_admin: bool = False, status: str = "approved") -> int | None:
    """Retorna o id, ou None se o nome já existe."""
    with connect() as con:
        try:
            cur = con.execute(
                "INSERT INTO users(username, pw_hash, created, is_admin, status) VALUES(?,?,?,?,?)",
                (username, pw_hash, time.time(), 1 if is_admin else 0, status),
            )
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            return None


def get_user_by_name(username: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else None


def set_password(user_id: int, pw_hash: str) -> None:
    with connect() as con:
        con.execute("UPDATE users SET pw_hash=? WHERE id=?", (pw_hash, user_id))
        con.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))   # derruba as outras sessões


def create_session(user_id: int, token_hash: str, days: int) -> None:
    now = time.time()
    with connect() as con:
        con.execute("DELETE FROM sessions WHERE expires < ?", (now,))
        con.execute(
            "INSERT INTO sessions(token_hash, user_id, created, expires) VALUES(?,?,?,?)",
            (token_hash, user_id, now, now + days * 86400),
        )


def user_for_session(token_hash: str) -> dict | None:
    with connect() as con:
        r = con.execute(
            "SELECT u.id, u.username, u.is_admin FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=? AND s.expires > ? AND u.status='approved'",
            (token_hash, time.time()),
        ).fetchone()
        return dict(r) if r else None


def delete_session(token_hash: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))


def requeue_interrupted() -> int:
    """Jobs 'running' ficam órfãos se o worker morreu; voltam para a fila."""
    with connect() as con:
        return con.execute(
            "UPDATE songs SET status='queued', message='Reenfileirado após reinício' WHERE status='running'"
        ).rowcount


def list_users() -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT u.id, u.username, u.created, u.is_admin, u.status, "
            "(SELECT COUNT(*) FROM songs WHERE owner_id = u.id) AS songs "
            "FROM users u ORDER BY (u.status='pending') DESC, u.created DESC"
        )
        return [dict(r) for r in rows]


def count_pending() -> int:
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM users WHERE status='pending'").fetchone()[0]


def get_user(user_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(r) if r else None


def set_user_status(user_id: int, status: str) -> None:
    """Fora de 'approved' o usuário perde as sessões abertas na hora."""
    if status not in USER_STATUSES:
        raise ValueError(status)
    with connect() as con:
        con.execute("UPDATE users SET status=? WHERE id=?", (status, user_id))
        if status != "approved":
            con.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))


def active_songs() -> list[dict]:
    """Fila inteira (todos os usuários), na ordem em que o worker as pega."""
    with connect() as con:
        rows = con.execute(
            "SELECT id, status, stage, created, started, duration, stage_started, progress, timings "
            "FROM songs WHERE status IN ('queued','running') ORDER BY (status='running') DESC, created"
        )
        return [_row(r) for r in rows]


def history(limit: int = 15) -> list[dict]:
    """Últimos jobs concluídos com os tempos de cada etapa (base da estimativa)."""
    with connect() as con:
        rows = con.execute(
            "SELECT duration, timings FROM songs WHERE status='done' AND duration > 0 AND timings IS NOT NULL "
            "ORDER BY finished DESC LIMIT ?", (limit,)
        )
        return [_row(r) for r in rows]
