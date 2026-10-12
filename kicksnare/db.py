"""Fila de jobs em SQLite. Sem Redis: 1 worker, 1 música por vez (VPS de 4 GB)."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .ingest import DATA_DIR

DB_PATH = DATA_DIR / "kicksnare.db"

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
    try:
        yield con
    finally:
        con.close()


def _row(r) -> dict:
    d = dict(r)
    d["stats"] = json.loads(d["stats"]) if d.get("stats") else None
    return d


def enqueue(song_id: str, source: str, title: str | None = None) -> dict:
    """Cria a música na fila. Se já existe e está pronta/em andamento, não duplica."""
    with connect() as con:
        cur = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        if cur and cur["status"] in ("queued", "running", "done"):
            return _row(cur)
        now = time.time()
        con.execute(
            "INSERT OR REPLACE INTO songs(id,title,source,status,stage,message,created) VALUES(?,?,?,?,?,?,?)",
            (song_id, title, source, "queued", None, "Na fila", now),
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
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE songs SET {cols} WHERE id=?", (*fields.values(), song_id))


def get(song_id: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        return _row(r) if r else None


def list_songs() -> list[dict]:
    with connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM songs ORDER BY created DESC")]


def requeue_interrupted() -> int:
    """Jobs 'running' ficam órfãos se o worker morreu; voltam para a fila."""
    with connect() as con:
        return con.execute(
            "UPDATE songs SET status='queued', message='Reenfileirado após reinício' WHERE status='running'"
        ).rowcount
