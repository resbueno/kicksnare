"""API e páginas do Kicksnare.

Rodar: uvicorn kicksnare.asgi:application --host 127.0.0.1 --port 8000

Contas: cadastro aberto; cada usuário vê apenas as músicas que enviou. Para a conta do
dono (e para que músicas criadas antes das contas passem a ser dele) defina
KICKSNARE_ADMIN_USER e KICKSNARE_ADMIN_PASSWORD.
"""
import contextvars
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth as authlib
from . import db, eta as etalib, sheet as sheetlib
from .ingest import DATA_DIR, file_song_id
from .separate import ensure_no_drums
from .patterns import (
    Edit, apply_edits, cluster_bars, edits_from_json, load_measures, with_state, _ranges,
)
from .transcribe import ORDER, PIECES, to_musicxml

STATIC = Path(__file__).parent / "static"
# Prefixo de URL quando roda atrás de um proxy em subcaminho (ex.: /kicksnare). Vazio = raiz.
BASE = os.environ.get("KICKSNARE_BASE", "").rstrip("/")
UPLOADS = DATA_DIR / "uploads"
MAX_UPLOAD = 300 * 1024 * 1024
MAX_SONGS = int(os.environ.get("KICKSNARE_MAX_SONGS", "25"))      # por usuário
MAX_ACTIVE = int(os.environ.get("KICKSNARE_MAX_ACTIVE", "3"))     # na fila + processando, por usuário
MAX_QUEUE_TOTAL = int(os.environ.get("KICKSNARE_MAX_QUEUE", "20"))  # fila inteira (todos os usuários)
MIN_FREE_GB = float(os.environ.get("KICKSNARE_MIN_FREE_GB", "15"))  # recusa novos envios abaixo disso
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
YT_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC / name).read_text().replace("__BASE__", BASE))


# ---------- sessão / usuário ----------
_user_var: contextvars.ContextVar = contextvars.ContextVar("ks_user", default=None)


class LoginRedirect(Exception):
    """Página protegida sem sessão: manda para /login."""


def _rel_path(request: Request) -> str:
    path = request.url.path
    if BASE and path.startswith(BASE):
        path = path[len(BASE):]
    return path or "/"


def _client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


def _is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"


def _session_user(request: Request) -> dict | None:
    tok = request.cookies.get(authlib.SESSION_COOKIE)
    return db.user_for_session(authlib.token_hash(tok)) if tok else None


PUBLIC_PATHS = {"/login", "/favicon.ico"}


async def auth_dep(request: Request) -> None:
    """async de propósito: o contextvar precisa ficar visível para os endpoints síncronos."""
    path = _rel_path(request)
    user = _session_user(request)
    _user_var.set(user)
    if user or path in PUBLIC_PATHS or path.startswith("/api/auth/"):
        return
    if path.startswith("/api/"):
        raise HTTPException(401, "faça login")
    raise LoginRedirect()


def _require_user() -> dict:
    user = _user_var.get()
    if not user:
        raise HTTPException(401, "faça login")
    return user


app = FastAPI(title="Kicksnare", dependencies=[Depends(auth_dep)])


@app.exception_handler(LoginRedirect)
async def _login_redirect(request: Request, exc: LoginRedirect) -> RedirectResponse:
    return RedirectResponse(f"{BASE}/login", status_code=303)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _fresh(song: dict) -> dict:
    """stats.json é a fonte de verdade: o banco guarda a cópia da última execução do worker,
    que fica velha se a transcrição for refeita (CLI, reprocessamento)."""
    f = DATA_DIR / song["id"] / "stats.json"
    if song.get("status") == "done" and f.exists():
        try:
            song = {**song, "stats": json.loads(f.read_text())}
        except (OSError, ValueError):
            pass
    return song


def _song(song_id: str) -> dict:
    """Música do usuário logado (ou qualquer uma, para o admin). De outro dono = 404."""
    user = _require_user()
    if not ID_RE.match(song_id):
        raise HTTPException(404, "música não encontrada")
    song = db.get(song_id)
    if not song or (song.get("owner_id") != user["id"] and not user.get("is_admin")):
        raise HTTPException(404, "música não encontrada")
    return _fresh(song)


def _ready(song_id: str) -> Path:
    song = _song(song_id)
    if song["status"] != "done":
        raise HTTPException(409, "a música ainda não foi processada")
    return DATA_DIR / song_id


# ---------- páginas ----------
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.png", media_type="image/png")


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if _session_user(request):
        return RedirectResponse(f"{BASE}/", status_code=303)
    return _page("login.html")


@app.get("/account", include_in_schema=False)
def account_page() -> HTMLResponse:
    return _page("account.html")


@app.get("/admin/users", include_in_schema=False)
def admin_users_page() -> HTMLResponse:
    _require_admin()
    return _page("admin-users.html")


# ---------- contas ----------
class Credentials(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current: str
    new: str


def _start_session(response: Response, request: Request, user_id: int) -> None:
    tok, th = authlib.new_token()
    db.create_session(user_id, th, authlib.SESSION_DAYS)
    response.set_cookie(
        authlib.SESSION_COOKIE, tok, max_age=authlib.SESSION_DAYS * 86400,
        httponly=True, samesite="lax", secure=_is_https(request), path=BASE or "/",
    )


@app.post("/api/auth/register")
def register(body: Credentials, request: Request, response: Response) -> dict:
    ip = _client_ip(request)
    if authlib.signups.blocked(ip):
        raise HTTPException(429, "muitos cadastros deste endereço; tente mais tarde")
    username = authlib.normalize_username(body.username)
    err = authlib.validate_username(username) or authlib.validate_password(body.password)
    if err:
        raise HTTPException(400, err)
    uid = db.create_user(username, authlib.hash_password(body.password), status="pending")
    if uid is None:
        raise HTTPException(409, "este usuário já existe")
    authlib.signups.hit(ip)
    # sem sessão: o acesso só começa depois que o administrador aprovar
    return {"username": username, "pending": True,
            "message": "Cadastro enviado. Você poderá entrar assim que o administrador aprovar."}


@app.post("/api/auth/login")
def login(body: Credentials, request: Request, response: Response) -> dict:
    username = authlib.normalize_username(body.username)
    key = f"{_client_ip(request)}|{username}"
    if authlib.login_failures.blocked(key):
        raise HTTPException(429, "muitas tentativas; aguarde alguns minutos")
    user = db.get_user_by_name(username)
    if user and authlib.verify_password(body.password, user["pw_hash"]):
        authlib.login_failures.clear(key)
        # o estado da conta só é revelado a quem sabe a senha
        if user.get("status", "approved") == "pending":
            raise HTTPException(403, "Sua conta está aguardando a aprovação do administrador.")
        if user.get("status") in ("rejected", "blocked"):
            raise HTTPException(403, "Esta conta não está autorizada a acessar.")
        _start_session(response, request, user["id"])
        return {"username": user["username"]}
    if not user:
        authlib.burn_time(body.password)   # mesmo custo: não revela se o usuário existe
    authlib.login_failures.hit(key)
    raise HTTPException(401, "usuário ou senha incorretos")


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict:
    tok = request.cookies.get(authlib.SESSION_COOKIE)
    if tok:
        db.delete_session(authlib.token_hash(tok))
    response.delete_cookie(authlib.SESSION_COOKIE, path=BASE or "/")
    return {"ok": True}


@app.get("/api/auth/me")
def me() -> dict:
    u = _require_user()
    out = {"username": u["username"], "is_admin": bool(u.get("is_admin"))}
    if u.get("is_admin"):
        out["pending_count"] = db.count_pending()
    return out


# ---------- administração de usuários ----------
def _require_admin() -> dict:
    u = _require_user()
    if not u.get("is_admin"):
        raise HTTPException(403, "apenas o administrador")
    return u


class StatusChange(BaseModel):
    status: str   # approved | rejected | blocked | pending


@app.get("/api/admin/users")
def admin_users() -> list[dict]:
    _require_admin()
    return db.list_users()


@app.post("/api/admin/users/{user_id}/status")
def admin_set_status(user_id: int, body: StatusChange) -> dict:
    admin = _require_admin()
    if body.status not in db.USER_STATUSES:
        raise HTTPException(400, "status inválido")
    target = db.get_user(user_id)
    if not target:
        raise HTTPException(404, "usuário não encontrado")
    if target["id"] == admin["id"] or target.get("is_admin"):
        raise HTTPException(400, "não é possível alterar a conta de um administrador")
    db.set_user_status(user_id, body.status)
    return {"id": user_id, "status": body.status}


@app.post("/api/auth/password")
def change_password(body: PasswordChange, request: Request, response: Response) -> dict:
    user = _require_user()
    row = db.get_user_by_name(user["username"])
    if not row or not authlib.verify_password(body.current, row["pw_hash"]):
        raise HTTPException(401, "senha atual incorreta")
    err = authlib.validate_password(body.new)
    if err:
        raise HTTPException(400, err)
    db.set_password(user["id"], authlib.hash_password(body.new))   # encerra as outras sessões
    _start_session(response, request, user["id"])
    return {"ok": True}


@app.get("/songs/{song_id}", include_in_schema=False)
def song_page(song_id: str) -> HTMLResponse:
    _song(song_id)
    return _page("song.html")


# ---------- fila ----------
class NewSong(BaseModel):
    url: str


def _scoped_id(user: dict, raw: str) -> str:
    """O mesmo vídeo enviado por dois usuários vira duas músicas (dados, correções e PDFs separados)."""
    return f"u{user['id']}-{raw}"


def _check_quota(user: dict) -> None:
    if user.get("is_admin"):          # o dono não tem cota por usuário (a trava de disco vale para todos)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(DATA_DIR).free < MIN_FREE_GB * 1024 ** 3:
            raise HTTPException(507, "servidor sem espaço disponível no momento")
        return
    if db.active_total() >= MAX_QUEUE_TOTAL:
        raise HTTPException(503, "a fila de processamento está cheia; tente novamente em alguns minutos")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DATA_DIR).free < MIN_FREE_GB * 1024 ** 3:
        raise HTTPException(507, "servidor sem espaço disponível no momento")
    total, active = db.song_counts(user["id"])
    if active >= MAX_ACTIVE:
        raise HTTPException(429, f"você já tem {active} músicas em processamento; aguarde terminarem")
    if total >= MAX_SONGS:
        raise HTTPException(409, f"limite de {MAX_SONGS} músicas; apague alguma para enviar outra")


@app.get("/api/songs")
def list_songs(mine: bool = False) -> list[dict]:
    """O admin vê as músicas de todos (com `owner`); `?mine=1` restringe às dele."""
    user = _require_user()
    owner = user["id"] if (mine or not user.get("is_admin")) else None
    songs = [_fresh(s) for s in db.list_songs(owner)]
    if any(s["status"] in ("queued", "running") for s in songs):
        etalib.annotate(songs, db.active_songs(), db.history())
    return songs


@app.post("/api/songs")
def add_song(body: NewSong) -> dict:
    user = _require_user()
    url = body.url.strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "informe uma URL http(s)")
    m = YT_RE.search(url)
    raw = m.group(1) if m else "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]
    song_id = _scoped_id(user, raw)
    existing = db.get(song_id)
    if not (existing and existing.get("owner_id") == user["id"]
            and existing["status"] in ("queued", "running", "done")):
        _check_quota(user)
    return db.enqueue(song_id, url, owner_id=user["id"])


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    user = _require_user()
    _check_quota(user)
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(file.filename or "audio").name)[:120]
    folder = UPLOADS / str(user["id"])
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "arquivo grande demais (máx. 300 MB)")
            out.write(chunk)
    return db.enqueue(_scoped_id(user, file_song_id(dest)), str(dest), title=Path(name).stem, owner_id=user["id"])


@app.get("/api/songs/{song_id}")
def get_song(song_id: str) -> dict:
    song = _song(song_id)
    if song["status"] in ("queued", "running"):
        etalib.annotate([song], db.active_songs(), db.history())
    return song


class Rename(BaseModel):
    title: str | None = None
    artist: str | None = None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", text or "")).strip()


@app.patch("/api/songs/{song_id}")
def rename_song(song_id: str, body: Rename) -> dict:
    """Edita nome da música e/ou banda (autor). Vale para o dono e para o admin."""
    song = _song(song_id)
    fields: dict = {}
    if body.title is not None:
        title = _clean(body.title)
        if not title:
            raise HTTPException(400, "o nome não pode ficar vazio")
        if len(title) > 120:
            raise HTTPException(400, "nome longo demais (máximo de 120 caracteres)")
        fields["title"] = title
    if body.artist is not None:
        artist = _clean(body.artist)
        if len(artist) > 120:
            raise HTTPException(400, "nome da banda longo demais (máximo de 120 caracteres)")
        fields["artist"] = artist or None
    if not fields:
        return song
    db.update(song_id, **fields)
    title = fields.get("title")
    if title and title != song.get("title"):
        song_dir = DATA_DIR / song_id
        meta = song_dir / "meta.json"
        try:
            info = json.loads(meta.read_text()) if meta.exists() else {}
            info["title"] = title
            meta.write_text(json.dumps(info))
            if song["status"] == "done" and (song_dir / "measures.json").exists():
                _, _, _, edited = _load(song_dir)        # o título também vai no MusicXML
                _rebuild_score(song_dir, edited)
        except (OSError, ValueError, KeyError):
            pass
    return _song(song_id)


@app.delete("/api/songs/{song_id}")
def delete_song(song_id: str) -> dict:
    song = _song(song_id)
    if song["status"] == "running":
        raise HTTPException(409, "está em processamento; aguarde terminar")
    db.delete_song(song_id)
    shutil.rmtree(DATA_DIR / song_id, ignore_errors=True)
    src = Path(song["source"])
    try:
        if src.is_file() and UPLOADS in src.resolve().parents:
            src.unlink()
    except OSError:
        pass
    return {"deleted": song_id}


@app.post("/api/songs/{song_id}/retry")
def retry(song_id: str) -> dict:
    song = _song(song_id)
    if song["status"] == "running":
        raise HTTPException(409, "já está em processamento")
    db.update(song_id, status="queued", message="Na fila", error=None, stage=None)
    return db.get(song_id)


# ---------- arquivos ----------
def _download_name(song_id: str, suffix: str, ext: str) -> str:
    """Nome de arquivo amigável: '<título> - <sufixo>.<ext>' (sem caracteres que quebram sistemas de arquivos)."""
    title = (_song(song_id).get("title") or song_id)
    title = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title).strip()[:80] or song_id
    return f"{title} - {suffix}.{ext}"


@app.get("/api/songs/{song_id}/score.musicxml")
def score(song_id: str, download: bool = False) -> FileResponse:
    f = _ready(song_id) / "score.musicxml"
    if download:
        return FileResponse(f, media_type="application/vnd.recordare.musicxml+xml",
                            filename=_download_name(song_id, "partitura", "musicxml"))
    return FileResponse(f, media_type="application/xml")


TRACKS = {"mix": ("audio.wav", "audio.mp3"), "drums": ("drums.wav", "drums.mp3"), "no_drums": ("no_drums.wav", "no_drums.mp3")}


def _ensure_track(song_dir: Path, track: str) -> Path:
    """mp3 da faixa pedida; gera sob demanda (músicas processadas antes desta função)."""
    if track not in TRACKS:
        raise HTTPException(404, "faixa inexistente")
    wav_name, mp3_name = TRACKS[track]
    mp3 = song_dir / mp3_name
    if mp3.exists():
        return mp3
    if track == "no_drums":
        ensure_no_drums(song_dir)
    wav = song_dir / wav_name
    if not wav.exists():
        raise HTTPException(404, "faixa indisponível")
    tmp = mp3.with_suffix(".tmp.mp3")
    subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-vn", "-b:a", "128k", str(tmp)], check=True, capture_output=True)
    tmp.replace(mp3)
    return mp3


@app.get("/api/songs/{song_id}/track/{track}.mp3")
def track_audio(song_id: str, track: str, download: bool = False) -> FileResponse:
    f = _ensure_track(_ready(song_id), track)
    if download:
        label = {"mix": "original", "drums": "so a bateria", "no_drums": "sem a bateria"}[track]
        return FileResponse(f, media_type="audio/mpeg", filename=_download_name(song_id, label, "mp3"))
    return FileResponse(f, media_type="audio/mpeg")


@app.get("/api/songs/{song_id}/audio.mp3")
def audio(song_id: str) -> FileResponse:
    return FileResponse(_ensure_track(_ready(song_id), "mix"), media_type="audio/mpeg")


@app.get("/api/songs/{song_id}/timing")
def timing(song_id: str) -> dict:
    """Beats (s) e início de cada compasso, para o cursor acompanhar o áudio."""
    song_dir = _ready(song_id)
    beats = json.loads((song_dir / "beats.json").read_text())["beats"]
    bars = json.loads((song_dir / "bars.json").read_text())
    return {"beats": beats, "bars": bars}


# ---------- partitura em PDF ----------
MAX_PDF = 50 * 1024 * 1024


def _sheet_dir(song_dir: Path) -> Path:
    return song_dir / "sheet"


def _sheet_state(song_dir: Path) -> dict:
    sdir = _sheet_dir(song_dir)
    layout_file = sdir / "layout.json"
    if not layout_file.exists():
        return {"exists": False}
    layout = json.loads(layout_file.read_text())
    measures = sheetlib.measures_from_layout(layout)
    sync = sheetlib.load_json(sdir / "sync.json", {"order_text": None, "anchors": []})
    suggested = sheetlib.suggest_order(measures) if measures else []
    order, order_error = suggested, None
    if sync.get("order_text"):
        try:
            order = sheetlib.parse_order(sync["order_text"], len(measures))
        except ValueError as e:
            order_error = str(e)
    bars = json.loads((song_dir / "bars.json").read_text())
    beats = json.loads((song_dir / "beats.json").read_text())["beats"]
    times = sheetlib.position_times(len(order), sync.get("anchors", []), bars, beats[-1]) if order else []
    return {
        "exists": True,
        "pages": [{"page": p["page"], "width": p["width"], "height": p["height"], "staves": [
            {"index": i, "y_top": s["y_top"], "y_bottom": s["y_bottom"], "space": s["space"],
             "x_left": s["x_left"], "x_right": s["x_right"], "ignore": s.get("ignore", False),
             "bars": s["bars"]} for i, s in enumerate(p["staves"])]} for p in layout["pages"]],
        "measures": measures,
        "order": order,
        "order_text": sheetlib.order_to_text(order),
        "order_error": order_error,
        "suggested_text": sheetlib.order_to_text(suggested),
        "anchors": sync.get("anchors", []),
        "times": [round(t, 3) for t in times],
        "audio_end": beats[-1],
    }


@app.post("/api/songs/{song_id}/sheet")
async def upload_sheet(song_id: str, file: UploadFile = File(...)) -> dict:
    song_dir = _ready(song_id)
    data = await file.read(MAX_PDF + 1)
    if len(data) > MAX_PDF:
        raise HTTPException(413, "PDF grande demais (máx. 50 MB)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "o arquivo não é um PDF")
    sdir = _sheet_dir(song_dir)
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    pdf = sdir / "source.pdf"
    pdf.write_bytes(data)
    try:
        pages = sheetlib.render_pdf(pdf, sdir)
    except Exception as e:  # noqa: BLE001 - PDF inválido/protegido/páginas demais
        shutil.rmtree(sdir, ignore_errors=True)
        raise HTTPException(400, f"não consegui abrir o PDF: {e}") from e
    imgs = [cv2.imread(str(sdir / f"page-{p['page']}.png"), cv2.IMREAD_GRAYSCALE) for p in pages]
    layout = sheetlib.detect_layout(imgs)
    (sdir / "layout.json").write_text(json.dumps(layout))
    state = _sheet_state(song_dir)
    state["detected_staves"] = sum(len(p["staves"]) for p in layout["pages"])
    return state


@app.get("/api/songs/{song_id}/sheet")
def get_sheet(song_id: str) -> dict:
    return _sheet_state(_ready(song_id))


@app.get("/api/songs/{song_id}/sheet/source.pdf")
def sheet_source(song_id: str) -> FileResponse:
    """O PDF exatamente como foi enviado."""
    f = _sheet_dir(_ready(song_id)) / "source.pdf"
    if not f.exists():
        raise HTTPException(404, "nenhuma partitura enviada")
    return FileResponse(f, media_type="application/pdf", filename=_download_name(song_id, "partitura enviada", "pdf"))


@app.get("/api/songs/{song_id}/sheet/page/{page}.png")
def sheet_page(song_id: str, page: int) -> FileResponse:
    f = _sheet_dir(_ready(song_id)) / f"page-{int(page)}.png"
    if not f.exists():
        raise HTTPException(404, "página não encontrada")
    return FileResponse(f, media_type="image/png")


class LayoutStaff(BaseModel):
    index: int
    ignore: bool = False
    bars: list[dict]  # [{"x": float, "kind": str}]


class LayoutBody(BaseModel):
    pages: list[dict]  # [{"page": n, "staves": [LayoutStaff...]}]


@app.put("/api/songs/{song_id}/sheet/layout")
def put_layout(song_id: str, body: LayoutBody) -> dict:
    song_dir = _ready(song_id)
    lf = _sheet_dir(song_dir) / "layout.json"
    if not lf.exists():
        raise HTTPException(404, "nenhuma partitura enviada")
    layout = json.loads(lf.read_text())
    by_page = {p["page"]: p for p in layout["pages"]}
    for pg in body.pages:
        target = by_page.get(pg.get("page"))
        if not target:
            continue
        for st in pg.get("staves", []):
            m = LayoutStaff(**st)
            if 0 <= m.index < len(target["staves"]):
                tgt = target["staves"][m.index]
                tgt["ignore"] = m.ignore
                tgt["bars"] = [
                    {"x": float(b["x"]), "kind": b.get("kind", "bar"), "thick": bool(b.get("thick", False))}
                    for b in m.bars
                ]
    lf.write_text(json.dumps(layout))
    return _sheet_state(song_dir)


class SyncBody(BaseModel):
    order_text: str | None = None
    anchors: list[dict] = []   # [{"pos": int, "time": float}]


@app.put("/api/songs/{song_id}/sheet/sync")
def put_sync(song_id: str, body: SyncBody) -> dict:
    song_dir = _ready(song_id)
    sdir = _sheet_dir(song_dir)
    if not (sdir / "layout.json").exists():
        raise HTTPException(404, "nenhuma partitura enviada")
    measures = sheetlib.measures_from_layout(json.loads((sdir / "layout.json").read_text()))
    if body.order_text and body.order_text.strip():
        try:
            sheetlib.parse_order(body.order_text, len(measures))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    anchors = []
    for a in body.anchors:
        try:
            anchors.append({"pos": int(a["pos"]), "time": round(float(a["time"]), 3)})
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, "âncora inválida") from e
    (sdir / "sync.json").write_text(json.dumps({"order_text": (body.order_text or "").strip() or None, "anchors": anchors}))
    return _sheet_state(song_dir)


@app.delete("/api/songs/{song_id}/sheet")
def delete_sheet(song_id: str) -> dict:
    shutil.rmtree(_sheet_dir(_ready(song_id)), ignore_errors=True)
    return {"exists": False}


@app.get("/songs/{song_id}/sheet", include_in_schema=False)
def sheet_html(song_id: str) -> HTMLResponse:
    _song(song_id)
    return _page("sheet.html")


# ---------- padrões ----------
def _load(song_dir: Path):
    measures = load_measures(song_dir)
    clusters = cluster_bars(measures)
    edits_file = song_dir / "edits.json"
    rows = json.loads(edits_file.read_text()) if edits_file.exists() else []
    edited = apply_edits(measures, clusters, edits_from_json(rows))
    return measures, clusters, rows, edited


def _rebuild_score(song_dir: Path, edited: list) -> None:
    stats = json.loads((song_dir / "stats.json").read_text())
    title = json.loads((song_dir / "meta.json").read_text()).get("title") or song_dir.name
    to_musicxml(edited, stats["bpb"], stats["bpm_median"], song_dir / "score.musicxml", title=title)


def _cluster_json(c) -> dict:
    n = c.beats * c.subdiv
    return {
        "id": c.id,
        "bars": len(c.bars),
        "ranges": _ranges(c.bars),
        "beats": c.beats,
        "subdiv": c.subdiv,
        "slots": n,
        "rows": [
            {
                "pitch": p,
                "name": PIECES[p][0],
                "freq": [round(c.freq.get((p, s), 0.0), 2) for s in range(n)],
                "on": [(p, s) in c.pattern for s in range(n)],
            }
            for p in ORDER
        ],
    }


@app.get("/api/songs/{song_id}/patterns")
def patterns(song_id: str, min_bars: int = 2, limit: int = 12) -> dict:
    song_dir = _ready(song_id)
    measures, clusters, rows, edited = _load(song_dir)
    state = with_state(clusters, edited)
    shown = [c for c in state if len(c.bars) >= min_bars][:limit]
    return {
        "total_bars": len(measures),
        "covered_bars": sum(len(c.bars) for c in shown),
        "edits": len(rows),
        "clusters": [_cluster_json(c) for c in shown],
    }


class PatternBody(BaseModel):
    pattern: list[tuple[int, int]]  # [(pitch, slot)] desejado para o grupo
    purge_rare: bool = True         # remove também golpes que aparecem em <50% dos compassos
    enforce: bool = False           # impõe o padrão das linhas alteradas a TODOS os compassos do grupo
    enforce_pitches: list[int] = []  # peças a impor mesmo sem mudança no padrão (ex.: [35] = bumbo)


@app.post("/api/songs/{song_id}/patterns/{cluster_id}")
def set_pattern(song_id: str, cluster_id: int, body: PatternBody) -> dict:
    song_dir = _ready(song_id)
    measures, clusters, rows, edited = _load(song_dir)
    state = {c.id: c for c in with_state(clusters, edited)}
    if cluster_id not in state:
        raise HTTPException(404, "grupo não encontrado")
    c = state[cluster_id]
    desired = {(int(p), int(s)) for p, s in body.pattern}
    valid = {(p, s) for p in ORDER for s in range(c.beats * c.subdiv)}
    if not desired <= valid:
        raise HTTPException(400, "golpe fora da grade do grupo")
    remove = c.pattern - desired
    add = desired - c.pattern
    if body.purge_rare:
        # só nas peças que o usuário mexeu: não apaga variações das demais
        touched = {p for p, _ in remove | add}
        remove |= {e for e, f in c.freq.items() if f < 0.5 and e not in desired and e[0] in touched}
    if body.enforce or body.enforce_pitches:
        # "o padrão é este": nas peças alteradas (ou pedidas), todos os compassos ficam
        # exatamente com os golpes desejados (sem isso só entram os que ainda não estavam
        # no padrão do grupo)
        touched = {p for p, _ in remove | add} | set(body.enforce_pitches)
        n_slots = c.beats * c.subdiv
        for p in touched:
            remove |= {(p, s) for s in range(n_slots) if (p, s) not in desired}
            add |= {e for e in desired if e[0] == p}
    if remove or add:
        rows.append({
            "cluster": cluster_id,
            "remove": sorted([p, s] for p, s in remove),
            "add": sorted([p, s] for p, s in add),
        })
        (song_dir / "edits.json").write_text(json.dumps(rows))
        edited = apply_edits(measures, clusters, edits_from_json(rows))
        _rebuild_score(song_dir, edited)
    new = {x.id: x for x in with_state(clusters, edited)}[cluster_id]
    return {"edits": len(rows), "cluster": _cluster_json(new), "changed": bool(remove or add)}


@app.post("/api/songs/{song_id}/edits/reset")
def reset_edits(song_id: str) -> dict:
    song_dir = _ready(song_id)
    (song_dir / "edits.json").unlink(missing_ok=True)
    measures, clusters, _, edited = _load(song_dir)
    _rebuild_score(song_dir, edited)
    return {"edits": 0}


def _bootstrap_admin() -> None:
    """Cria a conta do dono (se faltar) e passa para ela as músicas anteriores às contas."""
    name = authlib.normalize_username(os.environ.get("KICKSNARE_ADMIN_USER", ""))
    pw = os.environ.get("KICKSNARE_ADMIN_PASSWORD", "")
    if not name or not pw:
        return
    user = db.get_user_by_name(name)
    uid = user["id"] if user else db.create_user(name, authlib.hash_password(pw), is_admin=True)
    if uid is not None:
        db.set_admin(uid, True)
        db.set_user_status(uid, "approved")
        db.assign_orphans(uid)


_bootstrap_admin()
