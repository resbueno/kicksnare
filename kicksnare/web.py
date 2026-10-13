"""API e páginas do Kicksnare.

Rodar: uvicorn kicksnare.web:app --host 127.0.0.1 --port 8000
Defina KICKSNARE_PASSWORD (e opcionalmente KICKSNARE_USER) para exigir login.
"""
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

import shutil
import subprocess

import cv2

from . import db, sheet as sheetlib
from .ingest import DATA_DIR, file_song_id
from .separate import ensure_no_drums
from .patterns import (
    Edit, apply_edits, cluster_bars, edits_from_json, load_measures, with_state, _ranges,
)
from .transcribe import ORDER, PIECES, to_musicxml

STATIC = Path(__file__).parent / "static"
# Prefixo de URL quando roda atrás de um proxy em subcaminho (ex.: /kicksnare). Vazio = raiz.
BASE = os.environ.get("KICKSNARE_BASE", "").rstrip("/")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC / name).read_text().replace("__BASE__", BASE))
UPLOADS = DATA_DIR / "uploads"
MAX_UPLOAD = 300 * 1024 * 1024
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
YT_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")

security = HTTPBasic(auto_error=False)


def auth(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
    password = os.environ.get("KICKSNARE_PASSWORD")
    if not password:
        return
    user = os.environ.get("KICKSNARE_USER", "kicksnare")
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username.encode(), user.encode())
        and secrets.compare_digest(creds.password.encode(), password.encode())
    )
    if not ok:
        raise HTTPException(401, "Não autorizado", headers={"WWW-Authenticate": "Basic"})


app = FastAPI(title="Kicksnare", dependencies=[Depends(auth)])
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
    if not ID_RE.match(song_id):
        raise HTTPException(404, "música não encontrada")
    song = db.get(song_id)
    if not song:
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


@app.get("/songs/{song_id}", include_in_schema=False)
def song_page(song_id: str) -> HTMLResponse:
    _song(song_id)
    return _page("song.html")


# ---------- fila ----------
class NewSong(BaseModel):
    url: str


@app.get("/api/songs")
def list_songs() -> list[dict]:
    return [_fresh(s) for s in db.list_songs()]


@app.post("/api/songs")
def add_song(body: NewSong) -> dict:
    url = body.url.strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "informe uma URL http(s)")
    m = YT_RE.search(url)
    song_id = m.group(1) if m else "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]
    return db.enqueue(song_id, url)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(file.filename or "audio").name)[:120]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / name
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "arquivo grande demais (máx. 300 MB)")
            out.write(chunk)
    return db.enqueue(file_song_id(dest), str(dest), title=Path(name).stem)


@app.get("/api/songs/{song_id}")
def get_song(song_id: str) -> dict:
    return _song(song_id)


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
