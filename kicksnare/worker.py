"""Worker: pega uma música da fila por vez e roda ingest -> demucs -> transcrição.

Uso: python -m kicksnare.worker
"""
import json
import subprocess
import sys
import time
import traceback

from . import db
from .ingest import DATA_DIR, ingest
from .separate import encode_mp3, ensure_no_drums, separate_drums


def _set(song_id: str, stage: str, message: str) -> None:
    """Muda de etapa: guarda quanto a anterior demorou (alimenta a previsão de tempo)."""
    row = db.get(song_id) or {}
    timings = dict(row.get("timings") or {})
    prev, since = row.get("stage"), row.get("stage_started")
    if prev and since:
        timings[prev] = round(time.time() - since, 1)
    db.update(song_id, stage=stage, message=message, stage_started=time.time(), progress=0.0, timings=timings)


def _duration(path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def process(song: dict) -> None:
    sid, source = song["id"], song["source"]

    _set(sid, "ingest", "Baixando áudio" if source.startswith("http") else "Preparando áudio")
    _, audio = ingest(source, song_id=sid)
    meta = DATA_DIR / sid / "meta.json"
    if meta.exists() and not (db.get(sid) or {}).get("title"):   # nome editado pelo usuário prevalece
        db.update(sid, title=json.loads(meta.read_text()).get("title"))
    db.update(sid, duration=_duration(audio))

    drums = audio.parent / "drums.wav"
    if not drums.exists():
        _set(sid, "separate", "Isolando a bateria (alguns minutos)")
        _, metrics = separate_drums(audio, on_progress=lambda p: db.update(sid, progress=round(p, 3)))
        print(f"[{sid}] demucs {metrics}", flush=True)

    _set(sid, "transcribe", "Transcrevendo")
    result = subprocess.run(
        [sys.executable, "-m", "kicksnare.transcribe_run", sid], capture_output=True, text=True
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"transcrição falhou:\n{tail}")
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")][-1]
    stats = json.loads(line[len("RESULT "):])

    _set(sid, "finalize", "Gerando áudio para o player")
    ensure_no_drums(audio.parent)
    for wav_name, mp3_name in (("audio.wav", "audio.mp3"), ("drums.wav", "drums-gate.mp3"), ("no_drums.wav", "no_drums.mp3")):
        mp3 = audio.parent / mp3_name
        if not mp3.exists():
            encode_mp3(audio.parent / wav_name, mp3, drums=wav_name == "drums.wav")
    row = db.get(sid) or {}
    timings = dict(row.get("timings") or {})
    if row.get("stage_started"):
        timings["finalize"] = round(time.time() - row["stage_started"], 1)
    db.update(sid, status="done", stage=None, message="Pronto", stats=stats, finished=time.time(), error=None,
              timings=timings, progress=1.0)


def main() -> None:
    n = db.requeue_interrupted()
    if n:
        print(f"reenfileirados {n} jobs interrompidos", flush=True)
    print("worker pronto", flush=True)
    while True:
        song = db.claim_next()
        if not song:
            time.sleep(2)
            continue
        print(f"[{song['id']}] iniciando", flush=True)
        try:
            process(song)
            print(f"[{song['id']}] pronto", flush=True)
        except Exception as e:  # noqa: BLE001 - qualquer falha vira erro visível na interface
            traceback.print_exc()
            db.update(song["id"], status="error", message="Falhou", error=str(e)[:2000], finished=time.time())


if __name__ == "__main__":
    main()
