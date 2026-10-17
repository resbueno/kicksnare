"""Obtém o áudio de uma música (URL do YouTube ou arquivo local) como WAV 44.1 kHz.

A chamada ao yt-dlp segue a do projeto ../downloader, mas o arquivo é nomeado
pelo ID do vídeo e a saída é WAV (o Demucs separa melhor a partir de áudio sem perdas).
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(os.environ.get("KICKSNARE_DATA") or Path(__file__).resolve().parent.parent / "data")

# yt-dlp novo exige Python >= 3.10; em dev (Python antigo) aponte KICKSNARE_YTDLP
# para outro interpretador, ex.: ".venv-ytdlp/bin/python -m yt_dlp"
YTDLP = shlex.split(os.environ.get("KICKSNARE_YTDLP", "")) or [sys.executable, "-m", "yt_dlp"]


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg não encontrado no PATH (apt install ffmpeg / brew install ffmpeg)")
    return path


def _video_meta(url: str) -> tuple[str, str, dict]:
    """(id, título, extras) do vídeo via yt-dlp, sem baixar. Extras: canal e, nos vídeos do YouTube
    Music, artista e faixa (servem para identificar a música; "NA" quando faltam)."""
    result = subprocess.run(
        [*YTDLP, "--no-playlist", "--print", "%(id)s\n%(title)s\n%(uploader)s\n%(artist)s\n%(track)s", url],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        raise RuntimeError(err[-1] if err else "yt-dlp falhou ao obter o ID")
    lines = result.stdout.strip().splitlines()
    extra = dict(zip(("uploader", "artist", "track"), lines[2:5]))
    return lines[0], (lines[1] if len(lines) > 1 else lines[0]), extra


def url_duration(url: str, timeout: int = 25) -> float | None:
    """Duração do vídeo em segundos via yt-dlp, sem baixar (para a previsão antes de enviar). None se
    o yt-dlp não souber ou demorar."""
    try:
        result = subprocess.run([*YTDLP, "--no-playlist", "--print", "%(duration)s", url],
                                capture_output=True, text=True, timeout=timeout)
        return float(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def file_song_id(path: Path) -> str:
    return "file-" + hashlib.sha1(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()[:10]


def ingest(source: str, song_id: str | None = None, data_dir: Path | None = None) -> tuple[str, Path]:
    """Retorna (song_id, caminho do audio.wav). Usa cache em <data_dir>/<song_id>/ (padrão: DATA_DIR;
    o GrooveBass passa a pasta dele)."""
    title, extra = None, {}
    if is_url(source):
        vid, title, extra = _video_meta(source)
        song_id = song_id or vid
    else:
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        song_id = song_id or file_song_id(src)
        title = src.stem

    song_dir = (data_dir or DATA_DIR) / song_id
    wav = song_dir / "audio.wav"
    song_dir.mkdir(parents=True, exist_ok=True)
    meta = song_dir / "meta.json"
    if not meta.exists():
        # source_title: título original (o "title" muda quando o usuário renomeia)
        meta.write_text(json.dumps({"title": title, "source_title": title,
                                    "source": source if is_url(source) else "upload", **extra}))
    if wav.exists():
        return song_id, wav

    if is_url(source):
        raw_tpl = str(song_dir / "source.%(ext)s")
        cmd = [
            *YTDLP, "--no-playlist",
            "-f", "bestaudio/best", "--output", raw_tpl, "--print", "after_move:filepath",
            "--ffmpeg-location", _ffmpeg(), source,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()
            raise RuntimeError(err[-1] if err else "yt-dlp falhou")
        raw = Path(result.stdout.strip().splitlines()[-1])
    else:
        raw = src

    subprocess.run(
        [_ffmpeg(), "-y", "-i", str(raw), "-vn", "-ar", "44100", "-ac", "2", str(wav)],
        check=True, capture_output=True,
    )
    return song_id, wav
