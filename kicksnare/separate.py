"""Isola o stem de bateria com o Demucs (CPU), com parâmetros para pouca RAM."""
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _children_peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # macOS reporta bytes; Linux reporta KB
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def separate_drums(audio: Path, model: str = "htdemucs", segment: int = 7, on_progress=None) -> tuple[Path, dict]:
    """Gera <dir do audio>/drums.wav (e no_drums.wav). `on_progress(0..1)` recebe o avanço do Demucs."""
    import os
    import re

    out = audio.parent / "drums.wav"
    if out.exists():
        return out, {"cached": True}

    pct = re.compile(r"(\d{1,3})%\|")
    with tempfile.TemporaryDirectory(prefix="kicksnare-demucs-") as tmp:
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model, "--two-stems", "drums",
            "--segment", str(segment), "-j", "1", "-d", "cpu",
            "-o", tmp, str(audio),
        ]
        started = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tail, last = b"", -1
        while True:
            chunk = os.read(proc.stderr.fileno(), 4096)    # o tqdm atualiza com \r, não com \n
            if not chunk:
                break
            tail = (tail + chunk)[-4096:]
            m = pct.findall(tail.decode("utf-8", "ignore"))
            if m and int(m[-1]) != last:
                last = int(m[-1])
                if on_progress:
                    on_progress(min(1.0, last / 100))
        proc.wait()
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            msg = "\n".join(tail.decode("utf-8", "ignore").replace("\r", "\n").strip().splitlines()[-8:])
            raise RuntimeError(f"demucs falhou:\n{msg}")
        produced = Path(tmp) / model / audio.stem / "drums.wav"
        shutil.move(str(produced), out)
        rest = Path(tmp) / model / audio.stem / "no_drums.wav"
        if rest.exists():
            shutil.move(str(rest), audio.parent / "no_drums.wav")

    return out, {"cached": False, "seconds": round(elapsed, 1), "peak_ram_mb": round(_children_peak_mb())}


def ensure_no_drums(song_dir: Path) -> Path:
    """Música sem a bateria. Se não foi guardada pelo Demucs, usa mix - bateria (as duas
    faixas do `--two-stems` somam o mix original, então a subtração é exata)."""
    import numpy as np
    import soundfile as sf

    out = song_dir / "no_drums.wav"
    if out.exists():
        return out
    mix, sr = sf.read(str(song_dir / "audio.wav"), dtype="float32")
    drums, sr2 = sf.read(str(song_dir / "drums.wav"), dtype="float32")
    if sr != sr2 or mix.shape != drums.shape:
        n = min(len(mix), len(drums))
        mix, drums = mix[:n], drums[:n]
    sf.write(str(out), np.clip(mix - drums, -1.0, 1.0), sr, subtype="PCM_16")
    return out



# Bateria solo do player. O stem do Demucs traz vazamento dos outros instrumentos entre os golpes
# (medido num rock: fundo a só 7-8 dB dos golpes, 98% da energia abaixo de 2 kHz, ou seja baixo e
# guitarra, não cauda de prato). Um gate por volume não separa 7 dB; então o "gate" usa os golpes
# já transcritos (events.json): aberto em volta de cada golpe, pelo tempo que a peça soa, e o resto
# é atenuado PLAY_FLOOR_DB (leve: ghost notes que a transcrição perdeu continuam audíveis). Depois,
# compressão leve e ganho de compensação. Só o mp3 é tratado: drums.wav fica exato (a transcrição
# e o "sem bateria" = mix - bateria dependem dele).
PLAY_FLOOR_DB = -12.0
PLAY_PRE = 0.03        # s antes do golpe (o ADTOF marca ~20 ms cedo)
PLAY_HOLD = {"kick": 0.25, "snare": 0.30, "hihat": 0.20, "cymbal": 1.5}  # tons: 0,5 s (padrão)
PLAY_RELEASE = 0.15    # s, volta suave ao nível do fundo
PLAY_REF_DB = -14.0    # nível (p95 do RMS em 50 ms) depois da normalização
PLAY_COMP = "acompressor=threshold=0.17783:ratio=2:attack=15:release=150:knee=4:makeup=1.41254"  # -15 dB, 2:1, +3 dB


def drums_playback_wav(song_dir: Path, out: Path) -> Path:
    """drums.wav com o fundo entre golpes atenuado e o nível normalizado (para o mp3 do player)."""
    import json

    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(song_dir / "drums.wav"), dtype="float32", always_2d=True)
    events = song_dir / "events.json"
    if events.exists():
        floor = 10 ** (PLAY_FLOOR_DB / 20)
        open_ = np.zeros(len(x), dtype=bool)
        for e in json.loads(events.read_text()):
            a = max(0, int((e["t"] - PLAY_PRE) * sr))
            open_[a:int((e["t"] + PLAY_HOLD.get(e["piece"], 0.5)) * sr)] = True
        gain = np.where(open_, 1.0, floor).astype(np.float32)
        # suaviza só as descidas (release); as subidas ficam instantâneas para não amaciar o ataque
        k = int(PLAY_RELEASE * sr)
        if k > 1:
            ramp = np.linspace(1.0, floor, k, dtype=np.float32)
            for c in np.flatnonzero(open_[:-1] & ~open_[1:]) + 1:
                seg = gain[c:c + k]
                np.maximum(seg, ramp[:len(seg)], out=seg)
        x = x * gain[:, None]
    mono = x.mean(axis=1)
    w = int(0.05 * sr)
    n = len(mono) // w
    db = 20 * np.log10(np.sqrt((mono[:n * w].reshape(n, w) ** 2).mean(axis=1)) + 1e-9)
    db = db[db > -90]
    if len(db):
        x = x * 10 ** ((PLAY_REF_DB - float(np.percentile(db, 95))) / 20)
    sf.write(str(out), x, sr, subtype="FLOAT")  # float: o ganho pode passar de 0 dBFS antes do limitador
    return out


def encode_mp3(wav: Path, mp3: Path, drums: bool = False) -> Path:
    """wav -> mp3 128k (via arquivo temporário). `drums=True` aplica o tratamento da bateria solo."""
    tmp = mp3.with_suffix(".tmp.mp3")
    src, af = wav, []
    if drums:
        src = drums_playback_wav(wav.parent, mp3.with_suffix(".tmp.wav"))
        af = ["-af", PLAY_COMP + ",alimiter=limit=0.95:level=false"]
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vn", *af, "-b:a", "128k", str(tmp)],
                       check=True, capture_output=True)
    finally:
        if src != wav:
            src.unlink(missing_ok=True)
    tmp.replace(mp3)
    return mp3
