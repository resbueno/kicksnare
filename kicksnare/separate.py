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
