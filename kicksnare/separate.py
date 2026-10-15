"""Isola o stem de bateria com o Demucs (CPU), com parâmetros para pouca RAM."""
import json
import os
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


# Modelo do Demucs. htdemucs_ft (4 redes, uma por instrumento) vaza menos baixo/guitarra na bateria
# que o htdemucs (medido num rock, trecho de 60 s: contraste golpe/fundo em 80-1500 Hz de 19,6 para
# 21,0 dB), mas leva ~4,4x mais (75 s -> 330 s de CPU por minuto de áudio). --shifts 2 dobrou o tempo
# sem ganho mensurável no mesmo trecho; fica em 1.
MODEL = os.environ.get("KICKSNARE_DEMUCS_MODEL", "htdemucs_ft")
SHIFTS = int(os.environ.get("KICKSNARE_DEMUCS_SHIFTS", "1"))
LEGACY = {"model": "htdemucs", "shifts": 1}   # músicas separadas antes de existir separation.json
BAG_SIZE = {"htdemucs_ft": 4, "mdx": 4, "mdx_extra": 4, "mdx_q": 4, "mdx_extra_q": 4}  # redes por modelo
# derivados de drums.wav: ficam velhos quando a separação é refeita
DERIVED = ("no_drums.wav", "no_drums.mp3", "drums-gate.mp3", "drums.mp3")


def separation_config() -> dict:
    return {"model": MODEL, "shifts": SHIFTS}


def needs_separation(song_dir: Path) -> bool:
    """Falta drums.wav, ou ele foi gerado com outro modelo/shifts (ex.: música antiga reprocessada)."""
    if not (song_dir / "drums.wav").exists():
        return True
    info = song_dir / "separation.json"
    done = json.loads(info.read_text()) if info.exists() else LEGACY
    return done != separation_config()


def separate_drums(audio: Path, segment: int = 7, on_progress=None) -> tuple[Path, dict]:
    """Gera <dir do audio>/drums.wav (e no_drums.wav). `on_progress(0..1)` recebe o avanço do Demucs."""
    import re

    out = audio.parent / "drums.wav"
    if not needs_separation(audio.parent):
        return out, {"cached": True}
    model, cfg = MODEL, separation_config()

    # Uma barra do tqdm por rede e por shift, cada uma de 0 a 100%. "63.0/63.0 [" casa só as barras
    # do áudio (em segundos); as do download dos pesos ("80.2M/80.2M") ficam de fora.
    pct = re.compile(r"(\d{1,3})%\|[^|]*\|\s*[\d.]+/[\d.]+ \[")
    passes = BAG_SIZE.get(model, 1) * SHIFTS
    with tempfile.TemporaryDirectory(prefix="kicksnare-demucs-") as tmp:
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model, "--two-stems", "drums", "--shifts", str(SHIFTS),
            "--segment", str(segment), "-j", "1", "-d", "cpu",
            "-o", tmp, str(audio),
        ]
        started = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tail, last, done = b"", -1, 0
        while True:
            chunk = os.read(proc.stderr.fileno(), 4096)    # o tqdm atualiza com \r, não com \n
            if not chunk:
                break
            tail = (tail + chunk)[-4096:]
            m = pct.findall(tail.decode("utf-8", "ignore"))
            if m and int(m[-1]) != last:
                if int(m[-1]) < last:   # começou a barra da próxima rede/shift
                    done += 1
                last = int(m[-1])
                if on_progress:
                    on_progress(min(1.0, (min(done, passes - 1) + last / 100) / passes))
        proc.wait()
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            msg = "\n".join(tail.decode("utf-8", "ignore").replace("\r", "\n").strip().splitlines()[-8:])
            raise RuntimeError(f"demucs falhou:\n{msg}")
        for name in DERIVED:
            (audio.parent / name).unlink(missing_ok=True)
        produced = Path(tmp) / model / audio.stem / "drums.wav"
        shutil.move(str(produced), out)
        rest = Path(tmp) / model / audio.stem / "no_drums.wav"
        if rest.exists():
            shutil.move(str(rest), audio.parent / "no_drums.wav")

    (audio.parent / "separation.json").write_text(json.dumps(cfg))
    return out, {"cached": False, **cfg, "seconds": round(elapsed, 1), "peak_ram_mb": round(_children_peak_mb())}


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
# o volume sobe até o pico ficar em PLAY_PEAK_DB, sem compressor nem limitador (a compressão deixava
# o som ruim). Só o mp3 é tratado: drums.wav fica exato (a transcrição e o "sem bateria" = mix -
# bateria dependem dele).
PLAY_FLOOR_DB = -12.0
PLAY_PRE = 0.03        # s antes do golpe (o ADTOF marca ~20 ms cedo)
PLAY_HOLD = {"kick": 0.25, "snare": 0.30, "hihat": 0.20, "cymbal": 1.5}  # tons: 0,5 s (padrão)
PLAY_RELEASE = 0.15    # s, volta suave ao nível do fundo
PLAY_PEAK_DB = -1.0    # pico depois da normalização (folga para o mp3 não clipar)


def _gate_intervals(song_dir: Path, sr: int) -> list[tuple[int, int]]:
    """Trechos (em amostras) em que o gate fica aberto, já fundidos e em ordem."""
    import json

    events = song_dir / "events.json"
    if not events.exists():
        return []
    spans = sorted(
        (max(0, int((e["t"] - PLAY_PRE) * sr)), int((e["t"] + PLAY_HOLD.get(e["piece"], 0.5)) * sr))
        for e in json.loads(events.read_text())
    )
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def drums_playback_wav(song_dir: Path, out: Path, block: int = 1 << 20) -> Path:
    """drums.wav com o fundo entre golpes atenuado e o pico normalizado (para o mp3 do player).

    Processa em blocos, em duas passadas (medir o pico, depois aplicar), para caber no container
    web (512 MB): carregar a faixa inteira e fazer cópias passava de 400 MB numa música de 4 min.
    """
    import numpy as np
    import soundfile as sf

    src = song_dir / "drums.wav"
    sr = sf.info(str(src)).samplerate
    spans = _gate_intervals(song_dir, sr)
    floor = 10 ** (PLAY_FLOOR_DB / 20)
    k = max(1, int(PLAY_RELEASE * sr))
    ramp = np.linspace(1.0, floor, k, dtype=np.float32)
    starts = np.array([a for a, _ in spans], dtype=np.int64)

    def gain(a: int, b: int) -> np.ndarray:
        if not spans:
            return np.ones(b - a, dtype=np.float32)
        g = np.full(b - a, floor, dtype=np.float32)
        # trechos que tocam [a - k, b): o próprio trecho aberto e a descida suave depois dele
        lo = max(0, int(np.searchsorted(starts, a - k, side="right")) - 1)
        for s0, e0 in spans[lo:]:
            if s0 >= b:
                break
            g[max(s0, a) - a:max(min(e0, b), a) - a] = 1.0
            r0, r1 = max(e0, a), min(e0 + k, b)   # release: as subidas ficam instantâneas
            if r1 > r0:
                np.maximum(g[r0 - a:r1 - a], ramp[r0 - e0:r1 - e0], out=g[r0 - a:r1 - a])
        return g

    peak = 0.0
    pos = 0
    for x in sf.blocks(str(src), blocksize=block, dtype="float32", always_2d=True):
        peak = max(peak, float(np.abs(x * gain(pos, pos + len(x))[:, None]).max(initial=0.0)))
        pos += len(x)
    norm = 10 ** (PLAY_PEAK_DB / 20) / peak if peak > 0 else 1.0

    pos = 0
    with sf.SoundFile(str(src)) as f, sf.SoundFile(str(out), "w", sr, f.channels, subtype="FLOAT") as o:
        for x in f.blocks(blocksize=block, dtype="float32", always_2d=True):
            o.write(x * (gain(pos, pos + len(x)) * norm)[:, None])
            pos += len(x)
    return out


def encode_mp3(wav: Path, mp3: Path, drums: bool = False) -> Path:
    """wav -> mp3 128k (via arquivo temporário). `drums=True` aplica o tratamento da bateria solo."""
    tmp = mp3.with_suffix(".tmp.mp3")
    src = drums_playback_wav(wav.parent, mp3.with_suffix(".tmp.wav")) if drums else wav
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-b:a", "128k", str(tmp)],
                       check=True, capture_output=True)
    finally:
        if src != wav:
            src.unlink(missing_ok=True)
    tmp.replace(mp3)
    return mp3
