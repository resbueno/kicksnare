"""Etapa 2: drums.wav (+ audio.wav) -> eventos -> grade de compassos -> MusicXML.

- Onsets por peça: ADTOF (5 classes: bumbo, caixa, tom, hi-hat, pratos).
- Beats: beat_this, rodado no mix completo (mais confiável que no stem).
- Compassos: a fase de cada compasso vem do backbeat (caixa nos tempos 2 e 4,
  bumbo no 1), via Viterbi sobre os beats. Os downbeats do beat_this entram só
  como desempate fraco, porque falham em trechos de meio-compasso e em quebras.
- Quantização: grade de semicolcheias por beat, interpolando entre os beats reais,
  então andamentos que variam não desalinham a partitura.
"""
import json
import statistics
import tempfile
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

SUBDIV = 4  # subdivisões por beat na grade reta (semicolcheias); tercinas usam 3

# pitch GM do ADTOF -> (nome, nota exibida na pauta, cabeça "x")
PIECES = {
    35: ("kick", ("F", 4), False),
    38: ("snare", ("C", 5), False),
    47: ("tom", ("A", 4), False),
    42: ("hihat", ("G", 5), True),
    49: ("cymbal", ("A", 5), True),
}
ORDER = [49, 42, 38, 47, 35]  # de cima para baixo na pauta
KICK, SNARE = 35, 38

SWITCH_PENALTY = 6.0   # custo de trocar a fase (~2 compassos de evidência)
DOWNBEAT_BONUS = 0.25  # desempate fraco a favor dos downbeats do beat_this


@dataclass
class Hit:
    time: float
    pitch: int


@dataclass
class Measure:
    beats: int
    slots: dict  # slot (0..beats*SUBDIV-1) -> set(pitches)
    start: float = 0.0  # tempo (s) do início do compasso
    subdiv: int = SUBDIV  # slots por beat (4 = semicolcheias, 3 = tercinas)


# Limiar do bumbo no ADTOF (padrão 0,22). Medido contra a partitura do Batera.com.br
# (rock, 115 compassos): 0,22 -> revocação 53%; 0,10 -> 80%; 0,08 -> 84%; 0,06 -> 88%, com a
# precisão estável em 89%. No pop (bumbo nos 4 tempos) a contagem fica em ~4,0/compasso até
# 0,10 e abaixo de 0,06 aparece 1-3% de ruído fora do tempo. 0,08 é o compromisso.
KICK_THRESHOLD = 0.08

HAT_CANDIDATES = (0.03, 0.06, 0.10, 0.15, 0.22)  # do mais sensível ao padrão do ADTOF
HAT_MAX_RATIO = 1.1  # hats detectados não podem passar 110% dos ataques agudos brutos


def _hf_onsets(wav: Path) -> np.ndarray:
    """Ataques de alta frequência (6-16 kHz) direto do áudio, sem modelo. Serve só de
    referência de quantos golpes de chimbal/prato existem, para calibrar o limiar."""
    import soundfile as sf
    from scipy.signal import find_peaks, stft

    x, sr = sf.read(str(wav))
    x = x.mean(axis=1) if x.ndim > 1 else x
    hop = 256
    f, t, z = stft(x, fs=sr, nperseg=1024, noverlap=1024 - hop)
    s = np.log1p(np.abs(z[(f >= 6000) & (f <= 16000)]).sum(axis=0) * 10)
    flux = np.maximum(0, np.diff(s, prepend=s[0]))
    flux = flux / max(np.percentile(flux, 99.5), 1e-9)
    peaks, _ = find_peaks(flux, height=0.25, distance=max(1, int(0.05 * sr / hop)))
    return t[peaks]


def detect_hits(drums_wav: Path, device: str = "cpu") -> tuple[list[Hit], dict]:
    """ADTOF com o limiar do hi-hat calibrado por música.

    O limiar padrão do ADTOF perde muitos hi-hats (medido: ~40% dos ataques agudos em
    um rock com tercinas contínuas). Roda a rede uma vez e escolhe o limiar mais
    sensível que não detecta mais do que HAT_MAX_RATIO x os ataques agudos brutos.
    """
    from adtof_pytorch import transcribe_to_midi
    from adtof_pytorch.post_processing import FRAME_RNN_THRESHOLDS, LABELS_5, PeakPicker

    pred = transcribe_to_midi(str(drums_wav), "unused.mid", return_activations=True, device=device)
    hat_idx = LABELS_5.index(42)
    ref_count = len(_hf_onsets(drums_wav))

    kick_idx = LABELS_5.index(35)

    def pick(hat_threshold: float) -> dict:
        thr = list(FRAME_RNN_THRESHOLDS)
        thr[hat_idx] = hat_threshold
        thr[kick_idx] = KICK_THRESHOLD
        return PeakPicker(thresholds=thr, fps=100).pick(pred, labels=LABELS_5, label_offset=0)[0]

    chosen, peaks = FRAME_RNN_THRESHOLDS[hat_idx], None
    for th in HAT_CANDIDATES:
        cand = pick(th)
        # chimbal e pratos dividem o mesmo conteúdo agudo, então o teto vale para os dois juntos
        if ref_count and len(cand.get(42, [])) + len(cand.get(49, [])) <= HAT_MAX_RATIO * ref_count:
            chosen, peaks = th, cand
            break
    if peaks is None:
        peaks = pick(chosen)

    hits = [Hit(float(t), int(pitch)) for pitch, times in peaks.items() for t in times]
    info = {"hihat_threshold": chosen, "kick_threshold": KICK_THRESHOLD, "hf_onsets": ref_count,
            "hihat_hits": len(peaks.get(42, [])), "kick_hits": len(peaks.get(35, []))}
    return sorted(hits, key=lambda h: (h.time, h.pitch)), info


def detect_beats(audio_wav: Path, device: str = "cpu") -> tuple[list[float], list[float]]:
    from beat_this.inference import File2Beats

    beats, downbeats = File2Beats(device=device, dbn=False)(str(audio_wav))
    return [float(b) for b in beats], [float(d) for d in downbeats]


def clean_beats(beats: list[float]) -> tuple[list[float], dict]:
    """Regulariza os beats do beat_this (sem DBN): descarta os espúrios e preenche buracos.

    Procura, por programação dinâmica, a cadeia de beats mais regular: cada passo deve
    durar ~1 intervalo local (ou 2-3, com beats faltando, que são interpolados). Beats
    fora da cadeia (intervalos de 0,1-0,3 s) são descartados.
    """
    b = np.asarray(beats, dtype=float)
    n = len(b)
    if n < 8:
        return list(beats), {"beats_removed": 0, "beats_added": 0}
    ibi = np.diff(b)
    ref = float(np.median(ibi[ibi > 0.6 * np.median(ibi)]))
    # andamento local: mediana dos intervalos "plausíveis" ao redor de cada beat
    plausible = (ibi > 0.7 * ref) & (ibi < 1.4 * ref)
    loc = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 10), min(n - 1, i + 10)
        w = ibi[lo:hi][plausible[lo:hi]]
        loc[i] = float(np.median(w)) if len(w) else ref

    NEG = -1e9
    best = np.zeros(n)
    prev = np.full(n, -1, dtype=int)
    steps = np.ones(n, dtype=int)
    for j in range(1, n):
        for i in range(j - 1, max(-1, j - 12), -1):
            d = b[j] - b[i]
            if d > 3.6 * loc[i]:
                break
            k = int(round(d / loc[i]))
            if k < 1 or k > 3:
                continue
            err = abs(d - k * loc[i]) / loc[i]
            if err > 0.3:
                continue
            score = best[i] + (1.0 - (err / 0.3) ** 2) - 0.45 * (k - 1)
            if score > best[j]:
                best[j], prev[j], steps[j] = score, i, k
    j = int(np.argmax(best))
    chain = [j]
    while prev[chain[-1]] >= 0:
        chain.append(int(prev[chain[-1]]))
    chain.reverse()

    out: list[float] = []
    added = 0
    for pos, idx in enumerate(chain):
        if pos > 0:
            k = int(steps[idx])
            for m in range(1, k):
                out.append(float(b[chain[pos - 1]] + (b[idx] - b[chain[pos - 1]]) * m / k))
                added += 1
        out.append(float(b[idx]))
    # extrapola nas pontas, até onde o rastreador tinha beats
    step0, step1 = float(loc[chain[0]]), float(loc[chain[-1]])
    head = []
    t = out[0] - step0
    while t > b[0] - 0.5 * step0:
        head.append(t)
        t -= step0
        added += 1
    out = head[::-1] + out
    t = out[-1] + step1
    while t < b[-1] + 0.5 * step1:
        out.append(t)
        t += step1
        added += 1
    kept = len(chain)
    return out, {"beats_removed": n - kept, "beats_added": added}


def beats_per_bar(beats: list[float], downbeats: list[float]) -> int:
    """Moda da quantidade de beats entre downbeats consecutivos, limitada a 3 ou 4.

    Com downbeats em meio-compasso (2 beats) a moda pode vir errada; 4/4 é o
    padrão seguro para bateria pop/rock, e 3/4 só vale se dominar claramente.
    """
    counts = [
        sum(1 for t in beats if a - 1e-3 <= t < b - 1e-3)
        for a, b in zip(downbeats, downbeats[1:])
    ]
    if not counts:
        return 4
    three = sum(1 for c in counts if c == 3)
    four = sum(1 for c in counts if c == 4)
    return 3 if three > 2 * four else 4


def _beat_features(hits: list[Hit], beats: list[float]) -> np.ndarray:
    """(n_beats, 2): golpes de caixa e de bumbo exatamente em cima de cada beat."""
    feats = np.zeros((len(beats), 2))
    b = np.asarray(beats)
    for h in hits:
        if h.pitch not in (KICK, SNARE):
            continue
        i = int(np.searchsorted(b, h.time))
        cand = [k for k in (i - 1, i) if 0 <= k < len(b)]
        k = min(cand, key=lambda c: abs(b[c] - h.time))
        left = b[k] - b[k - 1] if k > 0 else b[k + 1] - b[k]
        right = b[k + 1] - b[k] if k + 1 < len(b) else left
        tol = 0.25 * min(left, right)  # só "no tempo": ignora contratempos
        if abs(b[k] - h.time) <= tol:
            feats[k, 0 if h.pitch == SNARE else 1] += 1
    return np.minimum(feats, 2.0)


def bar_phase(hits, beats, downbeats, bpb: int) -> list[int]:
    """Fase do compasso (0..bpb-1) para cada beat, por Viterbi.

    Fase p significa que o beat i está na posição (i - p) % bpb do compasso.
    """
    n = len(beats)
    feats = _beat_features(hits, beats)
    if bpb == 4:
        t_snare = np.array([-0.5, 1.0, -0.5, 1.0])
        t_kick = np.array([1.0, 0.0, 0.5, 0.0])
    else:
        t_snare = np.array([-0.5] + [1.0] * (bpb - 1))
        t_kick = np.array([1.0] + [0.0] * (bpb - 1))

    down = np.zeros(n)
    for d in downbeats:
        down[int(np.argmin(np.abs(np.asarray(beats) - d)))] = DOWNBEAT_BONUS

    # score[i, p]: evidência de que o beat i está na fase p
    score = np.zeros((n, bpb))
    for p in range(bpb):
        pos = (np.arange(n) - p) % bpb
        score[:, p] = feats[:, 0] * t_snare[pos] + feats[:, 1] * t_kick[pos] + down * (pos == 0)

    # Viterbi: manter a fase é grátis, trocar custa SWITCH_PENALTY
    best = score[0].copy()
    back = np.zeros((n, bpb), dtype=int)
    for i in range(1, n):
        stay = best
        switch_from = int(np.argmax(best))
        switch = best[switch_from] - SWITCH_PENALTY
        new = np.empty(bpb)
        for p in range(bpb):
            if stay[p] >= switch:
                new[p], back[i, p] = stay[p], p
            else:
                new[p], back[i, p] = switch, switch_from
        best = new + score[i]
    phase = [0] * n
    phase[-1] = int(np.argmax(best))
    for i in range(n - 1, 0, -1):
        phase[i - 1] = int(back[i, phase[i]])
    return phase


def bar_starts_from_phase(phase: list[int], bpb: int) -> list[int]:
    """Índices de beat onde começa cada compasso. O 1º compasso começa no beat 0
    (se o beat 0 não é tempo 1, sai um compasso curto de anacruse)."""
    starts = [0]
    for i in range(1, len(phase)):
        if (i - phase[i]) % bpb == 0:
            starts.append(i)
    return starts


def backbeat_ratio(hits, beats, starts, bpb: int) -> float:
    """Fração das caixas "no tempo" que caem nos tempos 2 e 4 (bpb=4). Métrica de sanidade."""
    pos_of = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(beats)
        for i in range(s, e):
            pos_of[i] = i - s
    b = np.asarray(beats)
    good = total = 0
    for h in hits:
        if h.pitch != SNARE:
            continue
        k = int(np.argmin(np.abs(b - h.time)))
        gap = (b[min(k + 1, len(b) - 1)] - b[max(k - 1, 0)]) / 2
        if abs(b[k] - h.time) > 0.25 * gap:
            continue
        total += 1
        if pos_of.get(k, 0) in (1, 3):
            good += 1
    return good / total if total else 0.0


def choose_subdiv(hits: list[Hit], beats: list[float]) -> int:
    """3 (tercinas) se os golpes caem em 1/3 e 2/3 do beat bem mais do que em 1/4 e 3/4."""
    b = np.asarray(beats)
    pos = []
    for h in hits:
        k = int(np.searchsorted(b, h.time)) - 1
        if 0 <= k < len(b) - 1:
            pos.append(int(round((h.time - b[k]) / (b[k + 1] - b[k]) * 12)) % 12)
    if not pos:
        return SUBDIV
    share = np.bincount(pos, minlength=12) / len(pos)
    triplet, straight = share[4] + share[8], share[3] + share[9]
    return 3 if triplet > 0.15 and triplet > 1.5 * straight else SUBDIV


def quantize(hits: list[Hit], beats: list[float], starts: list[int], subdiv: int = SUBDIV):
    """Mapeia cada hit para (compasso, slot). Compassos podem ter tamanhos diferentes.

    Hits antes do primeiro beat ou depois do último são descartados.
    """
    n = len(beats)
    bounds = list(starts) + [n - 1]  # o último compasso termina no último beat
    measures = [
        Measure(beats=bounds[k + 1] - bounds[k], slots={}, start=beats[bounds[k]], subdiv=subdiv)
        for k in range(len(starts))
        if bounds[k + 1] > bounds[k]
    ]
    bar_of_beat = np.empty(n - 1, dtype=int)
    first_beat = []
    for k, m in enumerate(measures):
        first_beat.append(bounds[k])
        bar_of_beat[bounds[k]:bounds[k] + m.beats] = k

    errors, dropped = [], 0
    j = 0
    for h in hits:
        if h.time < beats[0] - 0.03 or h.time >= beats[-1]:
            dropped += 1
            continue
        while j + 1 < n - 1 and beats[j + 1] <= h.time:
            j += 1
        frac = (h.time - beats[j]) / (beats[j + 1] - beats[j])
        sub = round(frac * subdiv)
        k = int(bar_of_beat[j])
        slot = (j - first_beat[k]) * subdiv + sub
        if slot >= measures[k].beats * subdiv:
            if k + 1 < len(measures):  # arredondou para o início do próximo compasso
                k, slot = k + 1, 0
            else:
                dropped += 1
                continue
        slot = max(slot, 0)
        q_time = beats[j] + sub / subdiv * (beats[j + 1] - beats[j])
        errors.append(abs(q_time - h.time))
        measures[k].slots.setdefault(slot, set()).add(h.pitch)

    lengths = [m.beats for m in measures]
    stats = {
        "hits_total": len(hits),
        "hits_dropped": dropped,
        "quant_err_mean_ms": round(1000 * float(statistics.mean(errors)), 1) if errors else None,
        "quant_err_max_ms": round(1000 * float(max(errors)), 1) if errors else None,
        "bars": len(measures),
        "subdiv": subdiv,
        "bar_lengths": {str(l): lengths.count(l) for l in sorted(set(lengths))},
        "bpm_median": round(60 / statistics.median(b - a for a, b in zip(beats, beats[1:])), 1),
    }
    return measures, stats


def ascii_grid(measures: list[Measure], start: int = 0, count: int = 4) -> str:
    """Grade em texto para conferir a transcrição sem abrir partitura."""
    chars = {49: "CY", 42: "HH", 38: "SD", 47: "TT", 35: "BD"}
    chunk = measures[start:start + count]
    lines = []
    for p in ORDER:
        row = [
            "".join("x" if p in m.slots.get(s, ()) else "." for s in range(m.beats * m.subdiv))
            for m in chunk
        ]
        lines.append(f"{chars[p]} |" + "|".join(row) + "|")
    return "\n".join(lines)


def to_musicxml(measures: list[Measure], bpb: int, bpm: float, out: Path, title: str = "Kicksnare") -> Path:
    from music21 import clef, duration, instrument, metadata, meter, note, percussion, stream, tempo

    def make_note(pitch: int, ql: float):
        _, (step, octave), x = PIECES[pitch]
        n = note.Unpitched(displayName=f"{step}{octave}")
        n.duration = duration.Duration(ql)
        if x:
            n.notehead = "x"
        return n

    def dur_slots(n: int, subdiv: int) -> list[int]:
        # partes em slots: reta -> 4 (semínima), 3 (colcheia pont.), 2, 1; tercina -> 3, 2, 1
        parts = []
        sizes = (4, 3, 2, 1) if subdiv == 4 else (3, 2, 1)
        while n > 0:
            for p in sizes:
                if p <= n:
                    parts.append(p)
                    n -= p
                    break
        return parts

    part = stream.Part()
    inst = instrument.UnpitchedPercussion()
    inst.instrumentName = inst.partName = "Bateria"
    part.insert(0, inst)
    part.insert(0, clef.PercussionClef())
    part.insert(0, tempo.MetronomeMark(number=round(bpm)))

    current_beats = None
    for idx, m in enumerate(measures):
        meas = stream.Measure(number=idx + 1)
        if m.beats != current_beats:  # compasso irregular/anacruse muda a fórmula
            meas.insert(0, meter.TimeSignature(f"{m.beats}/4"))
            current_beats = m.beats
        sd = m.subdiv
        for beat in range(m.beats):
            base = beat * sd
            hit_slots = [s for s in range(sd) if (base + s) in m.slots]
            cursor = 0
            if not hit_slots:
                meas.append(note.Rest(quarterLength=1.0))
                continue
            for k, s in enumerate(hit_slots):
                if s > cursor:  # pausa antes do próximo ataque
                    for p in dur_slots(s - cursor, sd):
                        meas.append(note.Rest(quarterLength=Fraction(p, sd)))
                end = hit_slots[k + 1] if k + 1 < len(hit_slots) else sd
                first = True
                for p in dur_slots(end - s, sd):
                    pitches = sorted(m.slots[base + s], key=ORDER.index)
                    if first:
                        el = (
                            make_note(pitches[0], Fraction(p, sd))
                            if len(pitches) == 1
                            else percussion.PercussionChord([make_note(q, Fraction(p, sd)) for q in pitches])
                        )
                        meas.append(el)
                        first = False
                    else:  # prolonga com pausa para não repetir o ataque
                        meas.append(note.Rest(quarterLength=Fraction(p, sd)))
                cursor = end
        part.append(meas)

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    score.insert(0, part)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out))
    return out


def transcribe(song_dir: Path, device: str = "cpu") -> dict:
    drums, audio = song_dir / "drums.wav", song_dir / "audio.wav"
    hits, hat_info = detect_hits(drums, device)
    beats, downbeats = detect_beats(audio, device)
    beats, beat_fix = clean_beats(beats)
    bpb = beats_per_bar(beats, downbeats)

    # referência: compassos fixos a partir do 1º downbeat do beat_this (método anterior)
    first_db = int(np.argmin(np.abs(np.asarray(beats) - downbeats[0]))) if downbeats else 0
    naive = [0] + list(range(first_db, len(beats), bpb)) if first_db else list(range(0, len(beats), bpb))

    phase = bar_phase(hits, beats, downbeats, bpb)
    starts = bar_starts_from_phase(phase, bpb)
    subdiv = choose_subdiv(hits, beats)
    measures, stats = quantize(hits, beats, starts, subdiv)
    stats["beats_per_bar"] = bpb
    stats.update(hat_info)
    stats.update(beat_fix)
    db_idx = {int(np.argmin(np.abs(np.asarray(beats) - d))) for d in downbeats}
    stats["downbeat_agreement"] = round(sum(i in db_idx for i in starts) / len(starts), 2)
    stats["backbeat_ratio"] = round(backbeat_ratio(hits, beats, starts, bpb), 3)
    stats["backbeat_ratio_downbeat_grid"] = round(backbeat_ratio(hits, beats, naive, bpb), 3)
    # sem backbeat claro nem concordância com o beat_this, as barras de compasso são chute
    stats["bars_reliable"] = stats["backbeat_ratio"] >= 0.75 and stats["downbeat_agreement"] >= 0.7

    (song_dir / "events.json").write_text(
        json.dumps([{"t": round(h.time, 4), "pitch": h.pitch, "piece": PIECES[h.pitch][0]} for h in hits])
    )
    (song_dir / "beats.json").write_text(json.dumps({"beats": beats, "downbeats": downbeats, "beats_per_bar": bpb}))
    (song_dir / "bars.json").write_text(
        json.dumps([{"n": i + 1, "start": round(m.start, 4), "beats": m.beats} for i, m in enumerate(measures)])
    )
    (song_dir / "measures.json").write_text(json.dumps([
        {"beats": m.beats, "subdiv": m.subdiv, "start": round(m.start, 4),
         "slots": {str(s): sorted(p) for s, p in sorted(m.slots.items())}}
        for m in measures
    ]))
    to_musicxml(measures, bpb, stats["bpm_median"], song_dir / "score.musicxml", title=song_dir.name)
    (song_dir / "stats.json").write_text(json.dumps({**stats, "bpb": bpb}))
    stats["grid_preview"] = ascii_grid(measures, start=min(8, max(0, len(measures) - 4)))
    return stats
