"""Partes da música (Intro, Verso, Refrão...) sem IA externa, a partir do que já foi calculado.

1. Cada compasso vira um vetor: harmonia (cromagrama da faixa sem bateria), volume, e o grupo de
   padrão de bateria (patterns.cluster_bars).
2. Cortes: onde a música muda de bloco. Soma a novidade harmônica (núcleo "tabuleiro" sobre a matriz
   de autossimilaridade), a troca de groove e de volume; uma virada no compasso anterior reforça o
   corte e cortes em múltiplos de 4 compassos ganham preferência.
3. Blocos parecidos recebem a mesma letra (A, B, C...) e as letras viram nomes por regras: o bloco
   repetido mais forte é o Refrão, o repetido que vem antes dele é o Verso, o que fica sempre entre
   os dois é o Pré-refrão, o começo e o fim que não se repetem são Intro e Final, e o resto que não
   se repete é Ponte.

sections.json: {"auto": [{"bar": 1, "label": "Intro", "group": "A"}, ...], "manual": [...] | null}.
"manual" (editado pelo dono) prevalece sobre "auto".
"""
import json
from pathlib import Path

import numpy as np

from .patterns import cluster_bars, load_measures, signature

TOMS = (50, 47, 43)
MIN_LEN = 4            # compassos por parte, no mínimo
KERNEL = 4             # meia largura do núcleo de novidade, em compassos


# ---------- características por compasso ----------
def _bar_audio(wav: Path, starts: list, end: float) -> tuple[np.ndarray, np.ndarray]:
    """(cromagrama 12-d normalizado, volume em dB) de cada compasso. Lê o wav em blocos (pouca RAM)."""
    import soundfile as sf

    info = sf.info(str(wav))
    sr = info.samplerate
    nfft, hop = 8192 if sr > 30000 else 4096, 4096 if sr > 30000 else 2048
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= 60) & (freqs <= 2100)
    pc = np.zeros(len(freqs), dtype=int)
    # classe de altura com 0 = Dó (o Lá 440 é a classe 9); as partes só comparam compassos entre
    # si, mas a cifra do GrooveBass usa este cromagrama e precisa dos nomes certos
    pc[band] = (np.round(12 * np.log2(freqs[band] / 440.0)).astype(int) + 9) % 12
    win = np.hanning(nfft).astype(np.float32)

    edges = list(starts) + [end]
    n_bars = len(starts)
    chroma = np.zeros((n_bars, 12))
    energy = np.zeros(n_bars)
    count = np.zeros(n_bars)
    bar = 0
    for i, block in enumerate(sf.blocks(str(wav), blocksize=nfft, overlap=nfft - hop, dtype="float32", always_2d=True)):
        t = (i * hop + nfft / 2) / sr
        while bar < n_bars and t >= edges[bar + 1]:
            bar += 1
        if bar >= n_bars:
            break
        if t < edges[bar]:
            continue
        x = block.mean(axis=1)
        if len(x) < nfft:
            x = np.pad(x, (0, nfft - len(x)))
        mag = np.abs(np.fft.rfft(x * win))
        chroma[bar] += np.bincount(pc[band], weights=mag[band], minlength=12)
        energy[bar] += float(np.mean(x * x))
        count[bar] += 1
    count[count == 0] = 1
    chroma /= count[:, None]
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9
    db = 10 * np.log10(energy / count + 1e-10)
    return chroma, db


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _is_fill(m, cluster_size: int) -> bool:
    """Virada: tons espalhados no compasso, ou um compasso "solto" (grupo minúsculo) carregado de caixa."""
    toms = sum(1 for p in m.slots.values() for x in p if x in TOMS)
    snares = sum(1 for p in m.slots.values() if 38 in p)
    return toms >= 2 or (cluster_size <= 2 and snares >= 4)


# ---------- cortes ----------
def _boundaries(chroma, db, groups, fills) -> list[int]:
    n = len(groups)
    if n < 2 * MIN_LEN:
        return [0]
    ssm = chroma @ chroma.T
    novelty = np.zeros(n)
    for i in range(1, n):
        k = min(KERNEL, i, n - i)
        if k < 2:
            continue
        a, b = slice(i - k, i), slice(i, i + k)
        within = (ssm[a, a].mean() + ssm[b, b].mean()) / 2
        across = ssm[a, b].mean()
        harm = max(0.0, within - across)
        prev, nxt = groups[i - k:i], groups[i:i + k]
        dom_prev = max(set(prev), key=prev.count)
        dom_next = max(set(nxt), key=nxt.count)
        groove = 1.0 if dom_prev != dom_next else 0.0
        vol = min(abs(db[i:i + k].mean() - db[i - k:i].mean()) / 6.0, 1.0)
        novelty[i] = 0.5 * harm / (ssm.std() + 1e-9) + 0.8 * groove + 0.6 * vol
        if fills[i - 1]:
            novelty[i] += 0.5
    # picos com espaçamento mínimo; múltiplos de 4 compassos preferidos (frases de 4/8/16)
    thr = novelty.mean() + 0.5 * novelty.std()
    cuts = [0]
    for i in sorted(range(1, n), key=lambda j: -novelty[j]):
        if novelty[i] < thr:
            break
        if all(abs(i - c) >= MIN_LEN for c in cuts) and n - i >= MIN_LEN // 2:
            cuts.append(i)
    cuts.sort()
    # puxa cada corte para a frase de 4 compassos mais próxima quando a novidade lá é parecida
    snapped = [0]
    for c in cuts[1:]:
        best = c
        for d in (-1, 1, -2, 2):
            j = c + d
            if 0 < j < n and (j - snapped[-1]) % 4 == 0 and novelty[j] >= 0.7 * novelty[c]:
                best = j
                break
        if best - snapped[-1] >= MIN_LEN // 2:
            snapped.append(best)
    return snapped


# ---------- letras e nomes ----------
def _segment_vec(chroma, db, groups, a: int, b: int, n_groups: int) -> dict:
    q = np.array_split(np.arange(a, b), 4)
    return {
        "harm": np.concatenate([chroma[idx].mean(axis=0) if len(idx) else np.zeros(12) for idx in q]),
        "groove": np.bincount(groups[a:b], minlength=n_groups) / (b - a),
        "db": float(db[a:b].mean()),
        "len": b - a,
    }


def _similar(x: dict, y: dict) -> float:
    harm = _cos(x["harm"], y["harm"])
    groove = float(np.minimum(x["groove"], y["groove"]).sum())
    vol = float(np.exp(-abs(x["db"] - y["db"]) / 4.0))
    return 0.5 * harm + 0.35 * groove + 0.15 * vol


def _letters(segs: list[dict], threshold: float = 0.75) -> list[int]:
    reps, out = [], []
    for s in segs:
        best, best_sim = None, threshold
        for k, r in enumerate(reps):
            sim = _similar(s, r)
            if sim >= best_sim:
                best, best_sim = k, sim
        if best is None:
            reps.append(s)
            best = len(reps) - 1
        out.append(best)
    return out


def _names(letters: list[int], segs: list[dict]) -> list[str]:
    n = len(letters)
    count = {l: letters.count(l) for l in set(letters)}
    repeated = [l for l in count if count[l] >= 2]
    loud = {l: np.mean([s["db"] for s, x in zip(segs, letters) if x == l]) for l in count}
    names = [None] * n

    # refrão: o repetido mais forte; quase nunca é o que abre a música
    chorus = max(repeated, key=lambda l: (letters.index(l) > 0, loud[l], count[l])) if repeated else None
    verse = None
    if chorus is not None:
        before = [letters[i - 1] for i in range(1, n) if letters[i] == chorus and letters[i - 1] != chorus]
        cands = [l for l in repeated if l != chorus]
        # verso: repetido que aparece antes do 1º refrão (ou o repetido mais frequente)
        first_ch = letters.index(chorus)
        early = [l for l in cands if l in letters[:first_ch]]
        verse = max(early or cands, key=lambda l: count[l]) if (early or cands) else None
        pre = [l for l in cands if l != verse and before.count(l) >= 2
               and all(letters[i + 1] == chorus for i in range(n - 1) if letters[i] == l)]
        pre = pre[0] if pre else None
    else:
        pre = None
        if repeated:
            verse = max(repeated, key=lambda l: count[l])

    other = {}
    v = 0
    for i, l in enumerate(letters):
        if l == chorus:
            names[i] = "Refrão"
        elif l == verse:
            v += 1
            names[i] = f"Verso {v}"
        elif l == pre:
            names[i] = "Pré-refrão"
        elif count[l] == 1 and i == 0:
            names[i] = "Intro"
        elif count[l] == 1 and i == n - 1:
            names[i] = "Final"
        elif count[l] == 1:
            names[i] = "Ponte"
        elif letters.index(l) == 0:                # tema da abertura que volta no meio
            names[i] = "Intro" if i == 0 else "Interlúdio"
        else:
            other.setdefault(l, f"Parte {chr(ord('A') + len(other))}")
            names[i] = other[l]
    # começo bem mais baixo que o resto vira Intro mesmo se a letra se repete adiante (ex.: intro = verso sem voz)
    if n > 2 and names[0] not in ("Intro",) and segs[0]["db"] < np.median([s["db"] for s in segs]) - 6:
        names[0] = "Intro"
        k = 0
        for i in range(n):                        # renumera os versos
            if names[i].startswith("Verso"):
                k += 1
                names[i] = f"Verso {k}"
    return names


def detect(song_dir: Path) -> list[dict]:
    """Partes automáticas: [{"bar": 1-based, "label": str, "group": "A"}...]."""
    bars = json.loads((song_dir / "bars.json").read_text())
    beats = json.loads((song_dir / "beats.json").read_text())["beats"]
    measures = load_measures(song_dir)
    n = min(len(bars), len(measures))
    if n < 2 * MIN_LEN:
        return [{"bar": 1, "label": "Música", "group": "A"}]
    wav = song_dir / "no_drums.wav"
    if not wav.exists():
        wav = song_dir / "audio.wav"
    starts = [b["start"] for b in bars[:n]]
    end = bars[n]["start"] if n < len(bars) else beats[-1]
    chroma, db = _bar_audio(wav, starts, end)

    clusters = cluster_bars(measures[:n])
    groups = [0] * n
    size = {}
    for c in clusters:
        size[c.id] = len(c.bars)
        for b in c.bars:
            groups[b] = c.id
    fills = [_is_fill(measures[i], size[groups[i]]) for i in range(n)]

    cuts = _boundaries(chroma, db, groups, fills)
    spans = list(zip(cuts, cuts[1:] + [n]))
    segs = [_segment_vec(chroma, db, np.asarray(groups), a, b, len(clusters)) for a, b in spans]
    spans, letters = _tidy(spans, _letters(segs))
    segs = [_segment_vec(chroma, db, np.asarray(groups), a, b, len(clusters)) for a, b in spans]
    names = _names(letters, segs)
    order = {}                                    # letras na ordem em que aparecem: A, B, C...
    for l in letters:
        order.setdefault(l, chr(ord("A") + len(order)))
    return [{"bar": a + 1, "label": name, "group": order[l]} for (a, _), l, name in zip(spans, letters, names)]


def _tidy(spans: list, letters: list) -> tuple[list, list]:
    """Limpa a segmentação: junta partes iguais seguidas, absorve transições curtas que não se
    repetem (viradas, respiros de 1-4 compassos) na parte anterior e emenda trechos únicos
    vizinhos (um solo cortado em três vira um só)."""
    def merge_same(sp, le):
        out_s, out_l = [], []
        for s, l in zip(sp, le):
            if out_l and out_l[-1] == l:
                out_s[-1] = (out_s[-1][0], s[1])
            else:
                out_s.append(s)
                out_l.append(l)
        return out_s, out_l

    spans, letters = merge_same(spans, letters)
    count = {l: letters.count(l) for l in letters}
    keep_s, keep_l = [], []
    for i, (s, l) in enumerate(zip(spans, letters)):
        short_unique = count[l] == 1 and s[1] - s[0] <= MIN_LEN and 0 < i < len(spans) - 1
        if short_unique and keep_s:
            keep_s[-1] = (keep_s[-1][0], s[1])
        else:
            keep_s.append(s)
            keep_l.append(l)
    spans, letters = merge_same(keep_s, keep_l)
    count = {l: letters.count(l) for l in letters}
    out_s, out_l = [], []
    for i, (s, l) in enumerate(zip(spans, letters)):
        inner = 0 < i < len(spans) - 1
        if (inner and count[l] == 1 and out_l and count.get(out_l[-1]) == 1 and len(out_s) > 1):
            out_s[-1] = (out_s[-1][0], s[1])
        else:
            out_s.append(s)
            out_l.append(l)
    return out_s, out_l


# ---------- arquivo ----------
def load(song_dir: Path) -> dict:
    f = song_dir / "sections.json"
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def current(song_dir: Path) -> list[dict]:
    """Partes em uso: as editadas pelo dono, senão as automáticas."""
    data = load(song_dir)
    return data.get("manual") or data.get("auto") or []


def labels_by_bar(song_dir: Path) -> dict:
    """{índice do compasso (0-based): nome da parte} para marcar na partitura."""
    return {s["bar"] - 1: s["label"] for s in current(song_dir)}


def ensure(song_dir: Path) -> bool:
    """Calcula as partes automáticas se faltarem (músicas anteriores a este módulo) e já as marca na
    partitura. True se calculou agora."""
    if "auto" in load(song_dir) or not (song_dir / "bars.json").exists():
        return False
    refresh(song_dir)
    return True


def refresh(song_dir: Path) -> list[dict]:
    """Recalcula as partes automáticas (mantém as editadas) e regrava a partitura com elas."""
    data = load(song_dir)
    data["auto"] = detect(song_dir)
    data.setdefault("manual", None)
    (song_dir / "sections.json").write_text(json.dumps(data, ensure_ascii=False))
    write_score(song_dir)
    return data["auto"]


def write_score(song_dir: Path) -> None:
    """score.musicxml = compassos transcritos + correções de padrão + partes."""
    from .patterns import apply_edits, edits_from_json
    from .transcribe import to_musicxml

    measures = load_measures(song_dir)
    edits_file = song_dir / "edits.json"
    rows = json.loads(edits_file.read_text()) if edits_file.exists() else []
    edited = apply_edits(measures, cluster_bars(measures), edits_from_json(rows))
    stats = json.loads((song_dir / "stats.json").read_text())
    meta = song_dir / "meta.json"
    title = (json.loads(meta.read_text()).get("title") if meta.exists() else None) or song_dir.name
    to_musicxml(edited, stats["bpb"], stats["bpm_median"], song_dir / "score.musicxml", title=title,
                sections=labels_by_bar(song_dir))


def save_manual(song_dir: Path, items: list | None) -> None:
    data = load(song_dir)
    data["manual"] = items
    (song_dir / "sections.json").write_text(json.dumps(data, ensure_ascii=False))
