"""Consolidação de compassos em padrões (loops) e correção em lote.

Bateria repete o groove compasso após compasso. Em vez de corrigir golpe a golpe,
agrupamos os compassos parecidos, mostramos o padrão de cada grupo (com a
frequência de cada golpe) e aplicamos uma correção do usuário a todo o grupo.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .transcribe import CHARS, ORDER, PIECES, Measure

Event = tuple  # (pitch, slot)


@dataclass
class Cluster:
    id: int
    beats: int
    subdiv: int
    bars: list = field(default_factory=list)   # índices em `measures`
    freq: dict = field(default_factory=dict)   # (pitch, slot) -> fração dos compassos
    pattern: set = field(default_factory=set)  # eventos com freq >= 0.5


def signature(m: Measure) -> set:
    return {(p, s) for s, pitches in m.slots.items() for p in pitches}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _consensus(sigs: list, min_frac: float = 0.5) -> tuple:
    counts: dict = {}
    for sig in sigs:
        for e in sig:
            counts[e] = counts.get(e, 0) + 1
    n = len(sigs)
    freq = {e: c / n for e, c in counts.items()}
    return freq, {e for e, f in freq.items() if f >= min_frac}


def cluster_bars(measures: list, threshold: float = 0.5) -> list:
    """Agrupa compassos de mesmo tamanho/grade por similaridade (Jaccard) ao consenso."""
    sigs = [signature(m) for m in measures]
    clusters: list = []
    centers: list = []
    for i, m in enumerate(measures):
        best, best_j = None, 0.0
        for c, center in zip(clusters, centers):
            if (c.beats, c.subdiv) != (m.beats, m.subdiv):
                continue
            j = _jaccard(sigs[i], center)
            if j > best_j:
                best, best_j = c, j
        if best is not None and best_j >= threshold:
            best.bars.append(i)
            idx = clusters.index(best)
        else:
            clusters.append(Cluster(id=len(clusters), beats=m.beats, subdiv=m.subdiv, bars=[i]))
            centers.append(sigs[i])
            idx = len(clusters) - 1
        _, centers[idx] = _consensus([sigs[b] for b in clusters[idx].bars])

    # 2ª passada: reatribui cada compasso ao consenso final mais próximo
    for c in clusters:
        _, centers[c.id] = _consensus([sigs[b] for b in c.bars])
    for c in clusters:
        c.bars = []
    for i, m in enumerate(measures):
        cands = [c for c in clusters if (c.beats, c.subdiv) == (m.beats, m.subdiv)]
        best = max(cands, key=lambda c: _jaccard(sigs[i], centers[c.id]))
        best.bars.append(i)
    clusters = [c for c in clusters if c.bars]
    for new_id, c in enumerate(sorted(clusters, key=lambda c: -len(c.bars))):
        c.id = new_id
        c.freq, c.pattern = _consensus([sigs[b] for b in c.bars])
    return sorted(clusters, key=lambda c: c.id)


@dataclass
class Edit:
    """Correção em lote. cluster=None vale para todos os compassos compatíveis."""
    remove: list = field(default_factory=list)  # [(pitch, slot)]
    add: list = field(default_factory=list)     # [(pitch, slot)]
    cluster: int = None


def apply_edits(measures: list, clusters: list, edits: list) -> list:
    """Devolve novos compassos com as correções aplicadas (não altera a entrada)."""
    out = [Measure(m.beats, {s: set(p) for s, p in m.slots.items()}, m.start, m.subdiv) for m in measures]
    member = {b: c for c in clusters for b in c.bars}
    for edit in edits:
        for i, m in enumerate(out):
            c = member.get(i)
            if c is None or (edit.cluster is not None and c.id != edit.cluster):
                continue
            n_slots = m.beats * m.subdiv
            for pitch, slot in edit.remove:
                if slot in m.slots:
                    m.slots[slot].discard(pitch)
                    if not m.slots[slot]:
                        del m.slots[slot]
            for pitch, slot in edit.add:
                if slot < n_slots:
                    m.slots.setdefault(slot, set()).add(pitch)
    return out


def _ranges(idx: list) -> str:
    idx = sorted(i + 1 for i in idx)
    parts, start, prev = [], idx[0], idx[0]
    for v in idx[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


def grid(cluster: Cluster) -> str:
    n = cluster.beats * cluster.subdiv
    lines = []
    for p in ORDER:
        row = "".join(
            ("X" if (p, s) in cluster.pattern else ("?" if cluster.freq.get((p, s), 0) >= 0.2 else "."))
            for s in range(n)
        )
        lines.append(f"  {CHARS[p]} |{row}|")
    return "\n".join(lines)


def report(clusters: list, total_bars: int, min_bars: int = 2, max_uncertain: int = 12) -> str:
    out = []
    covered = sum(len(c.bars) for c in clusters if len(c.bars) >= min_bars)
    out.append(
        f"{len(clusters)} grupos; {covered}/{total_bars} compassos "
        f"({100 * covered / total_bars:.0f}%) em grupos com {min_bars}+ compassos\n"
    )
    for c in clusters:
        if len(c.bars) < min_bars:
            continue
        out.append(f"Grupo {c.id}: {len(c.bars)} compassos ({100 * len(c.bars) / total_bars:.0f}%), "
                   f"{c.beats} beats, grade {c.subdiv}/beat — compassos {_ranges(c.bars)}")
        out.append(grid(c))
        unsure = sorted(
            ((e, f) for e, f in c.freq.items() if 0.2 <= f < 0.8),
            key=lambda ef: (ORDER.index(ef[0][0]), ef[0][1]),
        )
        if unsure:
            txt = ", ".join(f"{PIECES[p][0]}@{s} {100 * f:.0f}%" for (p, s), f in unsure[:max_uncertain])
            out.append(f"  incertos (X = no padrão, ? = aparece em 20–50%): {txt}")
        out.append("")
    return "\n".join(out)


def load_measures(song_dir: Path) -> list:
    raw = json.loads((song_dir / "measures.json").read_text())
    return [
        Measure(r["beats"], {int(s): set(p) for s, p in r["slots"].items()}, r["start"], r["subdiv"])
        for r in raw
    ]


def with_state(clusters: list, measures: list) -> list:
    """Cópia dos grupos (mesma pertinência) com freq/padrão recalculados sobre `measures`."""
    out = []
    for c in clusters:
        n = Cluster(id=c.id, beats=c.beats, subdiv=c.subdiv, bars=list(c.bars))
        n.freq, n.pattern = _consensus([signature(measures[b]) for b in c.bars])
        out.append(n)
    return out


def edits_from_json(rows: list) -> list:
    return [
        Edit(cluster=r["cluster"], remove=[tuple(e) for e in r["remove"]], add=[tuple(e) for e in r["add"]])
        for r in rows
    ]
