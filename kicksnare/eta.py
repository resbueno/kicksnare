"""Previsão de término do processamento.

Cada etapa é modelada como `fixo + ritmo * duração do áudio`. Os valores iniciais vêm de medições
(container de 2 vCPU: 40 s de áudio levaram 132 s); depois de algumas músicas o ritmo de cada etapa
passa a ser a mediana do que este servidor realmente fez. Na separação (a etapa longa) usa-se o
progresso impresso pelo Demucs. A espera na fila soma o que falta para os jobs à frente.
"""
import os
import statistics
import time

STAGES = ("ingest", "separate", "transcribe", "finalize")
# (segundos fixos, segundos por segundo de áudio)
DEFAULTS = {
    "ingest": (8.0, 0.03),
    # 2,0 medido com htdemucs; o htdemucs_ft levou ~4,4x mais no mesmo trecho
    "separate": (6.0, 2.0 * (4.4 if os.environ.get("KICKSNARE_DEMUCS_MODEL", "htdemucs_ft") == "htdemucs_ft" else 1.0)),
    "transcribe": (14.0, 0.55),
    "finalize": (3.0, 0.08),
}
DEFAULT_DURATION = 240.0     # até o áudio ser conhecido (depois do download)


def learned_rates(history: list[dict]) -> dict:
    """Ritmo por etapa (s por s de áudio) = mediana do histórico, descontando o tempo fixo."""
    out = {}
    for st in STAGES:
        fixed, default = DEFAULTS[st]
        vals = []
        for h in history:
            t, dur = (h.get("timings") or {}).get(st), h.get("duration")
            if t and dur and dur > 5:
                vals.append(max(0.0, (t - fixed) / dur))
        if len(vals) >= 2:
            out[st] = statistics.median(vals)
    return out


def stage_estimate(stage: str, duration: float | None, rates: dict) -> float:
    fixed, default = DEFAULTS[stage]
    rate = rates.get(stage, default)
    return fixed + rate * (duration or DEFAULT_DURATION)


def total_estimate(duration: float | None, rates: dict) -> float:
    return sum(stage_estimate(st, duration, rates) for st in STAGES)


CPU_STAGES = ("separate", "transcribe", "finalize")   # o download depende da rede, não da CPU


def measured_speed(song: dict, rates: dict, now: float) -> float:
    """Quanto este servidor está mais lento (>1) ou mais rápido (<1) que o previsto, medido NESTA música.

    Usa o que o Demucs já avançou na separação ou, depois dela, o tempo real das etapas já concluídas.
    Vale 1.0 quando ainda não há medida.
    """
    dur = song.get("duration")
    if song.get("status") != "running" or song.get("stage") not in STAGES:
        return 1.0
    i = STAGES.index(song["stage"])
    p = song.get("progress") or 0.0
    elapsed = max(0.0, now - (song.get("stage_started") or now))
    est_cur = stage_estimate(song["stage"], dur, rates)
    if song["stage"] == "separate" and p > 0.1 and elapsed > 3:
        speed = (elapsed / p) / est_cur
    else:
        timings = song.get("timings") or {}
        done = [st for st in STAGES[:i] if st in CPU_STAGES and st in timings]
        est = sum(stage_estimate(st, dur, rates) for st in done)
        speed = sum(timings[st] for st in done) / est if est > 0 else 1.0
    return min(5.0, max(0.15, speed))


def remaining(song: dict, rates: dict, now: float | None = None, speed: float | None = None) -> float:
    """Segundos que faltam para esta música terminar (sem contar a espera na fila)."""
    now = now or time.time()
    dur = song.get("duration")
    if speed is None:
        speed = measured_speed(song, rates, now)
    if song.get("status") != "running" or song.get("stage") not in STAGES:
        return total_estimate(dur, rates) * speed
    i = STAGES.index(song["stage"])
    est = stage_estimate(song["stage"], dur, rates) * (speed if song["stage"] in CPU_STAGES else 1.0)
    elapsed = max(0.0, now - (song.get("stage_started") or now))
    p = song.get("progress") or 0.0
    if p > 0.03:                                   # progresso medido (Demucs)
        left = elapsed * (1 - p) / p
    else:
        left = max(est - elapsed, 0.15 * est)      # passou do previsto: "quase lá", nunca zero
    later = sum(stage_estimate(st, dur, rates) * (speed if st in CPU_STAGES else 1.0) for st in STAGES[i + 1:])
    return left + later


def annotate(songs: list[dict], active: list[dict], history: list[dict], now: float | None = None) -> None:
    """Acrescenta `eta` (dict) às músicas ainda em andamento. `active` = fila inteira, em ordem."""
    now = now or time.time()
    rates = learned_rates(history)
    # a velocidade medida no job em andamento vale também para os da fila (mesma máquina)
    running = next((a for a in active if a.get("status") == "running"), None)
    speed = measured_speed(running, rates, now) if running else 1.0
    waits, acc = {}, 0.0
    for a in active:                               # espera = o que falta para todos que estão à frente
        waits[a["id"]] = acc
        acc += remaining(a, rates, now, speed)
    ahead = {a["id"]: i for i, a in enumerate(active)}
    for s in songs:
        if s.get("status") not in ("queued", "running") or s["id"] not in waits:
            continue
        own = remaining(next(a for a in active if a["id"] == s["id"]), rates, now, speed)
        total = total_estimate(s.get("duration"), rates) * (speed if s["status"] == "running" else 1.0)
        wait = waits[s["id"]]
        s["eta"] = {
            "seconds": round(wait + own),
            "wait_seconds": round(wait),
            "queue_ahead": ahead[s["id"]],
            "progress": round(max(0.0, min(1.0, 1 - own / total)), 3) if s["status"] == "running" else 0.0,
            "finish_at": round(now + wait + own),
            "calibrated": bool(rates) or speed != 1.0,
        }
