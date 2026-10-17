"""Previsão de término do processamento.

Cada etapa é modelada como `fixo + ritmo * duração do áudio`. Os valores iniciais vêm de medições
(container de 2 vCPU: 40 s de áudio levaram 132 s; HiFi separa ~4,4x mais devagar); depois de algumas
músicas o ritmo de cada etapa passa a ser a mediana do que este servidor realmente fez, separado por
qualidade (cada uma tem o seu worker e o seu ritmo). Na separação (a etapa longa) usa-se o progresso
impresso pelo Demucs.

A espera na fila soma o que falta para os jobs à frente no MESMO worker: o worker atende as filas do
Kicksnare e do GrooveBass (quem espera há mais tempo primeiro), então a fila de um worker junta as duas.
A segunda passada de uma música (o espelho no outro sistema) reaproveita a separação e não a conta.
"""
import statistics
import time

from .quality import HIFI, QUALITIES, STANDARD, normalize, worker_groups

STAGES = ("ingest", "separate", "transcribe", "finalize")
FIXED = {"ingest": 8.0, "separate": 6.0, "transcribe": 14.0, "finalize": 3.0}     # segundos fixos
# segundos por segundo de áudio; 2,0 na separação medido com htdemucs no servidor. HiFi (htdemucs_ft):
# 4,9x a CPU do htdemucs, medido num bloco de 3 min (72 -> 355 s de CPU por minuto de áudio)
DEFAULT_RATES = {
    STANDARD: {"ingest": 0.03, "separate": 2.0, "transcribe": 0.55, "finalize": 0.08},
    HIFI: {"ingest": 0.03, "separate": 2.0 * 4.9, "transcribe": 0.55, "finalize": 0.08},
}
DEFAULT_DURATION = 240.0     # até o áudio ser conhecido (depois do download)
CPU_STAGES = ("separate", "transcribe", "finalize")   # o download depende da rede, não da CPU


def rates_for(history: list[dict], quality: str = STANDARD) -> dict:
    """Ritmo de cada etapa (s por s de áudio): mediana do histórico desta qualidade, descontando o
    tempo fixo; sem histórico suficiente (2 músicas), o valor medido de antemão. A chave `learned`
    diz se algo veio do histórico."""
    out = dict(DEFAULT_RATES[normalize(quality)])
    learned = False
    for st in STAGES:
        vals = []
        for h in history:
            t, dur = (h.get("timings") or {}).get(st), h.get("duration")
            if t and dur and dur > 5:
                vals.append(max(0.0, (t - FIXED[st]) / dur))
        if len(vals) >= 2:
            out[st] = statistics.median(vals)
            learned = True
    out["learned"] = learned
    return out


def stage_estimate(stage: str, duration: float | None, rates: dict) -> float:
    return FIXED[stage] + rates[stage] * (duration or DEFAULT_DURATION)


def total_estimate(duration: float | None, rates: dict, skip: tuple = ()) -> float:
    return sum(stage_estimate(st, duration, rates) for st in STAGES if st not in skip)


def measured_speed(song: dict, rates: dict, now: float) -> float:
    """Quanto este worker está mais lento (>1) ou mais rápido (<1) que o previsto, medido NESTA música.

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


def remaining(song: dict, rates: dict, now: float | None = None, speed: float | None = None,
              skip: tuple = ()) -> float:
    """Segundos que faltam para esta música terminar (sem contar a espera na fila). `skip`: etapas que
    não vão custar nada (separação reaproveitada)."""
    now = now or time.time()
    dur = song.get("duration")
    if speed is None:
        speed = measured_speed(song, rates, now)
    if song.get("status") != "running" or song.get("stage") not in STAGES:
        return total_estimate(dur, rates, skip) * speed
    i = STAGES.index(song["stage"])
    est = stage_estimate(song["stage"], dur, rates) * (speed if song["stage"] in CPU_STAGES else 1.0)
    elapsed = max(0.0, now - (song.get("stage_started") or now))
    p = song.get("progress") or 0.0
    if p > 0.03:                                   # progresso medido (Demucs)
        left = elapsed * (1 - p) / p
    else:
        left = max(est - elapsed, 0.15 * est)      # passou do previsto: "quase lá", nunca zero
    later = sum(stage_estimate(st, dur, rates) * (speed if st in CPU_STAGES else 1.0)
                for st in STAGES[i + 1:] if st not in skip)
    return left + later


def worker_queue(systems: list, qualities: tuple) -> list[dict]:
    """Fila de UM worker, na ordem em que ele vai processar: o que está rodando e depois, a cada passo,
    a próxima da fila do sistema cuja música espera há mais tempo (como o worker faz).

    `systems`: módulos de banco (Kicksnare, GrooveBass) com active_songs(qualidades) e history(quality=).
    Cada item: {"sys": índice do sistema, "song", "rates", "skip"}.
    """
    lists = []
    for i, sysdb in enumerate(systems):
        rates = {q: rates_for(sysdb.history(quality=q), q) for q in qualities}
        lists.append([{"sys": i, "song": s, "rates": rates[normalize(s.get("quality"))], "skip": ()}
                      for s in sysdb.active_songs(qualities)])
    order = [it for lst in lists for it in lst if it["song"]["status"] == "running"]
    queued = [[it for it in lst if it["song"]["status"] == "queued"] for lst in lists]
    while any(queued):
        i = min((k for k, lst in enumerate(queued) if lst),
                key=lambda k: min(it["song"]["created"] for it in queued[k]))
        order.append(queued[i].pop(0))
    seen = set()
    for it in order:                 # espelho no outro sistema: a separação já estará feita
        key = (it["song"]["id"], normalize(it["song"].get("quality")))
        if key in seen and it["song"]["status"] == "queued":
            it["skip"] = ("separate",)
        seen.add(key)
    return order


def _waits(queue: list[dict], now: float) -> tuple[dict, float, float]:
    """(espera de cada item, espera total da fila, velocidade medida no job em andamento)."""
    running = next((it for it in queue if it["song"].get("status") == "running"), None)
    speed = measured_speed(running["song"], running["rates"], now) if running else 1.0
    waits, acc = {}, 0.0
    for it in queue:
        waits[(it["sys"], it["song"]["id"])] = acc
        acc += remaining(it["song"], it["rates"], now, speed, it["skip"])
    return waits, acc, speed


def annotate(songs: list[dict], systems: list, now: float | None = None) -> None:
    """Acrescenta `eta` (dict) às músicas ainda em andamento. `songs` são do sistema systems[0]."""
    now = now or time.time()
    for qualities in worker_groups():
        queue = worker_queue(systems, qualities)
        waits, _, speed = _waits(queue, now)
        ahead = {}
        for n, it in enumerate(queue):
            ahead[(it["sys"], it["song"]["id"])] = n
        for s in songs:
            key = (0, s["id"])
            if s.get("status") not in ("queued", "running") or key not in waits:
                continue
            it = next(i for i in queue if (i["sys"], i["song"]["id"]) == key)
            own = remaining(it["song"], it["rates"], now, speed, it["skip"])
            total = total_estimate(s.get("duration"), it["rates"], it["skip"]) * (speed if s["status"] == "running" else 1.0)
            wait = waits[key]
            s["eta"] = {
                "seconds": round(wait + own),
                "wait_seconds": round(wait),
                "queue_ahead": ahead[key],
                "progress": round(max(0.0, min(1.0, 1 - own / total)), 3) if s["status"] == "running" else 0.0,
                "finish_at": round(now + wait + own),
                "calibrated": it["rates"].get("learned", False) or speed != 1.0,
            }


def forecast(duration: float | None, systems: list, now: float | None = None) -> dict:
    """Previsão para uma música que ainda vai ser enviada, em cada qualidade: espera na fila do worker
    daquela qualidade + o processamento dela (ritmo do sistema systems[0]). `duration` None = 4 min."""
    now = now or time.time()
    out = {}
    for qualities in worker_groups():
        queue = worker_queue(systems, qualities)
        _, wait, speed = _waits(queue, now)
        for q in qualities:
            rates = rates_for(systems[0].history(quality=q), q)
            own = total_estimate(duration, rates) * speed
            out[q] = {"seconds": round(wait + own), "wait_seconds": round(wait), "own_seconds": round(own),
                      "queue_ahead": len(queue), "calibrated": rates["learned"] or speed != 1.0}
    return {q: out[q] for q in QUALITIES}


_DURATIONS: dict = {}     # url -> duração (o yt-dlp leva alguns segundos; o usuário costuma repetir o link)


def new_song(url: str | None, duration: float | None, systems: list) -> dict:
    """Resposta de /api/estimate: previsão de uma música nova em cada qualidade, antes de enviar.
    `url`: a duração vem do yt-dlp; `duration` (s): lida pelo navegador no arquivo; sem nenhum dos
    dois, uma música de 4 min."""
    import re

    from .ingest import url_duration
    from .quality import LABELS

    source = "default"
    if url and re.match(r"^https?://\S+$", url.strip()):
        url = url.strip()
        if url not in _DURATIONS:
            if len(_DURATIONS) > 500:
                _DURATIONS.clear()
            _DURATIONS[url] = url_duration(url)
        duration = _DURATIONS[url]
        source = "url" if duration else "default"
    elif duration and duration > 0:
        source = "file"
    duration = max(5.0, min(float(duration), 3 * 3600.0)) if duration and source != "default" else None
    return {"duration": round(duration) if duration else None, "duration_source": source, "labels": LABELS,
            **forecast(duration, systems)}
