"""Qualidade da separação, escolhida por música: "standard" (htdemucs, rápido) ou "hifi" (htdemucs_ft:
4 redes, uma por instrumento, menos vazamento entre as faixas; ~4,4x mais CPU).

Cada qualidade tem a sua fila e o seu worker (KICKSNARE_WORKER_QUALITY), que rodam em paralelo; o
worker HiFi tem prioridade de CPU menor e só usa o que sobra. Uma música separada em HiFi não é
refeita em padrão (reprocessar em padrão reaproveita a separação melhor).
"""
import os

STANDARD, HIFI = "standard", "hifi"
QUALITIES = (STANDARD, HIFI)
LABELS = {STANDARD: "Padrão", HIFI: "HiFi"}
# modelo do Demucs de cada qualidade (KICKSNARE_DEMUCS_MODEL continua valendo para o padrão)
MODELS = {
    STANDARD: os.environ.get("KICKSNARE_DEMUCS_MODEL", "htdemucs"),
    HIFI: os.environ.get("KICKSNARE_DEMUCS_MODEL_HIFI", "htdemucs_ft"),
}


def normalize(value: str | None) -> str:
    """Valor vindo da API ou do banco -> qualidade válida (o que não for HiFi é padrão)."""
    return HIFI if (value or "").strip().lower() == HIFI else STANDARD


def sql_filter(qualities=None) -> tuple[str, tuple]:
    """(`quality IN (?,...)`, valores) para filtrar as filas pelas qualidades pedidas (todas, se None)."""
    qs = tuple(QUALITIES if qualities is None else (qualities,) if isinstance(qualities, str) else qualities)
    return f"quality IN ({','.join('?' * len(qs))})", qs


def worker_qualities() -> tuple[str, ...]:
    """Filas que este worker atende: KICKSNARE_WORKER_QUALITY = standard | hifi | all (padrão: all,
    um worker só para as duas, como antes de existirem dois)."""
    v = os.environ.get("KICKSNARE_WORKER_QUALITY", "all").strip().lower()
    if v in QUALITIES:
        return (v,)
    return QUALITIES


def worker_groups() -> list[tuple[str, ...]]:
    """Como as filas se dividem entre os workers, para a previsão de tempo (vale no container web):
    KICKSNARE_SPLIT_WORKERS=1 -> um worker por qualidade, em paralelo; senão um worker só para as duas."""
    if os.environ.get("KICKSNARE_SPLIT_WORKERS", "0").strip() in ("1", "true", "yes"):
        return [(q,) for q in QUALITIES]
    return [QUALITIES]
