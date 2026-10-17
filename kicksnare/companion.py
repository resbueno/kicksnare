"""Biblioteca compartilhada com o GrooveBass (baixo), quando ele está instalado na imagem.

Uma música enviada aqui entra também na fila do GrooveBass (mesmo id, dono e qualidade); nome, banda
e compartilhamento acompanham. Apagar continua independente em cada sistema. Sem o pacote groovebass,
tudo aqui vira no-op.
"""
import shutil
from pathlib import Path

from .quality import STANDARD, normalize
from .separate import needs_separation

# áudio e identificação: dispensam o download
BASE_FILES = ("audio.wav", "meta.json")
# separação: dispensam o Demucs. separation.json vai por último (sem ele a pasta conta como sem separação)
STEM_FILES = ("drums.wav", "no_drums.wav", "bass.wav", "separation.json")
# feitos a partir das faixas: ficam velhos quando as faixas são trocadas
STALE_FILES = ("no_drums.mp3", "drums-gate.mp3", "drums.mp3")


def _gb():
    try:
        import groovebass  # noqa: F401
    except ModuleNotFoundError:
        return None
    from groovebass import config, db
    return db, config.DATA_DIR


def systems() -> list:
    """Bancos das filas que dividem os mesmos workers (este primeiro): base da previsão de tempo."""
    from . import db
    gb = _gb()
    return [db, gb[0]] if gb else [db]


def mirror_enqueue(song_id: str, source: str, title: str | None, owner_id: int | None,
                   quality: str = STANDARD) -> None:
    gb = _gb()
    if gb and not gb[0].get(song_id):
        gb[0].enqueue(song_id, source, title=title, owner_id=owner_id, quality=quality)


def mirror_update(song_id: str, **fields) -> None:
    """Nome, banda, compartilhamento e qualidade valem nos dois sistemas."""
    gb = _gb()
    if gb and fields and gb[0].get(song_id):
        gb[0].update(song_id, **fields)


def mirror_identified(song_id: str, fields: dict) -> None:
    """Banda/nome achados pela identificação automática: só onde o GrooveBass ainda não tem banda
    (o que o usuário editou lá prevalece)."""
    gb = _gb()
    row = gb[0].get(song_id) if gb else None
    if row and not row.get("artist") and fields:
        gb[0].update(song_id, **fields)


def copy_separation(twin: Path, song_dir: Path, quality: str) -> list[str]:
    """Copia áudio e separação de `twin` (a mesma música no outro sistema) para `song_dir`. A separação
    só vem se está completa lá, serve para a qualidade pedida e falta aqui (ou a daqui não serve).
    Devolve o que copiou. Usado nos dois sentidos (o GrooveBass chama daqui)."""
    if not (twin / "audio.wav").exists():
        return []
    song_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in BASE_FILES:
        if (twin / name).exists() and not (song_dir / name).exists():
            shutil.copyfile(twin / name, song_dir / name)
            copied.append(name)
    quality = normalize(quality)
    if ((twin / "separation.json").exists() and not needs_separation(twin, quality)
            and needs_separation(song_dir, quality)):
        (song_dir / "separation.json").unlink(missing_ok=True)
        for name in STALE_FILES:
            (song_dir / name).unlink(missing_ok=True)
        for name in STEM_FILES:
            if (twin / name).exists():
                shutil.copyfile(twin / name, song_dir / name)
                copied.append(name)
    return copied


def reuse_from_groovebass(song_dir: Path, quality: str = STANDARD) -> list[str]:
    """Copia áudio e separação do GrooveBass, se ele já processou a música. Devolve o que copiou."""
    gb = _gb()
    if not gb:
        return []
    return copy_separation(gb[1] / song_dir.name, song_dir, quality)
