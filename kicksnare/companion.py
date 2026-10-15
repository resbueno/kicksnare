"""Biblioteca compartilhada com o GrooveBass (baixo), quando ele está instalado na imagem.

Uma música enviada aqui entra também na fila do GrooveBass (mesmo id e mesmo dono); nome, banda e
compartilhamento acompanham. Apagar continua independente em cada sistema. Sem o pacote groovebass,
tudo aqui vira no-op.
"""
import shutil
from pathlib import Path

# arquivos que dispensam download e Demucs quando o GrooveBass já processou a mesma música
REUSE_FILES = ("audio.wav", "meta.json", "drums.wav", "no_drums.wav", "bass.wav", "separation.json")


def _gb():
    try:
        import groovebass  # noqa: F401
    except ModuleNotFoundError:
        return None
    from groovebass import config, db
    return db, config.DATA_DIR


def mirror_enqueue(song_id: str, source: str, title: str | None, owner_id: int | None) -> None:
    gb = _gb()
    if gb and not gb[0].get(song_id):
        gb[0].enqueue(song_id, source, title=title, owner_id=owner_id)


def mirror_update(song_id: str, **fields) -> None:
    """Nome, banda e compartilhamento valem nos dois sistemas."""
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


def reuse_from_groovebass(song_dir: Path) -> list[str]:
    """Copia áudio e separação do GrooveBass, se ele já processou a música. Devolve o que copiou."""
    gb = _gb()
    if not gb:
        return []
    twin = gb[1] / song_dir.name
    if not (twin / "audio.wav").exists():
        return []
    song_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REUSE_FILES:
        if name == "separation.json" and not (twin / "drums.wav").exists():
            continue
        if (twin / name).exists() and not (song_dir / name).exists():
            shutil.copyfile(twin / name, song_dir / name)
            copied.append(name)
    return copied
