"""Spike da etapa 1: URL ou arquivo -> audio.wav -> drums.wav, com tempo e pico de RAM.

Uso: python spike.py <url-do-youtube | caminho-do-arquivo>
"""
import subprocess
import sys

from kicksnare.ingest import ingest
from kicksnare.separate import separate_drums


def duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    song_id, audio = ingest(sys.argv[1])
    print(f"[ingest] {song_id}: {audio} ({duration(audio):.0f}s de áudio)")
    drums, metrics = separate_drums(audio)
    print(f"[demucs] {drums}")
    print(f"[demucs] {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
