"""Etapa 2: data/<id>/drums.wav -> events.json, beats.json, score.musicxml

Uso: python transcribe_cli.py <song_id>
"""
import sys
import time
from pathlib import Path

from kicksnare.transcribe import transcribe

DATA = Path(__file__).resolve().parent / "data"

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(2)
started = time.monotonic()
stats = transcribe(DATA / sys.argv[1])
preview = stats.pop("grid_preview")
print(f"[transcribe] {time.monotonic() - started:.1f}s", stats)
print(preview)
