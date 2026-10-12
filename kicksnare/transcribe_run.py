"""Roda a transcrição num processo próprio (libera a RAM do torch ao terminar).

Uso: python -m kicksnare.transcribe_run <song_id>   -> imprime JSON das estatísticas
"""
import json
import sys

from .ingest import DATA_DIR
from .transcribe import transcribe

if __name__ == "__main__":
    stats = transcribe(DATA_DIR / sys.argv[1])
    stats.pop("grid_preview", None)
    print("RESULT " + json.dumps(stats))
