"""Agrupa os compassos de data/<id>/ em padrões e mostra o relatório.

Uso: python patterns_cli.py <song_id>
"""
import sys
from pathlib import Path

from kicksnare.patterns import cluster_bars, load_measures, report

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(2)
measures = load_measures(Path(__file__).resolve().parent / "data" / sys.argv[1])
print(report(cluster_bars(measures), len(measures)))
