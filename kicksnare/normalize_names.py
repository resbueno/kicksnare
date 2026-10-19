"""Revisa banda e nome das músicas do Kicksnare e do GrooveBass: primeira letra de cada palavra
maiúscula e grafias parecidas unificadas ("Guns N Roses" / "guns n' roses" / "GUNS N' ROSES").

Duas grafias são "a mesma" quando coincidem sem acentos, pontuação, caixa e "The" inicial. Vence a
grafia mais usada no catálogo dos dois sistemas (empate: a que tem mais acentos/pontuação). A mesma
música (mesmo id) nos dois sistemas fica com o mesmo nome. Nomes diferentes de verdade não são
trocados: só aparecem no relatório como "parecidos" para conferência.

Uso: python -m kicksnare.normalize_names            (só mostra o que mudaria)
     python -m kicksnare.normalize_names --apply    (grava)
"""
import difflib
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from . import companion, db
from .identify import title_case


def _key(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("&", " and ")
    text = re.sub(r"^the\s+", "", text.strip())
    return re.sub(r"[^0-9a-z]+", "", text)


def _richness(text: str) -> tuple:
    """Desempate: prefere a grafia com acentos e pontuação ("Guns N' Roses" a "Guns N Roses")."""
    return (sum(not c.isascii() for c in text), sum(not c.isalnum() and not c.isspace() for c in text))


def _canonical(spellings: list[str]) -> str:
    counts = Counter(spellings)
    return max(counts, key=lambda s: (counts[s], _richness(s)))


def _load() -> dict:
    """{"ks": [rows], "gb": [rows]} (gb vazio sem o GrooveBass)."""
    with db.connect() as con:
        ks = [dict(r) for r in con.execute("SELECT id, artist, title FROM songs")]
    gb = companion._gb()
    if not gb:
        return {"ks": ks, "gb": []}
    with gb[0].connect() as con:
        return {"ks": ks, "gb": [dict(r) for r in con.execute("SELECT id, artist, title FROM songs")]}


def plan(catalog: dict) -> tuple[dict, list[str]]:
    """({(sistema, id): {campo: novo}}, avisos)."""
    rows = [(sys_, r) for sys_, rs in catalog.items() for r in rs]
    cased = {(s, r["id"]): {f: title_case(r[f]) for f in ("artist", "title")} for s, r in rows}

    # mesma música nos dois sistemas: um lado sem banda/nome herda o do outro
    by_id = defaultdict(dict)
    for (s, sid), v in cased.items():
        by_id[sid][s] = v
    notes = []
    for sid, sides in by_id.items():
        if len(sides) < 2:
            continue
        a, b = sides["ks"], sides["gb"]
        for f in ("artist", "title"):
            if not a[f] or not b[f]:
                a[f] = b[f] = a[f] or b[f]
            elif _key(a[f]) != _key(b[f]):
                notes.append(f"{sid}: {f} diferente nos dois sistemas: Kicksnare {a[f]!r}, GrooveBass {b[f]!r}")

    # grafias parecidas de banda, e de nome dentro da mesma banda
    artists = defaultdict(list)
    for v in cased.values():
        if v["artist"]:
            artists[_key(v["artist"])].append(v["artist"])
    artist_of = {k: _canonical(sp) for k, sp in artists.items()}
    titles = defaultdict(list)
    for v in cased.values():
        if v["title"]:
            titles[(_key(v["artist"]), _key(v["title"]))].append(v["title"])
    title_of = {k: _canonical(sp) for k, sp in titles.items()}
    for v in cased.values():
        if v["title"]:
            v["title"] = title_of[(_key(v["artist"]), _key(v["title"]))]
        if v["artist"]:
            v["artist"] = artist_of[_key(v["artist"])]

    keys = sorted(artist_of)
    for i, k in enumerate(keys):
        for other in keys[i + 1:]:
            if difflib.SequenceMatcher(None, k, other).ratio() >= 0.85:
                notes.append(f"bandas parecidas (não unificadas): {artist_of[k]!r} / {artist_of[other]!r}")

    original = {(s, r["id"]): r for s, r in rows}
    changes = {}
    for ref, v in cased.items():
        diff = {f: v[f] for f in ("artist", "title") if v[f] != original[ref][f]}
        if diff:
            changes[ref] = diff
    return changes, notes


def main() -> None:
    apply = "--apply" in sys.argv[1:]
    catalog = _load()
    changes, notes = plan(catalog)
    original = {(s, r["id"]): r for s, rs in catalog.items() for r in rs}
    name = {"ks": "Kicksnare", "gb": "GrooveBass"}
    for (s, sid), diff in sorted(changes.items(), key=lambda x: (x[0][1], x[0][0])):
        old = original[(s, sid)]
        desc = "; ".join(f"{f}: {old[f]!r} -> {new!r}" for f, new in diff.items())
        print(f"[{name[s]}] {sid}  {desc}")
    for n in notes:
        print("AVISO", n)
    print(f"{len(changes)} músicas a corrigir de {sum(len(r) for r in catalog.values())}"
          + ("" if apply else " (simulação; use --apply para gravar)"))
    if not apply:
        return
    gb = companion._gb()
    for (s, sid), diff in changes.items():
        (db if s == "ks" else gb[0]).update(sid, **diff)
    print("gravado")


if __name__ == "__main__":
    main()
