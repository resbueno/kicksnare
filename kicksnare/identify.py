"""Banda e nome da música a partir do título do vídeo (ou do nome do arquivo), via Groq.

Títulos do YouTube trazem lixo ("(Official Video)", "[4K Remaster]", "Lyrics", canal "XYZVEVO") e
nem sempre a ordem "Banda - Música". Quando o YouTube Music já informa artista e faixa, usa-se isso
direto, sem chamar a API. Sem KICKSNARE_GROQ_API_KEY, a etapa é pulada.
"""
import json
import os
import re
import urllib.error
import urllib.request

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("KICKSNARE_GROQ_MODEL", "openai/gpt-oss-120b")

PROMPT = """Identify the band/artist and the song name from metadata of a music video or audio file.
Remove decorations such as "(Official Video)", "[Remastered 2011]", "Lyrics", "Live at ...", "HD",
"ft. X" stays only if it is part of the official song title. Channel names like "BandVEVO" or
"Band - Topic" refer to the band. Use the official spelling of the band and song, written in Title
Case: the first letter of every word uppercase, including short words ("Garota De Ipanema",
"Back In Black"); acronyms such as "AC/DC" stay uppercase.
If you are not reasonably sure of a field, return null for it. Answer only with JSON:
{"artist": string|null, "title": string|null}"""

# início de palavra: começo do texto ou depois de espaço, parêntese, colchete, barra, hífen ou aspas
_WORD = re.compile(r"(^|[\s(\[/\-\"“])([^\W\d_])([^\s(\[/\-\"“]*)")


def title_case(text: str | None) -> str | None:
    """Primeira letra de cada palavra maiúscula, sempre (a IA nem sempre obedece).

    O resto da palavra só vira minúsculo quando estava todo em maiúsculas ("BLACK" -> "Black"); siglas
    curtas ("AC/DC", "U2", "ABBA") ficam como estão, a não ser que o texto inteiro esteja em caixa alta.
    Grafias como "McCartney" são mantidas; apóstrofo não começa palavra ("Don't")."""
    if not text:
        return text
    shouting = text.upper() == text and sum(c.isalpha() for c in text) > 4

    def word(m):
        head, first, rest = m.groups()
        whole = first + rest
        letters = sum(c.isalpha() for c in whole)
        if whole.isupper() and (shouting or letters > 4):
            rest = rest.lower()
        return head + first.upper() + rest

    return _WORD.sub(word, text)


def enabled() -> bool:
    return bool(os.environ.get("KICKSNARE_GROQ_API_KEY"))


def _clean(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return title_case(value[:120]) if value and value.upper() not in ("NA", "NULL", "UNKNOWN") else None


def identify(raw_title: str, uploader: str | None = None, artist: str | None = None,
             track: str | None = None) -> dict:
    """{"artist", "title"} (cada um pode ser None). Lança RuntimeError se a API falhar."""
    artist, track = _clean(artist), _clean(track)
    if artist and track:   # metadados do YouTube Music: "A, B" -> artista principal
        return {"artist": artist.split(",")[0].strip(), "title": track}

    meta = {"title": raw_title, "channel": _clean(uploader)}
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": json.dumps(meta, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {os.environ['KICKSNARE_GROQ_API_KEY']}",
        "Content-Type": "application/json",
        "User-Agent": "kicksnare",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            answer = json.loads(resp.read())["choices"][0]["message"]["content"]
        found = json.loads(answer)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Groq HTTP {e.code}: {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Groq falhou: {e}") from e
    return {"artist": _clean(found.get("artist")), "title": _clean(found.get("title"))}
