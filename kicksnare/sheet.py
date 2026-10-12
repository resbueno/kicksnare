"""Partitura em PDF: páginas, compassos, ordem de execução e sincronia com o áudio.

Sem OMR. Para sincronizar não precisamos ler as notas, só saber onde cada compasso
está na página. As barras de compasso de uma pauta de 5 linhas são linhas verticais
que cobrem a pauta de ponta a ponta, o que se detecta bem com morfologia; o usuário
corrige o que errar e informa a ordem real de execução (repetições, voltas).
"""
import bisect
import json
import re
from pathlib import Path

import cv2
import numpy as np

DPI = 150
MAX_PAGES = 40


# ---------- PDF -> imagens ----------
def render_pdf(pdf_path: Path, out_dir: Path, dpi: int = DPI) -> list:
    """Renderiza cada página em PNG (pypdfium2: licença BSD/Apache, ao contrário do PyMuPDF/AGPL)."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(pdf) > MAX_PAGES:
            raise ValueError(f"PDF com {len(pdf)} páginas (máximo {MAX_PAGES})")
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=dpi / 72)
            img = bitmap.to_numpy()                      # BGRA
            cv2.imwrite(str(out_dir / f"page-{i + 1}.png"), cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
            pages.append({"page": i + 1, "width": img.shape[1], "height": img.shape[0]})
    finally:
        pdf.close()
    return pages


# ---------- detecção ----------
def _groups(idx: np.ndarray, gap: int) -> list:
    """Agrupa índices inteiros ordenados em (início, fim) com lacunas <= gap."""
    if len(idx) == 0:
        return []
    out, s, p = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - p > gap:
            out.append((s, p))
            s = v
        p = v
    out.append((s, p))
    return out


def _find_staves(bw: np.ndarray) -> list:
    h, w = bw.shape
    k = max(30, w // 20)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    row_sum = horiz.sum(axis=1)
    rows = np.where(row_sum >= max(k, 0.10 * w))[0]
    lines = []
    for y0, y1 in _groups(rows, 1):
        cols = np.where(horiz[y0:y1 + 1].any(axis=0))[0]
        xl, xr = int(cols.min()), int(cols.max())
        lines.append(((y0 + y1) / 2, xl, xr))
        if y1 - y0 + 1 > 4:
            # linha da pauta colada em viga/barra horizontal: o centro do grupo erra,
            # então oferece também as bordas como posições candidatas
            lines.append((y0 + 0.5, xl, xr))
            lines.append((y1 - 0.5, xl, xr))
    lines.sort(key=lambda l: l[0])

    staves, i = [], 0
    ys = [l[0] for l in lines]
    while i < len(lines):
        found = None
        # j é candidata à 5ª linha: o espaçamento sai da distância total (o erro de
        # arredondamento de 1 px por linha não se acumula como na medida entre vizinhas)
        for j in range(i + 1, len(lines)):
            span = ys[j] - ys[i]
            if span > 160:
                break
            sp = span / 4
            if sp < 4:
                continue
            tol = max(1.5, 0.12 * sp)
            idx = [i]
            ok = True
            for m in (1, 2, 3):
                want = ys[i] + m * sp
                cand = [k for k in range(i + 1, j) if abs(ys[k] - want) <= tol]
                if not cand:
                    ok = False
                    break
                idx.append(min(cand, key=lambda k: abs(ys[k] - want)))
            if ok:
                found = idx + [j]
                break
        if found is None:
            i += 1
            continue
        yy = [ys[k] for k in found]
        sp = float(np.mean(np.diff(yy)))
        xl = int(np.median([lines[k][1] for k in found]))
        xr = int(np.median([lines[k][2] for k in found]))
        staves.append({"y_top": yy[0], "y_bottom": yy[-1], "space": sp, "x_left": xl, "x_right": xr})
        i = found[-1] + 1
    return staves


def _dots_near(bw: np.ndarray, staff: dict, x: float, side: int) -> bool:
    """Dois pontos (sinal de repetição) ao lado de uma barra, nos espaços 2 e 3."""
    sp = staff["space"]
    xs = int(x + side * 0.9 * sp)
    r = max(2, int(0.25 * sp))
    hits = 0
    for k in (1.5, 2.5):
        yc = int(staff["y_top"] + k * sp)
        win = bw[yc - r:yc + r + 1, xs - r:xs + r + 1]
        if win.size and win.mean() > 0.5:
            hits += 1
    return hits == 2


def _vertical_extent(bw: np.ndarray, x: int, y_top: int, y_bottom: int, limit: int) -> tuple:
    """Quanto a linha vertical em x continua acima/abaixo da pauta (tolera 1 px de falha)."""
    h = bw.shape[0]
    col = bw[:, max(0, x - 1):x + 2].max(axis=1)

    def walk(start: int, step: int) -> int:
        n, gap, y = 0, 0, start
        while 0 <= y < h and n < limit:
            if col[y]:
                n += 1 + gap
                gap = 0
            else:
                gap += 1
                if gap > 1:
                    break
            y += step
        return n

    return walk(y_top - 1, -1), walk(y_bottom + 1, +1)


def _has_notehead_beside(bw: np.ndarray, staff: dict, c0: int, c1: int) -> bool:
    """Cabeça de nota encostada na linha vertical => é haste, não barra de compasso.

    Olha as colunas imediatamente ao lado, dentro da pauta e sem contar as linhas da
    pauta: uma cabeça deixa uma corrida vertical de ~1 espaço; barra isolada, nada.
    """
    sp = staff["space"]
    y0, y1 = int(round(staff["y_top"])), int(round(staff["y_bottom"]))
    off = max(3, int(round(0.3 * sp)))   # além do halo do anti-aliasing da própria barra
    need = 0.6 * sp
    for xc in (c0 - off, c1 + off):
        if xc < 0 or xc >= bw.shape[1]:
            continue
        col = bw[y0:y1 + 1, xc].copy()
        for k in range(5):  # apaga as 5 linhas da pauta (±1 px)
            yl = int(round(k * sp))
            col[max(0, yl - 1):yl + 2] = 0
        run = best = 0
        for v in col:
            run = run + 1 if v else 0
            best = max(best, run)
        if best >= need:
            return True
    return False


def _barlines(bw: np.ndarray, staff: dict, others: list = ()) -> list:
    sp, y0, y1 = staff["space"], int(round(staff["y_top"])), int(round(staff["y_bottom"]))
    x0 = max(0, staff["x_left"] - 3)
    x1 = min(bw.shape[1], staff["x_right"] + 4)
    band = bw[y0:y1 + 1, x0:x1]
    kh = max(3, int(0.95 * (y1 - y0 + 1)))
    vert = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
    cols = np.where(vert.any(axis=0))[0] + x0
    out = []
    for c0, c1 in _groups(cols, max(2, int(0.45 * sp))):
        width = c1 - c0 + 1
        x = (c0 + c1) / 2
        # haste de nota: passa dos limites da pauta (barra de compasso termina na 1ª e na 5ª linha)
        up, down = _vertical_extent(bw, int(round(x)), y0, y1, int(8 * sp))
        lim = max(2.0, 0.12 * sp)   # barra termina na 1ª/5ª linha; haste passa mais (medido nos dois renderizadores: barra <=2 px, haste >=3 px)
        joins_up = any(abs((o["y_bottom"]) - (y0 - up)) <= lim for o in others if o["y_bottom"] < y0)
        joins_down = any(abs((o["y_top"]) - (y1 + down)) <= lim for o in others if o["y_top"] > y1)
        if (up > lim and not joins_up) or (down > lim and not joins_down):
            continue
        if _has_notehead_beside(bw, staff, c0, c1):
            continue
        kind = "bar"
        thick = width > 0.35 * sp + 1
        if _dots_near(bw, staff, c0, -1):
            kind = "end_repeat"
        elif _dots_near(bw, staff, c1, +1):
            kind = "start_repeat"
        out.append({"x": round(x, 1), "kind": kind, "thick": bool(thick)})
    return out


def detect_layout(images: list) -> dict:
    """images: lista de arrays em tons de cinza (uma por página)."""
    pages = []
    for n, gray in enumerate(images, start=1):
        bw = (gray < 150).astype(np.uint8)
        staves = []
        found = _find_staves(bw)
        for st in found:
            bars = _barlines(bw, st, [o for o in found if o is not st])
            st = {**st, "bars": bars, "ignore": False}
            staves.append(st)
        pages.append({"page": n, "width": int(gray.shape[1]), "height": int(gray.shape[0]), "staves": staves})
    return {"pages": pages}


# ---------- layout -> compassos ----------
def measures_from_layout(layout: dict) -> list:
    """Lista plana de compassos, em ordem de leitura (página, pauta, esquerda->direita)."""
    out = []
    for pg in layout["pages"]:
        for si, st in enumerate(pg["staves"]):
            if st.get("ignore"):
                continue
            sp = st["space"]
            xl, xr = st["x_left"], st["x_right"]
            bars = sorted(st["bars"], key=lambda b: b["x"])
            inner = [b for b in bars if b["x"] > xl + 1.5 * sp]
            bounds = [{"x": xl, "kind": bars[0]["kind"] if bars and bars[0]["x"] <= xl + 1.5 * sp else "bar"}]
            for b in inner:
                if b["x"] >= xr - 1.5 * sp:
                    continue
                bounds.append(b)
            closing = [b for b in inner if b["x"] >= xr - 1.5 * sp]
            bounds.append({"x": xr, "kind": closing[-1]["kind"] if closing else "bar"})
            # compasso estreito demais é quase sempre uma haste colada na barra (ou um traço
            # da fórmula de compasso, no começo da linha): funde a fronteira falsa (a de dentro)
            min_w = 4.0 * sp
            changed = True
            while changed and len(bounds) > 2:
                changed = False
                for i in range(len(bounds) - 1):
                    # o 1º compasso da linha leva clave e fórmula de compasso: precisa de mais espaço
                    need_w = 7.0 * sp if i == 0 else min_w
                    if bounds[i + 1]["x"] - bounds[i]["x"] < need_w:
                        del bounds[i if i > 0 else i + 1]
                        changed = True
                        break
            for a, b in zip(bounds, bounds[1:]):
                out.append({
                    "n": len(out) + 1,
                    "page": pg["page"], "staff": si,
                    "x0": round(a["x"], 1), "x1": round(b["x"], 1),
                    "y0": round(st["y_top"] - 2.0 * sp, 1), "y1": round(st["y_bottom"] + 2.0 * sp, 1),
                    "start_repeat": a["kind"] == "start_repeat",
                    "end_repeat": b["kind"] == "end_repeat",
                })
    return out


def suggest_order(measures: list) -> list:
    """Ordem de execução a partir dos sinais de repetição simples (sem voltas/D.S.)."""
    order, sec, i, done = [], 0, 0, set()
    while i < len(measures):
        if measures[i]["start_repeat"]:
            sec = i
        order.append(i + 1)
        if measures[i]["end_repeat"] and (sec, i) not in done:
            done.add((sec, i))
            i = sec
            continue
        i += 1
    return order


_TOKEN = re.compile(r"^(\d+)(?:\s*[-–]\s*(\d+))?(?:\s*[x×]\s*(\d+))?$", re.I)


def parse_order(text: str, n_measures: int) -> list:
    """'1-8, 1-8, 9-16, 3 x2' -> [1..8, 1..8, 9..16, 3, 3] (validado contra n_measures)."""
    order = []
    for raw in re.split(r"[,;\n]+", text):
        tok = raw.strip()
        if not tok:
            continue
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"trecho inválido: '{tok}'")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        times = int(m.group(3) or 1)
        if a < 1 or b > n_measures or b < a:
            raise ValueError(f"'{tok}' fora do intervalo 1–{n_measures}")
        if times < 1 or times > 64:
            raise ValueError(f"repetição inválida em '{tok}'")
        order.extend(list(range(a, b + 1)) * times)
    if not order:
        raise ValueError("ordem vazia")
    return order


def order_to_text(order: list) -> str:
    """Compacta [1..8,1..8,9..16] em '1-8, 1-8, 9-16'."""
    parts, i = [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1] == order[j] + 1:
            j += 1
        parts.append(str(order[i]) if i == j else f"{order[i]}-{order[j]}")
        i = j + 1
    return ", ".join(parts)


# ---------- sincronia: posição na ordem -> tempo no áudio ----------
def position_times(n_pos: int, anchors: list, bars: list, audio_end: float) -> list:
    """Tempo (s) de início de cada posição da ordem de execução (1..n_pos).

    anchors: [{"pos": int, "time": float}]. Sem âncoras, a posição 1 é o primeiro
    compasso do áudio. Entre âncoras, se o número de compassos do áudio bate com o da
    partitura usa-se a grade de compassos detectada; senão, interpolação linear.
    Depois da última âncora segue a grade de compassos do áudio.
    """
    starts = [b["start"] for b in bars]
    if not starts:
        raise ValueError("áudio sem compassos detectados")

    def bar_at(t: float) -> int:
        """Compasso do áudio cujo início é o mais próximo de t (clique 30 ms antes do
        tempo 1 não pode cair no compasso anterior)."""
        i = bisect.bisect_left(starts, t)
        cand = [k for k in (i - 1, i) if 0 <= k < len(starts)]
        return min(cand, key=lambda k: abs(starts[k] - t))

    def snap(t: float) -> float:
        k = bar_at(t)
        width = (starts[k + 1] - starts[k]) if k + 1 < len(starts) else (starts[k] - starts[k - 1] if k else 1.0)
        return starts[k] if abs(starts[k] - t) <= 0.25 * width else t

    def chain(a: dict, pos: int):
        k = bar_at(a["time"]) + (pos - a["pos"])
        return starts[k] if 0 <= k < len(starts) else None

    anc = sorted({int(a["pos"]): snap(float(a["time"])) for a in anchors}.items())
    pts = [{"pos": p, "time": t} for p, t in anc] or [{"pos": 1, "time": starts[0]}]
    times: list = [None] * (n_pos + 1)

    def fill_segment(a: dict, b: dict) -> None:
        span = b["pos"] - a["pos"]
        inside = sum(1 for s in starts if a["time"] - 1e-3 <= s < b["time"] - 1e-3)
        use_chain = inside == span
        for p in range(a["pos"], b["pos"]):
            if p < 1 or p > n_pos:
                continue
            t = chain(a, p) if use_chain else None
            if t is None:
                t = a["time"] + (b["time"] - a["time"]) * (p - a["pos"]) / span
            times[p] = t

    first = pts[0]
    for p in range(1, min(first["pos"], n_pos + 1)):   # antes da 1ª âncora: grade para trás
        k = bar_at(first["time"]) - (first["pos"] - p)
        times[p] = starts[k] if k >= 0 else 0.0
    for a, b in zip(pts, pts[1:]):
        if b["pos"] > a["pos"]:
            fill_segment(a, b)
    last = pts[-1]
    for p in range(last["pos"], n_pos + 1):
        t = chain(last, p)
        times[p] = t if t is not None else audio_end
    times[n_pos] = times[n_pos] if times[n_pos] is not None else audio_end
    return [t if t is not None else audio_end for t in times[1:]]


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default
