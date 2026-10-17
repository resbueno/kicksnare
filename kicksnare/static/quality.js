// Escolha da qualidade da separação (Padrão ou HiFi) com a previsão de tempo de cada uma, antes de
// enviar a música. A previsão vem de GET <base>/api/estimate (fila do worker de cada qualidade + o
// tempo da própria música); a duração vem do link (o servidor pergunta ao yt-dlp) ou do arquivo
// (lida aqui no navegador). O GrooveBass usa uma cópia deste arquivo.
//
//   const picker = qualityPicker(document.querySelector("#quality"), BASE);
//   picker.forUrl(url) | picker.forFile(file) | picker.forDuration(seg) | picker.reset()
//   picker.value -> "standard" | "hifi" (também atribuível)
function qualityPicker(box, base) {
  const DESC = {
    standard: "separação rápida",
    hifi: "4 redes, uma por instrumento: menos vazamento entre as faixas, bem mais demorado",
  };
  box.classList.add("quality");
  box.innerHTML = '<span class="muted small">Qualidade da separação:</span>';
  const spans = {};
  for (const q of ["standard", "hifi"]) {
    const label = document.createElement("label");
    label.className = "qopt";
    label.title = DESC[q];
    const input = document.createElement("input");
    input.type = "radio"; input.name = "quality"; input.value = q; input.checked = q === "standard";
    const name = document.createElement("b"); name.textContent = q === "hifi" ? "HiFi" : "Padrão";
    spans[q] = document.createElement("span"); spans[q].className = "muted small";
    label.append(input, " ", name, " ", spans[q]);
    box.append(label);
  }
  const note = document.createElement("div"); note.className = "muted small qnote";
  box.append(note);

  let ctx = {}, seq = 0;
  const minutes = (s) => {
    if (s < 45) return "menos de 1 min";
    const m = Math.round(s / 60);
    return m >= 60 ? `~${Math.floor(m / 60)} h ${m % 60} min` : `~${m} min`;
  };
  const clock = (sec) => new Date(Date.now() + sec * 1000).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

  async function refresh() {
    const my = ++seq;
    const qs = new URLSearchParams();
    if (ctx.url) qs.set("url", ctx.url);
    if (ctx.duration) qs.set("duration", String(ctx.duration));
    if (ctx.url) for (const q in spans) spans[q].textContent = "· calculando…";
    let e;
    try {
      const r = await fetch(`${base}/api/estimate?${qs}`);
      if (!r.ok) return;
      e = await r.json();
    } catch { return; }
    if (my !== seq) return;                       // chegou uma resposta mais nova
    for (const q in spans) {
      const f = e[q];
      if (!f) { spans[q].textContent = ""; continue; }
      const fila = f.wait_seconds >= 60 ? `, ${minutes(f.wait_seconds)} de fila` : "";
      spans[q].textContent = `· pronta em ${minutes(f.seconds)} (${clock(f.seconds)}${fila})`;
    }
    const base_ = e.duration ? `música de ${mmss(e.duration)}` : "música de ~4 min (duração ainda desconhecida)";
    const calib = e.standard && e.standard.calibrated ? "" : " · estimativa inicial";
    note.textContent = `Previsão para ${base_}${calib}.`;
  }

  setInterval(() => { if (!document.hidden) refresh(); }, 20000);   // a fila anda
  refresh();
  return {
    get value() { return box.querySelector("input[name=quality]:checked").value; },
    set value(q) { const i = box.querySelector(`input[name=quality][value=${q === "hifi" ? "hifi" : "standard"}]`); if (i) i.checked = true; },
    forDuration(sec) { ctx = sec > 0 ? { duration: sec } : {}; refresh(); },
    reset() { ctx = {}; refresh(); },
    forUrl(url) { ctx = /^https?:\/\/\S+/.test(url || "") ? { url: url.trim() } : {}; refresh(); },
    forFile(file) {
      ctx = {};
      const a = new Audio(), src = URL.createObjectURL(file);
      a.preload = "metadata";
      a.onloadedmetadata = () => { URL.revokeObjectURL(src); if (isFinite(a.duration)) ctx = { duration: a.duration }; refresh(); };
      a.onerror = () => { URL.revokeObjectURL(src); refresh(); };   // formato que o navegador não lê
      a.src = src;
    },
  };
}
