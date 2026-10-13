// Sessão: redireciona para o login quando a API responde 401 e mostra "usuário · Sair" no cabeçalho.
(() => {
  const base = document.currentScript.dataset.base || "";
  const orig = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const r = await orig(...args);
    const url = String(args[0] && args[0].url ? args[0].url : args[0]);
    if (r.status === 401 && !url.includes("/api/auth/login") && !url.includes("/api/auth/register")
        && !location.pathname.endsWith("/login")) {
      location.href = base + "/login";
    }
    return r;
  };

  addEventListener("DOMContentLoaded", async () => {
    const bar = document.querySelector(".brandbar .in");
    if (!bar) return;
    let me;
    try {
      const r = await orig(base + "/api/auth/me");
      if (!r.ok) return;
      me = await r.json();
    } catch { return; }
    const box = document.createElement("div");
    box.className = "userbox";
    const who = document.createElement("a");
    who.href = base + "/account"; who.textContent = me.username; who.title = "Minha conta";
    const out = document.createElement("button");
    out.type = "button"; out.textContent = "Sair";
    out.onclick = async () => {
      await orig(base + "/api/auth/logout", { method: "POST" });
      location.href = base + "/login";
    };
    box.append(who, out);
    bar.append(box);
  });
})();
