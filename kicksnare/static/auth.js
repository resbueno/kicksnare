// Sessão: redireciona para o login quando a API responde 401 e mostra "usuário · Sair" no cabeçalho.
(() => {
  const base = document.currentScript.dataset.base || "";
  const orig = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const r = await orig(...args);
    const url = String(args[0] && args[0].url ? args[0].url : args[0]);
    // 401 em login/cadastro/troca de senha é "credencial errada" (a tela mostra o erro), não sessão expirada
    const credentialCall = ["/api/auth/login", "/api/auth/register", "/api/auth/password"].some((p) => url.includes(p));
    if (r.status === 401 && !credentialCall && !location.pathname.endsWith("/login")) {
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
    const who = document.createElement("span");
    who.className = "who"; who.textContent = me.username + (me.is_admin ? " (admin)" : "");
    const pw = document.createElement("a");
    pw.className = "pwlink"; pw.href = base + "/account"; pw.textContent = "Alterar senha";
    const out = document.createElement("button");
    out.type = "button"; out.textContent = "Sair";
    out.onclick = async () => {
      await orig(base + "/api/auth/logout", { method: "POST" });
      location.href = base + "/login";
    };
    box.append(who, pw, out);
    bar.append(box);
  });
})();
