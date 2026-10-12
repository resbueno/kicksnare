# Deploy (Ubuntu/Debian, nginx + systemd)

Pré-requisitos: Python 3.10–3.13, `ffmpeg`, `git`, nginx. Mínimo recomendado: 4 GB de RAM + swap de 4 GB
(o Demucs usa ~1,8 GB de pico) e 4 vCPUs. Sem GPU, conte com 5–10 min por música.

```bash
sudo useradd -r -m -d /opt/kicksnare -s /usr/sbin/nologin kicksnare
sudo mkdir -p /var/lib/kicksnare && sudo chown kicksnare: /var/lib/kicksnare
sudo -u kicksnare git clone https://github.com/<usuario>/kicksnare /opt/kicksnare
cd /opt/kicksnare && sudo -u kicksnare python3 -m venv .venv
sudo -u kicksnare .venv/bin/pip install -r requirements.txt   # baixa torch (~1 GB)

sudo cp deploy/kicksnare.env.example /etc/kicksnare.env && sudo chmod 600 /etc/kicksnare.env
sudo editor /etc/kicksnare.env                                   # defina a senha
sudo cp deploy/kicksnare-*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now kicksnare-web kicksnare-worker
# nginx: inclua deploy/nginx-kicksnare.conf no server { } e recarregue
sudo nginx -t && sudo systemctl reload nginx
```

Atualizar: `git pull`, `pip install -r requirements.txt`, `systemctl restart kicksnare-web kicksnare-worker`.

Notas
- O primeiro job baixa os pesos do Demucs e do beat_this (~160 MB) para o cache do usuário `kicksnare`.
- O YouTube costuma bloquear IPs de datacenter. Se o download falhar, use "Enviar arquivo de áudio".
- Dados em `KICKSNARE_DATA` (banco SQLite + uma pasta por música). Faça backup só se precisar das correções.
