# Deploy com Docker (recomendado em servidor compartilhado)

Duas instâncias da mesma imagem (web e worker) compartilham um volume de dados. O worker (Demucs)
recebe o limite de memória/CPU: se estourar, cai só ele.

```bash
docker build -t kicksnare:latest .
docker volume create kicksnare-data
cat > kicksnare.env <<'ENV'
KICKSNARE_BASE=/kicksnare
KICKSNARE_ADMIN_USER=<seu usuário>
KICKSNARE_ADMIN_PASSWORD=<defina>
ENV
chmod 600 kicksnare.env

docker run -d --name kicksnare-web --restart unless-stopped --network <rede> \
  --env-file kicksnare.env -v kicksnare-data:/data \
  --memory 512m --memory-swap 512m --cpus 0.5 kicksnare:latest

docker run -d --name kicksnare-worker --restart unless-stopped --network <rede> \
  --env-file kicksnare.env -v kicksnare-data:/data \
  --memory 3g --memory-swap 3g --cpus 2 --cpu-shares 256 \
  kicksnare:latest python -m kicksnare.worker
```

No nginx, aponte `/kicksnare/` para `kicksnare-web:8765` **sem reescrever o caminho** (o app já monta
tudo sob `KICKSNARE_BASE`). Use `resolver 127.0.0.11` + variável no `proxy_pass` para que o nginx não
deixe de subir se o container estiver parado.
