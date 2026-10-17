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

### Padrão e HiFi em paralelo

Cada música escolhe a qualidade no envio: Padrão (htdemucs) ou HiFi (htdemucs_ft, ~4,9x a CPU). Para
as duas filas andarem ao mesmo tempo, rode um worker por qualidade e avise o web
(`KICKSNARE_SPLIT_WORKERS=1` no `kicksnare.env`, usado pela previsão de tempo):

```bash
docker run -d --name kicksnare-worker ... -e KICKSNARE_WORKER_QUALITY=standard \
  --memory 3g --memory-swap 3g --cpus 2 --cpu-shares 256 kicksnare:latest python -m kicksnare.worker
docker run -d --name kicksnare-worker-hifi ... -e KICKSNARE_WORKER_QUALITY=hifi \
  --memory 3g --memory-swap 3g --cpus 2 --cpu-shares 64 kicksnare:latest python -m kicksnare.worker
```

O `--cpu-shares` menor faz o HiFi usar só a CPU que sobra: o padrão e os outros serviços do servidor
quase não sentem, e o HiFi fica mais lento quando há disputa. Memória: pico medido do Demucs HiFi de
~2,25 GB (padrão: ~1,6 GB), dentro dos 3 GB de cada worker. Sem o worker HiFi, deixe
`KICKSNARE_SPLIT_WORKERS` desligado: um worker só (`all`) atende as duas filas.

No nginx, aponte `/kicksnare/` para `kicksnare-web:8765` **sem reescrever o caminho** (o app já monta
tudo sob `KICKSNARE_BASE`). Use `resolver 127.0.0.11` + variável no `proxy_pass` para que o nginx não
deixe de subir se o container estiver parado.
