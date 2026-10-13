FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KICKSNARE_DATA=/data \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch só-CPU: o wheel padrão do PyPI no Linux inclui CUDA (~2,5 GB) e aqui não há GPU
RUN pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
EXPOSE 8765
# web por padrão; o worker usa:  python -m kicksnare.worker
CMD ["uvicorn", "kicksnare.asgi:application", "--host", "0.0.0.0", "--port", "8765"]
