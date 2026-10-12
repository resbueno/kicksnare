"""Ponto de entrada para produção.

    KICKSNARE_BASE=/kicksnare uvicorn kicksnare.asgi:application --host 127.0.0.1 --port 8000

Com KICKSNARE_BASE o app inteiro é montado sob esse prefixo, então o proxy (nginx) repassa
o caminho sem reescrevê-lo. Sem a variável, roda na raiz.
"""
from fastapi import FastAPI

from .web import BASE, app

if BASE:
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.mount(BASE, app)
else:
    application = app
