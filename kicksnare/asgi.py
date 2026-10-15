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

# GrooveBass (baixo) roda no mesmo processo quando o pacote está instalado na imagem
try:
    import groovebass  # noqa: F401
except ModuleNotFoundError:
    groovebass = None
if groovebass is not None and BASE:
    from groovebass.web import BASE as GB_BASE, app as gb_app   # erro aqui deve aparecer, não sumir
    application.mount(GB_BASE, gb_app)
