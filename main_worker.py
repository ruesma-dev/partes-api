# main_worker.py
"""Worker de sv2 (extraccion).

Consume 'q-extraccion', lee el PDF de blob 'input/{document_id}.pdf', extrae
con el pipeline (Gemini primario), escribe el envelope en
'envelopes/{document_id}.json' y encola 'q-persistencia'. Auth a cola/blob por
managed identity. Reutiliza el wiring de build_app via app.state.pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from application.pipelines.extract_parte_pipeline import ExtractParteRequest
from config.logging_config import configure_logging
from config.settings import Settings
from infrastructure.azure.credenciales import (
    construir_blob_cliente,
    construir_cola_cliente,
)
from interface_adapters.api.app import build_app

logger = logging.getLogger(__name__)

COLA_ENTRADA = os.getenv("COLA_EXTRACCION", "q-extraccion")
COLA_SALIDA = os.getenv("COLA_PERSISTENCIA", "q-persistencia")
CONTENEDOR_INPUT = os.getenv("BLOB_INPUT", "input")
CONTENEDOR_ENVELOPES = os.getenv("BLOB_ENVELOPES", "envelopes")


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)
    logger.info("[sv2-worker] arrancando. %s -> %s", COLA_ENTRADA, COLA_SALIDA)

    colas_cs = settings.colas_connection_string
    colas_url = settings.colas_account_url or os.getenv("COLAS_ACCOUNT_URL")
    if not colas_cs and not colas_url:
        logger.error(
            "[sv2-worker] falta storage: define COLAS_CONNECTION_STRING "
            "(local/Azurite) o COLAS_ACCOUNT_URL (nube) en el .env / "
            "Container App."
        )
        return 1

    cola = construir_cola_cliente(
        connection_string=colas_cs, account_url=colas_url,
        visibility_timeout_s=int(os.getenv("COLA_VISIBILITY_S", "600")),
    )
    blob = construir_blob_cliente(
        connection_string=settings.blobs_connection_string,
        account_url=settings.blobs_account_url
        or os.getenv("BLOBS_ACCOUNT_URL"),
        colas_connection_string=colas_cs,
    )
    if colas_cs:
        # Modo local (Azurite arranca vacio): asegura colas y contenedores.
        cola.asegurar_colas([COLA_ENTRADA, COLA_SALIDA])
        blob.asegurar_contenedores([CONTENEDOR_INPUT, CONTENEDOR_ENVELOPES])
    pipeline = build_app(settings).state.pipeline

    def handler(payload: dict) -> None:
        document_id = payload["document_id"]
        filename = payload.get("filename", "document.pdf")
        mime = payload.get("mime_type", "application/pdf")
        pdf = blob.descargar(CONTENEDOR_INPUT, f"{document_id}.pdf")
        envelope = pipeline.run(ExtractParteRequest(
            filename=filename, mime_type=mime, file_bytes=pdf))
        blob.subir(
            CONTENEDOR_ENVELOPES, f"{document_id}.json",
            json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
            "application/json")
        cola.enviar(COLA_SALIDA, {
            "document_id": document_id, "filename": filename,
            "mime_type": mime, "context": payload.get("context", {}),
        })

    cola.consumir(COLA_ENTRADA, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
