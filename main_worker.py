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
from infrastructure.azure.blob_cliente import BlobCliente
from infrastructure.azure.cola_cliente import ColaCliente
from infrastructure.azure.credenciales import build_credential
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

    cred = build_credential()
    cola = ColaCliente(
        os.environ["COLAS_ACCOUNT_URL"], cred,
        visibility_timeout_s=int(os.getenv("COLA_VISIBILITY_S", "600")),
    )
    blob = BlobCliente(os.environ["BLOBS_ACCOUNT_URL"], cred)
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
