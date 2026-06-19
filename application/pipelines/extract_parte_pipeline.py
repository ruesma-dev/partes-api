# application/pipelines/extract_parte_pipeline.py
"""Pipeline de extraccion de un parte de trabajo (sv2, una fase).

Valida el adjunto, ejecuta el proveedor LLM configurado y devuelve un
envelope ``{meta, data, debug}`` con la misma forma que el sv2 de
albaranes, para que sv3 lo consuma de forma homogenea.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from application.services.parte_extraction_service import (
    ParteExtractionService,
    ProviderExtractionResult,
)
from application.services.auxhor_catalog_provider import AuxhorCatalogProvider
from domain.models.llm_attachment import LlmAttachment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractParteRequest:
    filename: str
    mime_type: str
    file_bytes: bytes


class ExtractPartePipeline:
    def __init__(
        self,
        *,
        extraction_service: ParteExtractionService,
        max_file_mb: int,
        service_version: str,
        provider: str,
        prompt_key: str,
        catalog_provider: AuxhorCatalogProvider | None = None,
    ) -> None:
        self._service = extraction_service
        self._max_file_mb = max_file_mb
        self._service_version = service_version
        self._provider = provider
        self._prompt_key = prompt_key
        self._catalog_provider = catalog_provider

    def run(self, request: ExtractParteRequest) -> Dict[str, Any]:
        attachment = self._build_attachment_validated(
            filename=request.filename,
            mime_type=request.mime_type,
            file_bytes=request.file_bytes,
        )
        dynamic_context = None
        if self._catalog_provider is not None and self._catalog_provider.enabled:
            block = self._catalog_provider.prompt_block()
            dynamic_context = block or None
        result = self._service.extract(
            attachment=attachment,
            provider=self._provider,
            prompt_key=self._prompt_key,
            dynamic_context=dynamic_context,
        )
        sha256 = hashlib.sha256(request.file_bytes).hexdigest()
        return self._envelope(
            provider_result=result,
            attachment=attachment,
            sha256=sha256,
        )

    # --------------------------------------------------------------- #
    # Helpers.
    # --------------------------------------------------------------- #
    def _build_attachment_validated(
        self,
        *,
        filename: str,
        mime_type: str,
        file_bytes: bytes,
    ) -> LlmAttachment:
        if not file_bytes:
            raise ValueError("Archivo vacio.")
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self._max_file_mb:
            raise ValueError(
                f"Archivo demasiado grande ({size_mb:.2f} MB) > MAX_FILE_MB={self._max_file_mb}"
            )
        return self._build_attachment(filename, mime_type, file_bytes)

    @staticmethod
    def _build_attachment(filename: str, mime_type: str, file_bytes: bytes) -> LlmAttachment:
        filename = filename or "document.bin"
        is_pdf = mime_type == "application/pdf" or filename.lower().endswith(".pdf")
        if is_pdf:
            return LlmAttachment(
                kind="pdf",
                filename=filename,
                mime_type="application/pdf",
                data=file_bytes,
            )
        guessed_mime, _ = mimetypes.guess_type(filename)
        final_mime = mime_type or guessed_mime or "image/jpeg"
        return LlmAttachment(
            kind="image",
            filename=filename,
            mime_type=final_mime,
            data=file_bytes,
        )

    @staticmethod
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _envelope(
        self,
        *,
        provider_result: ProviderExtractionResult,
        attachment: LlmAttachment,
        sha256: str,
    ) -> Dict[str, Any]:
        meta = {
            "phase": "extract",
            "prompt_key": provider_result.prompt_key,
            "schema": provider_result.schema_name,
            "provider": provider_result.provider,
            "model": provider_result.model_name,
            "source_filename": attachment.filename,
            "source_mime_type": attachment.mime_type,
            "source_sha256": sha256,
            "processed_at_utc": self._utc_iso(),
            "service": "partes-extractor-api",
            "service_version": self._service_version,
        }
        return {
            "meta": meta,
            "data": provider_result.parsed.model_dump(),
            "debug": provider_result.debug_payload,
        }
