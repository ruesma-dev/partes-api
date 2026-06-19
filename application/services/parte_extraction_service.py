# application/services/parte_extraction_service.py
"""Servicio de extraccion de partes de trabajo (sv2).

Una sola fase: ejecuta UN proveedor LLM contra el documento adjunto para
producir un ``ParteTrabajo`` conforme al schema. Que proveedor se usa lo
decide la capa superior (app.py) leyendo ``IA_PROVIDER`` del .env.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Type

from pydantic import BaseModel

from application.services.schema_registry import SchemaRegistry
from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient
from domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderClientSpec:
    provider: str
    model_name: str
    client: LlmVisionClient


@dataclass(frozen=True)
class ProviderExtractionResult:
    provider: str
    model_name: str
    schema_name: str
    prompt_key: str
    parsed: BaseModel
    debug_payload: Dict[str, Any]


class ParteExtractionService:
    def __init__(
        self,
        *,
        providers: Iterable[ProviderClientSpec],
        prompt_repo: PromptRepository,
        schema_registry: SchemaRegistry,
    ) -> None:
        self._providers_by_name: Dict[str, ProviderClientSpec] = {
            spec.provider: spec for spec in providers
        }
        self._prompts = prompt_repo
        self._schemas = schema_registry

    def extract(
        self,
        *,
        attachment: LlmAttachment,
        provider: str,
        prompt_key: str,
        dynamic_context: str | None = None,
    ) -> ProviderExtractionResult:
        spec = self._require_provider(provider)
        prompt_spec = self._prompts.get(prompt_key)
        response_model = self._schemas.get(prompt_spec.schema)

        instructions = self._compose_instructions(
            system=prompt_spec.system,
            task=prompt_spec.task,
            schema_hint=prompt_spec.schema_hint,
            dynamic_context=dynamic_context,
        )
        user_text = (
            "Documento adjunto: parte de trabajo. Extrae los datos siguiendo "
            "las reglas del prompt. Devuelve SOLO JSON valido conforme al schema."
        )

        logger.info(
            "Extraccion parte proveedor=%s prompt_key=%s schema=%s model=%s filename=%s",
            spec.provider,
            prompt_key,
            prompt_spec.schema,
            spec.model_name,
            attachment.filename,
        )

        parsed = spec.client.extract_document(
            model=spec.model_name,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            response_model=response_model,
        )

        debug_payload: Dict[str, Any] = {
            f"{spec.provider}_request": {
                "provider": spec.provider,
                "model": spec.model_name,
                "prompt_key": prompt_key,
                "attachment": self._attachment_debug(attachment),
            },
            f"{spec.provider}_response": {
                "schema": prompt_spec.schema,
                "parsed_keys": (
                    list(parsed.model_dump().keys())
                    if hasattr(parsed, "model_dump") else []
                ),
            },
        }

        return ProviderExtractionResult(
            provider=spec.provider,
            model_name=spec.model_name,
            schema_name=prompt_spec.schema,
            prompt_key=prompt_key,
            parsed=parsed,
            debug_payload=debug_payload,
        )

    # --------------------------------------------------------------- #
    # Helpers.
    # --------------------------------------------------------------- #
    def _require_provider(self, name: str) -> ProviderClientSpec:
        spec = self._providers_by_name.get(name)
        if spec is None:
            available = ", ".join(sorted(self._providers_by_name.keys()))
            raise KeyError(
                f"Proveedor LLM '{name}' no instanciado. Disponibles: {available}. "
                f"Revisa los flags ENABLE_* del .env."
            )
        return spec

    @staticmethod
    def _compose_instructions(
        *,
        system: str | None,
        task: str | None,
        schema_hint: str | None,
        dynamic_context: str | None = None,
    ) -> str:
        parts = [system, task, schema_hint, dynamic_context]
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _attachment_debug(attachment: LlmAttachment) -> Dict[str, Any]:
        return {
            "kind": attachment.kind,
            "filename": attachment.filename,
            "mime_type": attachment.mime_type,
            "size_bytes": len(attachment.data),
            "sha256": hashlib.sha256(attachment.data).hexdigest(),
        }
