# infrastructure/llm/gemini_genai_client.py
from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient
from infrastructure.llm.llm_call_logger import LlmCallLogger
from infrastructure.llm.retry_policy import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)


class GeminiGenAiVisionClient(LlmVisionClient):
    def __init__(
        self,
        api_key: str,
        *,
        media_resolution: str | None = None,
        retry_policy: RetryPolicy | None = None,
        call_logger: LlmCallLogger | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._retry_policy = retry_policy or RetryPolicy()
        # Logger best-effort (puede ser None si IA_LOGGING_ENABLED=false).
        self._call_logger = call_logger
        # Resolucion de medios (Gemini 3); None = no forzar (default del SDK).
        self._media_resolution = self._resolve_media_resolution(media_resolution)

    @staticmethod
    def _resolve_media_resolution(value: str | None) -> Any | None:
        """Mapea un string de configuracion a types.MediaResolution.

        Devuelve None (no forzar resolucion) para "default"/"unspecified"/
        vacio, o si el SDK instalado no soporta el parametro. Asi el
        comportamiento por defecto no cambia ni consume tokens extra.
        """
        token = (value or "").strip().lower()
        if token in ("", "default", "unspecified",
                     "media_resolution_unspecified"):
            return None
        enum_cls = getattr(types, "MediaResolution", None)
        cfg_cls = getattr(types, "GenerateContentConfig", None)
        supported = (
            enum_cls is not None
            and cfg_cls is not None
            and "media_resolution" in getattr(cfg_cls, "model_fields", {})
        )
        if not supported:
            logger.warning(
                "GEMINI_MEDIA_RESOLUTION=%s ignorado: el SDK google-genai "
                "instalado no soporta el parametro media_resolution.",
                value,
            )
            return None
        mapping = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "med": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
            "media_resolution_low": "MEDIA_RESOLUTION_LOW",
            "media_resolution_medium": "MEDIA_RESOLUTION_MEDIUM",
            "media_resolution_high": "MEDIA_RESOLUTION_HIGH",
        }
        member = mapping.get(token)
        resolved = getattr(enum_cls, member, None) if member else None
        if resolved is None:
            logger.warning(
                "GEMINI_MEDIA_RESOLUTION=%s no reconocido; se usa el valor "
                "por defecto del SDK.",
                value,
            )
            return None
        logger.info("Gemini media_resolution=%s", member)
        return resolved

    def extract_document(
        self,
        *,
        model: str,
        instructions: str,
        user_text: str,
        attachment: LlmAttachment,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        logger.info(
            "Gemini call. model=%s kind=%s filename=%s mime=%s size=%s schema=%s",
            model,
            attachment.kind,
            attachment.filename,
            attachment.mime_type,
            len(attachment.data),
            response_model.__name__,
        )

        response_schema = response_model.model_json_schema()

        # ----- Captura del request para el call_logger ----- #
        request_summary = self._build_request_summary(
            model=model,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            schema_name=response_model.__name__,
            response_schema=response_schema,
        )

        # Orden recomendado por Gemini para 1 documento + texto: primero el
        # texto, luego el adjunto.
        contents = [
            user_text,
            types.Part.from_bytes(
                data=attachment.data,
                mime_type=attachment.mime_type,
            ),
        ]
        config_kwargs: dict[str, Any] = {
            "system_instruction": instructions,
            "response_mime_type": "application/json",
            "response_json_schema": response_schema,
        }
        if self._media_resolution is not None:
            config_kwargs["media_resolution"] = self._media_resolution
        config = types.GenerateContentConfig(**config_kwargs)

        response = None
        error_str: str | None = None
        t0 = time.time()
        try:
            response = run_with_retry(
                provider="gemini",
                operation=lambda: self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                policy=self._retry_policy,
            )
        except Exception as exc:
            error_str = (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=3)}"
            )
            self._safe_log_call(
                model=model,
                request_summary=request_summary,
                response_payload=None,
                error=error_str,
                duration_ms=int((time.time() - t0) * 1000),
            )
            raise

        duration_ms = int((time.time() - t0) * 1000)

        # Intentamos parsear ANTES de loguear, para que el log refleje
        # el resultado real (parsed exitoso o no). Si el parseo falla,
        # también lo logueamos.
        try:
            parsed_obj = self._parse_response(response, response_model)
        except Exception as exc:
            error_str = f"PARSE ERROR: {type(exc).__name__}: {exc}"
            self._safe_log_call(
                model=model,
                request_summary=request_summary,
                response_payload=response,
                error=error_str,
                duration_ms=duration_ms,
            )
            raise

        # Log OK con response del SDK + objeto parseado (lo más útil).
        self._safe_log_call(
            model=model,
            request_summary=request_summary,
            response_payload={
                "raw_sdk_response": response,
                "parsed_pydantic": (
                    parsed_obj.model_dump(mode="json")
                    if isinstance(parsed_obj, BaseModel) else None
                ),
                "duration_ms": duration_ms,
            },
            error=None,
            duration_ms=duration_ms,
        )
        return parsed_obj

    # ------------------------------------------------------------ #
    # Helpers internos.
    # ------------------------------------------------------------ #
    def _safe_log_call(
        self,
        *,
        model: str,
        request_summary: dict,
        response_payload: Any,
        error: str | None,
        duration_ms: int,
    ) -> None:
        """Llama a call_logger.log_call() ignorando cualquier error."""
        if self._call_logger is None:
            return
        try:
            request_summary["duration_ms"] = duration_ms
            self._call_logger.log_call(
                provider="gemini",
                model=model,
                request_summary=request_summary,
                response_payload=response_payload,
                error=error,
            )
        except Exception:
            logger.warning(
                "GeminiGenAiVisionClient: error guardando call log; "
                "se continúa.",
                exc_info=True,
            )

    @staticmethod
    def _build_request_summary(
        *,
        model: str,
        instructions: str,
        user_text: str,
        attachment: LlmAttachment,
        schema_name: str,
        response_schema: dict,
    ) -> dict:
        return {
            "model": model,
            "instructions": instructions,
            "user_text": user_text,
            "attachment": LlmCallLogger.attachment_summary(
                kind=attachment.kind,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                data=attachment.data,
            ),
            "schema_name": schema_name,
            "response_format": "json_schema",
            "response_schema_keys": list((response_schema or {}).keys()),
        }

    def _parse_response(
        self,
        response: Any,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_model):
            return parsed
        if isinstance(parsed, BaseModel):
            return response_model.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return response_model.model_validate(parsed)

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            return response_model.model_validate_json(
                self._sanitize_json_text(response_text)
            )

        parsed_from_candidates = self._extract_json_dict(response)
        if parsed_from_candidates is not None:
            return response_model.model_validate(parsed_from_candidates)

        raise ValueError("Gemini no devolvió parsed ni text JSON válido.")

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned

    @classmethod
    def _extract_json_dict(cls, response: Any) -> dict[str, Any] | None:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    try:
                        loaded = json.loads(cls._sanitize_json_text(text))
                    except Exception:
                        continue
                    if isinstance(loaded, dict):
                        return loaded
        return None
