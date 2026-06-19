# infrastructure/llm/claude_messages_client.py
from __future__ import annotations

import base64
import json
import logging
import re
import time
import traceback
from typing import Any, Type

import anthropic
from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient
from infrastructure.llm.llm_call_logger import LlmCallLogger
from infrastructure.llm.retry_policy import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)

_TOOL_NAME = "emit_albaran_extraction"
_TOOL_DESCRIPTION = (
    "Devuelve la extracción estructurada del albarán/factura "
    "conforme al esquema exigido. Debes llamar SIEMPRE a esta "
    "herramienta y solo a ella."
)


class ClaudeMessagesVisionClient(LlmVisionClient):
    def __init__(
        self,
        *,
        api_key: str,
        max_tokens: int = 8192,
        timeout_s: int = 120,
        retry_policy: RetryPolicy | None = None,
        call_logger: LlmCallLogger | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=float(timeout_s),
        )
        self._max_tokens = int(max_tokens)
        self._retry_policy = retry_policy or RetryPolicy()
        # Logger best-effort (puede ser None si IA_LOGGING_ENABLED=false).
        self._call_logger = call_logger

    @staticmethod
    def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return schema
        cleaned = {k: v for k, v in schema.items() if k != "title"}
        if "properties" in cleaned and isinstance(cleaned["properties"], dict):
            cleaned["properties"] = {
                key: ClaudeMessagesVisionClient._sanitize_schema(value)
                for key, value in cleaned["properties"].items()
            }
        if "$defs" in cleaned and isinstance(cleaned["$defs"], dict):
            cleaned["$defs"] = {
                key: ClaudeMessagesVisionClient._sanitize_schema(value)
                for key, value in cleaned["$defs"].items()
            }
        if "items" in cleaned and isinstance(cleaned["items"], dict):
            cleaned["items"] = ClaudeMessagesVisionClient._sanitize_schema(
                cleaned["items"]
            )
        return cleaned

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _safe_filename(filename: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "").strip("_")
        return cleaned or fallback

    def _build_content_block(
        self,
        *,
        attachment: LlmAttachment,
        user_text: str,
    ) -> list[dict[str, Any]]:
        if attachment.kind == "pdf":
            document_block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": self._b64(attachment.data),
                },
            }
            return [
                document_block,
                {"type": "text", "text": user_text},
            ]

        image_block: dict[str, Any] = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": attachment.mime_type or "image/jpeg",
                "data": self._b64(attachment.data),
            },
        }
        return [
            image_block,
            {"type": "text", "text": user_text},
        ]

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
            "Claude call. model=%s kind=%s filename=%s mime=%s "
            "size=%s schema=%s",
            model,
            attachment.kind,
            attachment.filename,
            attachment.mime_type,
            len(attachment.data),
            response_model.__name__,
        )

        raw_schema = response_model.model_json_schema()
        input_schema = self._sanitize_schema(raw_schema)

        tool_spec = {
            "name": _TOOL_NAME,
            "description": _TOOL_DESCRIPTION,
            "input_schema": input_schema,
        }

        content = self._build_content_block(
            attachment=attachment,
            user_text=user_text,
        )

        # Resumen del request (sin bytes binarios) para call_logger.
        request_summary = {
            "model": model,
            "instructions": instructions,
            "user_text": user_text,
            "attachment": LlmCallLogger.attachment_summary(
                kind=attachment.kind,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                data=attachment.data,
            ),
            "schema_name": response_model.__name__,
            "tool_name": _TOOL_NAME,
            "max_tokens": self._max_tokens,
            "response_format": "tool_use forced",
        }

        response = None
        error_str: str | None = None
        t0 = time.time()
        try:
            response = run_with_retry(
                provider="claude",
                operation=lambda: self._client.messages.create(
                    model=model,
                    max_tokens=self._max_tokens,
                    system=instructions,
                    tools=[tool_spec],
                    tool_choice={"type": "tool", "name": _TOOL_NAME},
                    messages=[{"role": "user", "content": content}],
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
        if self._call_logger is None:
            return
        try:
            request_summary["duration_ms"] = duration_ms
            self._call_logger.log_call(
                provider="claude",
                model=model,
                request_summary=request_summary,
                response_payload=response_payload,
                error=error,
            )
        except Exception:
            logger.warning(
                "ClaudeMessagesVisionClient: error guardando call log; "
                "se continúa.",
                exc_info=True,
            )

    @classmethod
    def _parse_response(
        cls,
        response: Any,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        for block in response.content or []:
            block_type = getattr(block, "type", None)
            if block_type != "tool_use":
                continue
            tool_name = getattr(block, "name", None)
            if tool_name != _TOOL_NAME:
                continue
            tool_input = getattr(block, "input", None)
            if isinstance(tool_input, dict):
                return response_model.model_validate(tool_input)
            if isinstance(tool_input, str):
                return response_model.model_validate_json(tool_input)

        # Fallback: por si el modelo devolvió texto JSON sin usar el tool.
        text_parts: list[str] = []
        for block in response.content or []:
            if getattr(block, "type", None) == "text":
                text_value = getattr(block, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)
        if text_parts:
            joined = "\n".join(text_parts).strip()
            try:
                payload = json.loads(cls._strip_code_fences(joined))
                if isinstance(payload, dict):
                    return response_model.model_validate(payload)
            except Exception as exc:
                logger.debug(
                    "Claude devolvió texto no parseable como JSON: %s", exc
                )

        raise ValueError(
            "Claude no devolvió tool_use ni JSON válido en la respuesta."
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned
