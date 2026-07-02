# infrastructure/llm/openai_responses_client.py
from __future__ import annotations

import base64
import json
import logging
import re
import time
import traceback
from typing import Any, Type

import openai
from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient
from infrastructure.llm.llm_call_logger import LlmCallLogger
from infrastructure.llm.retry_policy import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)


class OpenAiResponsesVisionClient(LlmVisionClient):
    """Adaptador OpenAI (Responses API) para extraccion estructurada.

    Se usa la Responses API porque admite PDF e imagen como entrada nativa
    y ofrece structured outputs. Estrategia de salida estructurada:

      1) Si el SDK expone ``responses.parse`` (lo normal en versiones
         recientes), se llama con ``text_format=<modelo Pydantic>``: el SDK
         genera un JSON Schema estricto a partir del modelo y devuelve la
         instancia ya parseada en ``output_parsed``.
      2) Si ``parse`` no existe, o si falla (p.ej. el schema del modelo no
         es compatible con el modo estricto, o el modelo no soporta
         structured outputs), se cae a ``responses.create`` en modo
         ``json_object`` y se parsea el texto con el modelo Pydantic.

    Igual que los clientes de Gemini/Claude: reintentos via run_with_retry,
    logging best-effort y parseo robusto de la respuesta.
    """

    def __init__(
        self,
        *,
        api_key: str,
        max_output_tokens: int | None = None,
        timeout_s: int = 120,
        temperature: float | None = None,
        retry_policy: RetryPolicy | None = None,
        call_logger: LlmCallLogger | None = None,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, timeout=float(timeout_s))
        self._max_output_tokens = (
            int(max_output_tokens) if max_output_tokens else None
        )
        # temperature opcional: por defecto None (no se envia), porque algunos
        # modelos de razonamiento no la aceptan.
        self._temperature = temperature
        self._retry_policy = retry_policy or RetryPolicy()
        # Logger best-effort (puede ser None si IA_LOGGING_ENABLED=false).
        self._call_logger = call_logger
        # structured outputs nativo si el SDK instalado lo soporta.
        self._supports_parse = hasattr(self._client.responses, "parse")

    # ------------------------------------------------------------ #
    # Helpers de construccion del request.
    # ------------------------------------------------------------ #
    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _safe_filename(filename: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "").strip("_")
        return cleaned or fallback

    def _build_input(
        self,
        *,
        attachment: LlmAttachment,
        user_text: str,
    ) -> list[dict[str, Any]]:
        # Orden: primero el texto, despues el adjunto.
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_text},
        ]
        b64 = self._b64(attachment.data)
        if attachment.kind == "pdf":
            filename = self._safe_filename(attachment.filename, "document.pdf")
            content.append(
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": f"data:application/pdf;base64,{b64}",
                }
            )
        else:
            mime = attachment.mime_type or "image/jpeg"
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                }
            )
        return [{"role": "user", "content": content}]

    def _optional_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return kwargs

    # ------------------------------------------------------------ #
    # API principal.
    # ------------------------------------------------------------ #
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
            "OpenAI call. model=%s kind=%s filename=%s mime=%s "
            "size=%s schema=%s parse=%s",
            model,
            attachment.kind,
            attachment.filename,
            attachment.mime_type,
            len(attachment.data),
            response_model.__name__,
            self._supports_parse,
        )

        input_messages = self._build_input(
            attachment=attachment,
            user_text=user_text,
        )
        opt = self._optional_kwargs()

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
            "max_output_tokens": self._max_output_tokens,
            "response_format": (
                "json_schema (parse)" if self._supports_parse else "json_object"
            ),
        }

        response = None
        used_fallback = False
        error_str: str | None = None
        t0 = time.time()
        try:
            if self._supports_parse:
                try:
                    response = run_with_retry(
                        provider="openai",
                        operation=lambda: self._client.responses.parse(
                            model=model,
                            instructions=instructions,
                            input=input_messages,
                            text_format=response_model,
                            **opt,
                        ),
                        policy=self._retry_policy,
                    )
                except Exception as parse_exc:
                    # El schema estricto puede no ser compatible o el modelo
                    # puede no soportar structured outputs: caemos a
                    # json_object, que el prompt ya pide explicitamente.
                    logger.warning(
                        "OpenAI responses.parse fallo (%s: %s). "
                        "Reintento en modo json_object.",
                        type(parse_exc).__name__,
                        str(parse_exc)[:200],
                    )
                    used_fallback = True
                    response = run_with_retry(
                        provider="openai",
                        operation=lambda: self._client.responses.create(
                            model=model,
                            instructions=instructions,
                            input=input_messages,
                            text={"format": {"type": "json_object"}},
                            **opt,
                        ),
                        policy=self._retry_policy,
                    )
            else:
                response = run_with_retry(
                    provider="openai",
                    operation=lambda: self._client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=input_messages,
                        text={"format": {"type": "json_object"}},
                        **opt,
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
        if used_fallback:
            request_summary["response_format"] = "json_object (fallback)"

        try:
            parsed_obj = self._parse_response(response, response_model)
        except Exception as exc:
            error_str = f"PARSE ERROR: {type(exc).__name__}: {exc}"
            self._safe_log_call(
                model=model,
                request_summary=request_summary,
                response_payload=self._response_debug(response),
                error=error_str,
                duration_ms=duration_ms,
            )
            raise

        self._safe_log_call(
            model=model,
            request_summary=request_summary,
            response_payload={
                "raw_output_text": getattr(response, "output_text", None),
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
                provider="openai",
                model=model,
                request_summary=request_summary,
                response_payload=response_payload,
                error=error,
            )
        except Exception:
            logger.warning(
                "OpenAiResponsesVisionClient: error guardando call log; "
                "se continua.",
                exc_info=True,
            )

    @classmethod
    def _parse_response(
        cls,
        response: Any,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        # 1) structured output ya parseado por el SDK (responses.parse).
        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, response_model):
            return parsed
        if isinstance(parsed, BaseModel):
            return response_model.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return response_model.model_validate(parsed)

        # 2) texto plano de conveniencia.
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return response_model.model_validate_json(
                cls._sanitize_json_text(text)
            )

        # 3) recorrido manual de los bloques de salida.
        payload = cls._extract_json_dict(response)
        if payload is not None:
            return response_model.model_validate(payload)

        raise ValueError(
            "OpenAI no devolvio output_parsed ni texto JSON valido."
        )

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
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

    @classmethod
    def _extract_json_dict(cls, response: Any) -> dict[str, Any] | None:
        outputs = getattr(response, "output", None) or []
        for item in outputs:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    try:
                        loaded = json.loads(cls._sanitize_json_text(text))
                    except Exception:
                        continue
                    if isinstance(loaded, dict):
                        return loaded
        return None

    @staticmethod
    def _response_debug(response: Any) -> dict[str, Any]:
        return {"output_text": getattr(response, "output_text", None)}
