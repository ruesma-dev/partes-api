# infrastructure/llm/llm_call_logger.py
"""Logger best-effort de llamadas a LLM para tuning de prompts.

Si ``base_dir`` es None (IA_LOGGING_ENABLED=false), todas las llamadas a
``log_call`` son no-op (cero I/O). Si se activa, escribe un JSON por
llamada en ``<base_dir>/<YYYYMMDD>/<HHMMSS_micros>_<provider>_<n>.json``
con el resumen del request + la respuesta. Nunca incluye los bytes
binarios del documento (solo metadatos + sha256) ni claves API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LlmCallLogger:
    def __init__(self, base_dir: str | Path | None) -> None:
        self._base_dir = Path(base_dir) if base_dir else None
        self._lock = threading.Lock()
        self._counter = 0

    @property
    def enabled(self) -> bool:
        return self._base_dir is not None

    @staticmethod
    def attachment_summary(
        *,
        kind: str,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def log_call(
        self,
        *,
        provider: str,
        model: str,
        request_summary: dict,
        response_payload: Any,
        error: str | None,
    ) -> None:
        if self._base_dir is None:
            return
        now = datetime.now(timezone.utc)
        day_dir = self._base_dir / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._counter += 1
            n = self._counter
        stamp = now.strftime("%H%M%S_%f")
        path = day_dir / f"{stamp}_{provider}_{n}.json"
        payload = {
            "provider": provider,
            "model": model,
            "logged_at_utc": now.isoformat(),
            "request": request_summary,
            "response": self._jsonable(response_payload),
            "error": error,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)
