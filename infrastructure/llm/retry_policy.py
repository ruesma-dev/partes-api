# infrastructure/llm/retry_policy.py
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Política de reintentos compartida por los 3 clientes LLM.

    Diseño:
      - ``max_retries`` = número de REINTENTOS tras el primer intento.
        Total de intentos = 1 + max_retries. Con el default (2), hacemos
        hasta 3 intentos antes de propagar el error.
      - Backoff exponencial con jitter: sleep = min(cap, base * 2^(n-1)) * (1 + jitter)
        donde jitter ∈ [0, 0.25]. Evita que N cliente sincronizados reintenten
        exactamente a la vez (problema de "thundering herd") si esto corre
        en paralelo.
      - Respeta ``Retry-After`` del servidor si lo obtenemos del error.
    """

    max_retries: int = 2
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 30.0


# ---------------------------------------------------------------------------
# Clasificación de errores
# ---------------------------------------------------------------------------

# HTTP status codes que merecen reintento. Cualquier otro 4xx (401, 403, 422...)
# es un problema de configuración/prompt que reintentar no arregla y gasta
# tokens y tiempo.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Excepciones de transporte httpx que típicamente indican "el socket se
# colgó pero la petición puede ser segura de reintentar".
_RETRYABLE_HTTPX_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# Nombres de clase de excepciones que SDKs (openai, anthropic, google-genai)
# usan para envolver errores de red/5xx. Los comprobamos por nombre porque
# no queremos imports duros de cada SDK aquí (el módulo retry_policy debe
# ser agnóstico al proveedor).
_RETRYABLE_EXC_CLASSNAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "RateLimitError",
    # Anthropic SDK
    "APIStatusError",  # solo si el status es 5xx/429 (filtramos abajo)
    # google-genai
    "ServerError",
    "DeadlineExceeded",
    "UnavailableError",
}

# Textos que aparecen en mensajes de error de "conexión rota" y que
# justifican un reintento aunque el tipo de excepción no lo sea.
_RETRYABLE_ERROR_SUBSTRINGS = (
    "server disconnected",
    "connection aborted",
    "connection reset",
    "broken pipe",
    "eof occurred",
    "temporarily unavailable",
)


def _extract_status_code(exc: BaseException) -> int | None:
    """Saca el HTTP status de un error si el SDK lo expone."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # Algunos SDKs guardan la respuesta en ``exc.response``
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _extract_retry_after(exc: BaseException) -> float | None:
    """Saca el valor de ``Retry-After`` (en segundos) si viene en el error."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    """Decide si este error concreto merece un reintento."""
    # 1) Transportes httpx conocidos
    if isinstance(exc, _RETRYABLE_HTTPX_EXCEPTIONS):
        return True

    # 2) Status code si lo podemos extraer
    status = _extract_status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES

    # 3) Clase del SDK por nombre (sin tener que importar cada SDK aquí).
    #    Subimos por el MRO para capturar subclases.
    for cls in type(exc).__mro__:
        if cls.__name__ in _RETRYABLE_EXC_CLASSNAMES:
            return True

    # 4) Como último recurso, buscamos patrones de "conexión rota" en el
    #    mensaje del error. Cubre el caso exacto del log:
    #    "Server disconnected without sending a response."
    message = str(exc).lower()
    for needle in _RETRYABLE_ERROR_SUBSTRINGS:
        if needle in message:
            return True

    return False


def _compute_sleep(
    *,
    attempt: int,
    policy: RetryPolicy,
    retry_after: float | None,
) -> float:
    """Cuánto dormir antes del siguiente intento (en segundos)."""
    exponential = policy.backoff_base_s * (2 ** (attempt - 1))
    exponential = min(exponential, policy.backoff_cap_s)
    jitter = random.uniform(0.0, 0.25)
    sleep_s = exponential * (1 + jitter)
    # Si el servidor pide esperar más tiempo concreto, respetamos su tiempo
    # (pero topado por cap_s para no colgar demasiado).
    if retry_after is not None:
        sleep_s = max(sleep_s, min(retry_after, policy.backoff_cap_s))
    return sleep_s


def run_with_retry(
    *,
    provider: str,
    operation: Callable[[], T],
    policy: RetryPolicy,
) -> T:
    """Ejecuta ``operation()`` aplicando la política de reintentos.

    Args:
        provider: nombre corto del proveedor (openai/gemini/claude), solo
            para logs.
        operation: callable sin argumentos que hace la llamada real al LLM.
            Diseño funcional: el cliente construye un lambda capturando
            todo lo que necesita y lo pasa aquí.
        policy: parámetros de la política.

    Returns:
        El valor devuelto por ``operation`` tras un intento exitoso.

    Raises:
        La última excepción si se agotan los reintentos o si el error no
        es retryable (en cuyo caso NO se reintenta).
    """
    total_attempts = 1 + policy.max_retries
    last_exc: BaseException | None = None

    for attempt in range(1, total_attempts + 1):
        try:
            if attempt > 1:
                logger.info(
                    "[llm-retry] %s intento %s/%s",
                    provider,
                    attempt,
                    total_attempts,
                )
            return operation()
        except BaseException as exc:  # incluye KeyboardInterrupt, pero abajo filtramos
            # No reintentamos si el usuario pulsó Ctrl-C o si el sistema
            # quiere tumbar el proceso.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            last_exc = exc
            retryable = _is_retryable(exc)
            is_last_attempt = attempt >= total_attempts

            if not retryable:
                logger.warning(
                    "[llm-retry] %s error NO retryable. type=%s msg=%s",
                    provider,
                    type(exc).__name__,
                    str(exc)[:300],
                )
                raise

            if is_last_attempt:
                logger.error(
                    "[llm-retry] %s agotados reintentos (%s). "
                    "Último error: type=%s msg=%s",
                    provider,
                    total_attempts,
                    type(exc).__name__,
                    str(exc)[:300],
                )
                raise

            retry_after = _extract_retry_after(exc)
            sleep_s = _compute_sleep(
                attempt=attempt,
                policy=policy,
                retry_after=retry_after,
            )
            logger.warning(
                "[llm-retry] %s intento %s/%s falló con error retryable "
                "(type=%s msg=%s retry_after=%s). Esperando %.2fs…",
                provider,
                attempt,
                total_attempts,
                type(exc).__name__,
                str(exc)[:200],
                retry_after,
                sleep_s,
            )
            time.sleep(sleep_s)

    # Inalcanzable: el bucle siempre sale con ``return`` o ``raise``.
    assert last_exc is not None
    raise last_exc
