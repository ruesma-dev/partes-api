# application/services/auxhor_catalog_provider.py
"""Provee el CATALOGO de codigos de hora (``auxhor``) de Sigrid, ya
FILTRADO, para inyectarlo en el prompt de la IA y que esta PROPONGA el
codigo adecuado por categoria + normal/extra + incidencia.

Reglas:
  - Solo codigos ACTIVOS (``fecbaj`` = 0/NULL; ya lo filtra el SQL).
  - Se EXCLUYE todo lo de maquinaria y becario antes de mandarlo a la IA
    (configurable por palabras clave / codigos).
  - Cache en memoria con TTL para no golpear Sigrid en cada extraccion.
  - Formatea un bloque de texto compacto: "COD - descripcion [extra]".

Si Sigrid no esta cableado o la consulta falla, devuelve un bloque vacio
y la IA no propondra codigo (sv3 cae a su resolucion por defecto).
"""
from __future__ import annotations

import logging
import time
import unicodedata

from infrastructure.sigrid.sigrid_lookup_client import (
    SigridLookupClient,
    TipoHoraOption,
)

logger = logging.getLogger(__name__)

# Palabras clave (en la descripcion) que marcan un codigo como maquinaria
# o becario, y que NO deben llegar a la IA.
_DEFAULT_EXCLUDE_KEYWORDS = (
    "MAQUINARIA",
    "MAQUINA",
    "BECARIO",
    "BECARIA",
)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _norm(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(_strip_accents(str(text)).upper().split())


class AuxhorCatalogProvider:
    def __init__(
        self,
        *,
        client: SigridLookupClient | None,
        exclude_keywords: tuple[str, ...] = _DEFAULT_EXCLUDE_KEYWORDS,
        exclude_codes: tuple[str, ...] = (),
        ttl_seconds: int = 600,
    ) -> None:
        self._client = client
        self._exclude_keywords = tuple(_norm(k) for k in exclude_keywords if k)
        self._exclude_codes = tuple(_norm(c) for c in exclude_codes if c)
        self._ttl = int(ttl_seconds)
        self._cache: list[TipoHoraOption] | None = None
        self._cache_ts = 0.0

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _is_excluded(self, opt: TipoHoraOption) -> bool:
        desc = _norm(opt.descripcion)
        cod = _norm(opt.codigo)
        if cod and cod in self._exclude_codes:
            return True
        for kw in self._exclude_keywords:
            if kw and (kw in desc or kw in cod):
                return True
        return False

    def _load(self) -> list[TipoHoraOption]:
        if self._client is None:
            return []
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._ttl:
            return self._cache
        try:
            rows = self._client.fetch_tipos_hora()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[auxhor-catalog] fallo al leer Sigrid: %r (catalogo vacio).",
                exc,
            )
            return self._cache or []
        filtered = [o for o in rows if not self._is_excluded(o)]
        self._cache = filtered
        self._cache_ts = now
        logger.info(
            "[auxhor-catalog] %s codigos (de %s; excluidos %s maq/becario).",
            len(filtered), len(rows), len(rows) - len(filtered),
        )
        return filtered

    def prompt_block(self) -> str:
        """Bloque de texto con el catalogo, listo para el prompt. Vacio si
        Sigrid no esta cableado o no hay codigos."""
        items = self._load()
        if not items:
            return ""
        normales: list[str] = []
        extra: list[str] = []
        for o in items:
            if not o.codigo:
                continue
            line = f"  {o.codigo} - {o.descripcion or ''}".rstrip()
            if o.ext == 1:
                extra.append(line)
            else:
                normales.append(line)

        parts: list[str] = [
            "CATALOGO DE CODIGOS DE HORA DE SIGRID (auxhor). Propon el "
            "codigo adecuado SOLO de esta lista (no inventes codigos):",
        ]
        if normales:
            parts.append("Codigos de HORA ORDINARIA / mensual / incidencias:")
            parts.append("\n".join(normales))
        if extra:
            parts.append("Codigos de HORA EXTRA:")
            parts.append("\n".join(extra))
        return "\n".join(parts)
