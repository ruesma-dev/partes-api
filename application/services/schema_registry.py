# application/services/schema_registry.py
from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from domain.models.parte_models import ParteTrabajo


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: Dict[str, Type[BaseModel]] = {
            "parte_trabajo": ParteTrabajo,
        }

    def get(self, schema_name: str) -> Type[BaseModel]:
        if schema_name in self._schemas:
            return self._schemas[schema_name]
        available = ", ".join(sorted(self._schemas.keys()))
        raise KeyError(
            f"Schema '{schema_name}' no registrado. Disponibles: {available}"
        )
