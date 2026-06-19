# infrastructure/prompts/yaml_prompt_repository.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml

from domain.ports.prompt_repository import PromptRepository, PromptSpec

logger = logging.getLogger(__name__)


class YamlPromptRepository(PromptRepository):
    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        self._specs: Dict[str, PromptSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"No existe prompts YAML: {self._path}")

        raw = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Formato YAML inválido en {self._path}. "
                "Se esperaba un mapping en raíz."
            )

        self._specs = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            self._specs[key] = PromptSpec(
                system=str(value.get("system") or "").strip(),
                task=str(value.get("task") or "").strip(),
                schema_hint=str(value.get("schema_hint") or "").strip(),
                schema=str(value.get("schema") or "").strip(),
            )

        logger.info(
            "Prompts cargados: %s | yaml=%s",
            ", ".join(sorted(self._specs.keys())) or "(none)",
            str(self._path),
        )

    def get(self, prompt_key: str) -> PromptSpec:
        if prompt_key not in self._specs:
            available = ", ".join(sorted(self._specs.keys()))
            raise KeyError(
                f"prompt_key '{prompt_key}' no existe. "
                f"Disponibles: {available}"
            )
        return self._specs[prompt_key]
