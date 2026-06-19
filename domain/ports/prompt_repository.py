# domain/ports/prompt_repository.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    system: str
    task: str
    schema_hint: str
    schema: str


class PromptRepository(ABC):
    @abstractmethod
    def get(self, prompt_key: str) -> PromptSpec:
        raise NotImplementedError
