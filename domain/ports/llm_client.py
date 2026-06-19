# domain/ports/llm_client.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment


class LlmVisionClient(ABC):
    @abstractmethod
    def extract_document(
        self,
        *,
        model: str,
        instructions: str,
        user_text: str,
        attachment: LlmAttachment,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        raise NotImplementedError
