# domain/models/llm_attachment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttachmentKind = Literal["image", "pdf"]


@dataclass(frozen=True)
class LlmAttachment:
    kind: AttachmentKind
    filename: str
    mime_type: str
    data: bytes
