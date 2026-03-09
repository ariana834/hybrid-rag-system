from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_sentence: int
    end_sentence: int
    metadata: dict[str, Any] = field(default_factory=dict)