from dataclasses import dataclass, field


@dataclass
class Source:
    chunk_id: str
    document_id: str
    chunk_index: int
    text_excerpt: str
    score: float


@dataclass
class Response:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    total_tokens: int = 0