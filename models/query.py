from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    top_k: int = 5
    filters: dict = field(default_factory=dict)  # ex: {"document_id": "abc123"}