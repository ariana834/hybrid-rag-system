from dataclasses import dataclass
@dataclass
class Document:
    id: str
    filename: str
    file_type: str
    content: str
    num_characters: int