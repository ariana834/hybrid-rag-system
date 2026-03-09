from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(  texts, convert_to_numpy=True, normalize_embeddings=True, )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode( [query],convert_to_numpy=True, normalize_embeddings=True,)[0]