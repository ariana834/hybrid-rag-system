import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from core.embeddings import EmbeddingService
from core.storage import StorageService


@dataclass
class RetrievedChunk:
    #Reprezintă un chunk găsit în urma căutării, împreună cu scorul său și tipul de retriever care l-a găsit.
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    retriever_type: str                  # "semantic", "bm25", sau "hybrid"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"RetrievedChunk(score={self.score:.4f}, "
            f"type={self.retriever_type!r}, "
            f"text={self.text[:60]!r}...)"
        )


class SemanticRetriever:
    # Căutare semantică folosind embeddings + pgvector (cosine similarity)
    # Query-ul este transformat în embedding
    # pgvector găsește cei mai apropiați vectori din DB
    # Returnează chunk-urile relevante cu scorul lor
    def __init__(
        self,
        embedding_service: EmbeddingService,
        storage_service: StorageService,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        self.embedding_service = embedding_service
        self.storage_service = storage_service
        self.top_k = top_k
        self.score_threshold = score_threshold  # dacă e setat, filtrează rezultatele slabe

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.encode_query(query)

        raw_results = self.storage_service.semantic_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        chunks = self._build_retrieved_chunks(raw_results, query_embedding)

        if self.score_threshold is not None:
            chunks = [c for c in chunks if c.score >= self.score_threshold]

        return chunks

    def _build_retrieved_chunks (self,raw_results: list[dict],query_embedding: np.ndarray, ) -> list[RetrievedChunk]:
        chunks = []
        for i, result in enumerate(raw_results):
            position_score = 1.0 - (i / max(len(raw_results), 1))
            chunks.append(
                RetrievedChunk(
                    chunk_id=result["id"],
                    document_id=result["document_id"],
                    chunk_index=result["chunk_index"],
                    text=result["text"],
                    score=position_score,
                    retriever_type="semantic",
                    metadata=result.get("metadata", {}),
                )
            )
        return chunks