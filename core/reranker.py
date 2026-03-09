from dataclasses import dataclass
from typing import Optional
from sentence_transformers import CrossEncoder
from core.retriever import RetrievedChunk


@dataclass
class RerankResult:
    #rezultat după reranking, cu scorul original și cel nou
    chunk: RetrievedChunk
    original_rank: int        # rangul înainte de reranking
    rerank_rank: int          # rangul după reranking
    original_score: float     # scorul din hybrid retriever
    rerank_score: float       # scorul din cross-encoder (între 0 și 1)
    rank_change: int          # câte poziții s-a mișcat (pozitiv = a urcat)

    def __repr__(self):
        direction = "↑" if self.rank_change > 0 else ("↓" if self.rank_change < 0 else "→")
        return (
            f"RerankResult({direction}{abs(self.rank_change)}, "
            f"score={self.rerank_score:.4f}, "
            f"text={self.chunk.text[:60]!r}...)"
        )


class Reranker:
    # Cross-Encoder reranker pentru reordonarea rezultatelor de căutare
    # Retrieval-ul (semantic + BM25) folosește bi-encoders: query și chunk sunt
    # encodate separat și comparate prin cosine similarity → rapid, dar mai puțin precis.
    # Cross-Encoder evaluează direct perechea [query + chunk] și produce un scor
    # de relevanță → mai precis, dar mai lent.
    # În RAG: retrieval rapid (top 20–50) → reranking cu cross-encoder → top K final.
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        normalize_scores: bool = True,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.normalize_scores = normalize_scores

        print(f"[Reranker] Loading model '{model_name}'...")
        self._model = CrossEncoder(model_name)
        print(f"[Reranker] Model loaded.")

    def rerank(self,query: str, chunks: list[RetrievedChunk]) -> list[RerankResult]:
        # Reordonează chunk-urile în funcție de relevanța față de query
        # Args: query = întrebarea utilizatorului, chunks = rezultate din retriever
        # Returns: listă de RerankResult sortată descrescător după rerank_score
        if not chunks:
            return []

        if len(chunks) == 1:
            return [
                RerankResult(
                    chunk=chunks[0],
                    original_rank=1,
                    rerank_rank=1,
                    original_score=chunks[0].score,
                    rerank_score=chunks[0].score,
                    rank_change=0,
                )
            ]
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = self._model.predict(pairs)

        if self.normalize_scores:
            scores = self._sigmoid(raw_scores)
        else:
            scores = list(raw_scores)

        # asociem fiecare chunk cu scorul său
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        original_ranks = {chunk.chunk_id: rank + 1 for rank, chunk in enumerate(chunks)}

        results = []
        for new_rank, (chunk, rerank_score) in enumerate(scored, start=1):
            original_rank = original_ranks[chunk.chunk_id]
            rank_change = original_rank - new_rank  # pozitiv = a urcat în clasament

            results.append(
                RerankResult(
                    chunk=chunk,
                    original_rank=original_rank,
                    rerank_rank=new_rank,
                    original_score=chunk.score,
                    rerank_score=float(rerank_score),
                    rank_change=rank_change,
                )
            )

        # filtrăm după threshold dacă e setat
        if self.score_threshold is not None:
            results = [r for r in results if r.rerank_score >= self.score_threshold]
        return results[: self.top_k]

    def rerank_to_chunks( self,  query: str,chunks: list[RetrievedChunk],) -> list[RetrievedChunk]:
        results = self.rerank(query, chunks)
        reranked_chunks = []
        for result in results:
            result.chunk.score = result.rerank_score
            result.chunk.retriever_type = "reranked"
            reranked_chunks.append(result.chunk)
        return reranked_chunks

    def _sigmoid(self, scores) -> list[float]:
        # Normalizează scorurile cross-encoder în intervalul (0, 1)
        # Aplică sigmoid pe logits pentru a obține probabilități interpretabile
        import math
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]

    def get_model_info(self) -> dict:
        return {
            "model": self._model.model.config.name_or_path,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "normalize_scores": self.normalize_scores,
        }