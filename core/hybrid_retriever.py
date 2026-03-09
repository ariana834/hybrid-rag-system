from dataclasses import dataclass
from core.retriever import SemanticRetriever, RetrievedChunk
from core.bm25_retriever import BM25Retriever


@dataclass
class HybridSearchConfig:
    # semantic_weight + bm25_weight = 1.0 (întotdeauna)
    # rrf_k = 60 (valoare standard din paper-ul original RRF)
    # top_k = câte rezultate finale returnăm după fuziune
    # deduplicate = dacă True, eliminăm duplicatele (safety net, nu ar trebui să apară după union)
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60
    top_k: int = 5
    deduplicate: bool = True

    def __post_init__(self):
        total = self.semantic_weight + self.bm25_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"semantic_weight + bm25_weight must equal 1.0, got {total:.2f}"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")


@dataclass
class HybridSearchResult:
    #rezultat final al căutării hibride
    chunk: RetrievedChunk
    semantic_rank: int | None      # poziția în lista semantică
    bm25_rank: int | None          # poziția în lista BM25
    semantic_rrf_score: float      # contribuția semantică la scorul final
    bm25_rrf_score: float          # contribuția BM25 la scorul final
    final_score: float             # scorul combinat final

    def __repr__(self):
        return (
            f"HybridResult(score={self.final_score:.4f}, "
            f"sem_rank={self.semantic_rank}, "
            f"bm25_rank={self.bm25_rank}, "
            f"text={self.chunk.text[:60]!r}...)"
        )


class HybridRetriever:
    # Combina semantic search + BM25 folosind Reciprocal Rank Fusion (RRF)
    # RRF calculează scorul final pe baza poziției documentelor în fiecare ranking
    # Evită normalizarea scorurilor (BM25 și cosine au scale diferite)
    # Returnează top_k rezultate combinate și elimină duplicatele
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        config: HybridSearchConfig | None = None,
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridSearchConfig()

    def retrieve(self, query: str) -> list[HybridSearchResult]:
        # Execută ambele căutări în paralel și combină rezultatele prin RRF.
        semantic_results = self._safe_retrieve_semantic(query)
        bm25_results = self._safe_retrieve_bm25(query)

        return self._fuse(semantic_results, bm25_results)

    def retrieve_chunks(self, query: str) -> list[RetrievedChunk]:
        # Versiune simplificată care returnează direct RetrievedChunk-uri.
        # Folosită de pipeline când nu avem nevoie de detaliile RRF.
        results = self.retrieve(query)

        chunks = []
        for result in results:
            result.chunk.score = result.final_score
            result.chunk.retriever_type = "hybrid"
            chunks.append(result.chunk)

        return chunks

    def _fuse(
        self,
        semantic_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
    ) -> list[HybridSearchResult]:
        # Aplică RRF pentru a combina cele două liste de rezultate.
        # mapăm chunk_id -> rang (1-based) pentru fiecare sursă
        semantic_ranks: dict[str, int] = {
            chunk.chunk_id: rank + 1
            for rank, chunk in enumerate(semantic_results)
        }
        bm25_ranks: dict[str, int] = {
            chunk.chunk_id: rank + 1
            for rank, chunk in enumerate(bm25_results)
        }

        # union de chunk_id-uri din ambele surse
        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        # lookup rapid chunk_id -> RetrievedChunk
        chunk_lookup: dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk
            for chunk in semantic_results + bm25_results
        }

        results = []
        k = self.config.rrf_k

        for chunk_id in all_chunk_ids:
            chunk = chunk_lookup[chunk_id]

            sem_rank = semantic_ranks.get(chunk_id)
            bm25_rank = bm25_ranks.get(chunk_id)

            # RRF score per sursă — dacă chunk-ul nu apare într-o sursă, contribuția e 0
            sem_rrf = (self.config.semantic_weight / (k + sem_rank) if sem_rank is not None else 0.0)
            bm25_rrf = ( self.config.bm25_weight / (k + bm25_rank)if bm25_rank is not None else 0.0)
            final_score = sem_rrf + bm25_rrf
            results.append(
                HybridSearchResult(
                    chunk=chunk,
                    semantic_rank=sem_rank,
                    bm25_rank=bm25_rank,
                    semantic_rrf_score=sem_rrf,
                    bm25_rrf_score=bm25_rrf,
                    final_score=final_score,
                )
            )

        # sortăm descrescător după scorul final
        results.sort(key=lambda r: r.final_score, reverse=True)

        if self.config.deduplicate:
            results = self._deduplicate(results)

        return results[: self.config.top_k]

    def _deduplicate(self, results: list[HybridSearchResult]) -> list[HybridSearchResult]:
        # Elimină duplicatele păstrând cel mai bun scor pentru fiecare chunk_id.
        seen: set[str] = set()
        unique = []
        for result in results:
            if result.chunk.chunk_id not in seen:
                seen.add(result.chunk.chunk_id)
                unique.append(result)
        return unique

    def _safe_retrieve_semantic(self, query: str) -> list[RetrievedChunk]:
        try:
            return self.semantic_retriever.retrieve(query)
        except Exception as error:
            print(f"[Hybrid] Semantic retrieval failed: {error}")
            return []

    def _safe_retrieve_bm25(self, query: str) -> list[RetrievedChunk]:
        try:
            return self.bm25_retriever.retrieve(query)
        except Exception as error:
            print(f"[Hybrid] BM25 retrieval failed: {error}")
            return []

    def update_config(self, **kwargs) -> None:

        current = {
            "semantic_weight": self.config.semantic_weight,
            "bm25_weight": self.config.bm25_weight,
            "rrf_k": self.config.rrf_k,
            "top_k": self.config.top_k,
            "deduplicate": self.config.deduplicate,
        }
        current.update(kwargs)
        self.config = HybridSearchConfig(**current)
        print(f"[Hybrid] Config updated: {self.config}")