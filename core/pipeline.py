import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.chunker import SemanticChunker
from core.embeddings import EmbeddingService
from core.storage import StorageService
from core.retriever import SemanticRetriever, RetrievedChunk
from core.bm25_retriever import BM25Retriever
from core.hybrid_retriever import HybridRetriever, HybridSearchConfig
from core.reranker import Reranker


class PipelineMode(Enum):
    """
    SEMANTIC_ONLY  → doar embeddings + pgvector (cel mai rapid)
    BM25_ONLY → doar keyword search (fără modele)
    HYBRID → semantic + BM25 fuzionate prin RRF
    HYBRID_RERANK → hybrid + cross-encoder reranking (cel mai precis)
    """
    SEMANTIC_ONLY = "semantic_only"
    BM25_ONLY     = "bm25_only"
    HYBRID        = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class PipelineConfig:
    # retrieval
    mode: PipelineMode = PipelineMode.HYBRID_RERANK
    retrieval_top_k: int = 20        # câte chunks luăm din retrieval înainte de reranking
    final_top_k: int = 5             # câte chunks ajung la LLM după reranking

    # hybrid weights
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60

    # reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_score_threshold: Optional[float] = 0.1

    # chunker
    similarity_threshold: float = 0.35
    min_sentences_per_chunk: int = 2
    max_sentences_per_chunk: int = 5
    context_window: int = 2

    # embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    def __post_init__(self):
        if self.retrieval_top_k < self.final_top_k:
            raise ValueError(
                f"retrieval_top_k ({self.retrieval_top_k}) must be >= "
                f"final_top_k ({self.final_top_k})"
            )


@dataclass
class IngestResult:
    #rezultatul procesării unui document nou
    success: bool
    document_id: Optional[str]
    filename: str
    num_chunks: int
    num_sentences: int
    elapsed_seconds: float
    error: Optional[str] = None

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return (
            f"IngestResult({status} '{self.filename}', "
            f"chunks={self.num_chunks}, "
            f"sentences={self.num_sentences}, "
            f"{self.elapsed_seconds:.2f}s)"
        )


@dataclass
class QueryResult:
    #rezultatul unui query — chunks găsite + metadate despre retrieval
    query: str
    chunks: list[RetrievedChunk]
    mode: PipelineMode
    elapsed_seconds: float
    retrieval_count: int           # câte chunks au ieșit din retrieval
    reranked: bool                 # dacă s-a aplicat reranking
    metadata: dict = field(default_factory=dict)

    @property
    def top_chunk(self) -> Optional[RetrievedChunk]:
        return self.chunks[0] if self.chunks else None

    @property
    def context(self) -> str:
        """Textul tuturor chunk-urilor concatenat — gata pentru LLM."""
        return "\n\n---\n\n".join(chunk.text for chunk in self.chunks)

    def __repr__(self):
        return (
            f"QueryResult(chunks={len(self.chunks)}, "
            f"mode={self.mode.value!r}, "
            f"{self.elapsed_seconds:.2f}s, "
            f"top_score={self.top_chunk.score:.4f if self.top_chunk else 0:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    # Retrieval-Augmented Generation (RAG) pipeline
    #
    # INGEST (document nou):
    # uploaded_file → parse text → creează chunk-uri → generează embeddings
    # → salvează în PostgreSQL + pgvector → actualizează indexul BM25
    #
    # QUERY (întrebare):
    # query → semantic retrieval + BM25 retrieval
    # → combinare rezultate (RRF) → reranking
    # → QueryResult cu chunk-urile finale pentru LLM

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._initialized = False

        # componente — inițializate lazy în setup()
        self._embedding_service: Optional[EmbeddingService] = None
        self._storage_service: Optional[StorageService] = None
        self._chunker: Optional[SemanticChunker] = None
        self._semantic_retriever: Optional[SemanticRetriever] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None


    def setup(self) -> "RAGPipeline":
        print("[Pipeline] Initializing components...")
        start = time.time()

        self._embedding_service = EmbeddingService(
            model_name=self.config.embedding_model
        )
        self._storage_service = StorageService()
        self._chunker = SemanticChunker(
            similarity_threshold=self.config.similarity_threshold,
            min_sentences_per_chunk=self.config.min_sentences_per_chunk,
            max_sentences_per_chunk=self.config.max_sentences_per_chunk,
            context_window=self.config.context_window,
        )

        self._semantic_retriever = SemanticRetriever(
            embedding_service=self._embedding_service,
            storage_service=self._storage_service,
            top_k=self.config.retrieval_top_k,
        )

        self._bm25_retriever = BM25Retriever(
            storage_service=self._storage_service,
            top_k=self.config.retrieval_top_k,
        )

        hybrid_config = HybridSearchConfig(
            semantic_weight=self.config.semantic_weight,
            bm25_weight=self.config.bm25_weight,
            rrf_k=self.config.rrf_k,
            top_k=self.config.retrieval_top_k,
        )

        self._hybrid_retriever = HybridRetriever(
            semantic_retriever=self._semantic_retriever,
            bm25_retriever=self._bm25_retriever,
            config=hybrid_config,
        )

        # reranker doar dacă modul îl cere — e cel mai lent la inițializare
        if self.config.mode == PipelineMode.HYBRID_RERANK:
            self._reranker = Reranker(
                model_name=self.config.reranker_model,
                top_k=self.config.final_top_k,
                score_threshold=self.config.rerank_score_threshold,
            )

        # construiește indexul BM25 din DB
        print("[Pipeline] Building BM25 index from database...")
        bm25_stats = self._bm25_retriever.initialize()
        print(f"[Pipeline] BM25 ready: {bm25_stats}")

        self._initialized = True
        elapsed = time.time() - start
        print(f"[Pipeline] Ready in {elapsed:.2f}s | mode={self.config.mode.value}")

        return self


    def ingest(self, uploaded_file, parser) -> IngestResult:
        # Procesează un document nou și îl salvează în DB
        # Args: uploaded_file = fișier uploadat, parser = instanță de DocumentParser
        # Returns: IngestResult cu detalii despre procesare
        self._check_initialized()
        start = time.time()
        filename = getattr(uploaded_file, "name", "unknown")

        # 1. verifică duplicate
        if self._storage_service.document_exists(filename):
            print(f"[Pipeline] Document '{filename}' already exists, skipping.")
            return IngestResult(
                success=False,
                document_id=None,
                filename=filename,
                num_chunks=0,
                num_sentences=0,
                elapsed_seconds=time.time() - start,
                error="Document already exists in database.",
            )

        # 2. parsează documentul
        print(f"[Pipeline] Parsing '{filename}'...")
        document = parser.parse(uploaded_file)

        if document is None:
            return IngestResult(
                success=False,
                document_id=None,
                filename=filename,
                num_chunks=0,
                num_sentences=0,
                elapsed_seconds=time.time() - start,
                error="Failed to parse document.",
            )

        # 3. împarte în propoziții și chunk-uri
        print(f"[Pipeline] Chunking '{filename}'...")
        sentences = self._chunker.split_sentences(document.content)

        if not sentences:
            return IngestResult(
                success=False,
                document_id=None,
                filename=filename,
                num_chunks=0,
                num_sentences=0,
                elapsed_seconds=time.time() - start,
                error="No sentences extracted from document.",
            )

        # 4. encodează propozițiile
        print(f"[Pipeline] Encoding {len(sentences)} sentences...")
        sentence_embeddings = self._embedding_service.encode_texts(sentences)

        # 5. construiește chunk-urile
        chunks = self._chunker.chunk_sentences(
            sentences=sentences,
            sentence_embeddings=sentence_embeddings,
            document_id=document.id,
        )

        # 6. encodează chunk-urile (embeddings pentru pgvector)
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = self._embedding_service.encode_texts(chunk_texts)

        # 7. salvează în DB
        print(f"[Pipeline] Saving {len(chunks)} chunks to database...")
        document_id = self._storage_service.save_document_with_chunks(
            document=document,
            chunks=chunks,
            embeddings=chunk_embeddings,
        )

        if document_id is None:
            return IngestResult(
                success=False,
                document_id=None,
                filename=filename,
                num_chunks=len(chunks),
                num_sentences=len(sentences),
                elapsed_seconds=time.time() - start,
                error="Failed to save document to database.",
            )

        # 8. actualizează indexul BM25 cu chunk-urile noi
        new_chunks_dicts = [
            {
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "start_sentence": chunk.start_sentence,
                "end_sentence": chunk.end_sentence,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self._bm25_retriever.add_chunks(new_chunks_dicts)

        elapsed = time.time() - start
        result = IngestResult(
            success=True,
            document_id=document_id,
            filename=filename,
            num_chunks=len(chunks),
            num_sentences=len(sentences),
            elapsed_seconds=elapsed,
        )

        print(f"[Pipeline] Ingest complete: {result}")
        return result


    def query(self, user_query: str) -> QueryResult:
        # Rulează un query și returnează cele mai relevante chunk-uri
        # Args: user_query = întrebarea utilizatorului
        # Returns: QueryResult cu chunk-urile finale și context pentru LLM
        self._check_initialized()

        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty.")

        start = time.time()
        user_query = user_query.strip()

        print(f"[Pipeline] Query: {user_query!r} | mode={self.config.mode.value}")

        # retrieval
        retrieved_chunks, retrieval_count = self._retrieve(user_query)

        # reranking (doar în HYBRID_RERANK)
        reranked = False
        if self.config.mode == PipelineMode.HYBRID_RERANK and self._reranker and retrieved_chunks:
            print(f"[Pipeline] Reranking {len(retrieved_chunks)} chunks...")
            retrieved_chunks = self._reranker.rerank_to_chunks(user_query, retrieved_chunks)
            reranked = True
        else:
            # dacă nu avem reranker, tăiem manual la final_top_k
            retrieved_chunks = retrieved_chunks[: self.config.final_top_k]

        elapsed = time.time() - start

        result = QueryResult(
            query=user_query,
            chunks=retrieved_chunks,
            mode=self.config.mode,
            elapsed_seconds=elapsed,
            retrieval_count=retrieval_count,
            reranked=reranked,
            metadata={
                "final_top_k": self.config.final_top_k,
                "retrieval_top_k": self.config.retrieval_top_k,
            },
        )

        print(
            f"[Pipeline] Query done: {len(retrieved_chunks)} chunks returned "
            f"(from {retrieval_count} retrieved) in {elapsed:.2f}s"
        )

        return result

    def _retrieve(self, query: str) -> tuple[list[RetrievedChunk], int]:
        # Execută retrieval-ul în funcție de modul configurat
        # Returns: (chunks, număr_total_retrieved)
        mode = self.config.mode
        if mode == PipelineMode.SEMANTIC_ONLY:
            chunks = self._semantic_retriever.retrieve(query)

        elif mode == PipelineMode.BM25_ONLY:
            chunks = self._bm25_retriever.retrieve(query)

        elif mode in (PipelineMode.HYBRID, PipelineMode.HYBRID_RERANK):
            chunks = self._hybrid_retriever.retrieve_chunks(query)
        else:
            raise ValueError(f"Unknown pipeline mode: {mode}")
        return chunks, len(chunks)


    def switch_mode(self, mode: PipelineMode) -> None:
        if mode == PipelineMode.HYBRID_RERANK and self._reranker is None:
            print("[Pipeline] Initializing reranker...")
            self._reranker = Reranker(
                model_name=self.config.reranker_model,
                top_k=self.config.final_top_k,
                score_threshold=self.config.rerank_score_threshold,
            )
        self.config.mode = mode
        print(f"[Pipeline] Mode switched to: {mode.value}")


    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call setup() first.\n"
                "Example: pipeline = RAGPipeline().setup()"
            )

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def bm25_ready(self) -> bool:
        return self._bm25_retriever is not None and self._bm25_retriever.is_ready