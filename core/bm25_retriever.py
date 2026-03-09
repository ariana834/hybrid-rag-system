import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from core.storage import StorageService
from core.retriever import RetrievedChunk


# Statistici interne pentru indexul BM25
@dataclass
class BM25Stats:
    num_documents: int
    avg_document_length: float
    vocabulary_size: int

    def __repr__(self):
        return (
            f"BM25Stats(docs={self.num_documents}, "
            f"avg_len={self.avg_document_length:.1f}, "
            f"vocab={self.vocabulary_size})"
        )


class BM25Index:
    # Implementare BM25 de la zero
    # Formula: score(D, Q) = Σ IDF(qi) * (TF(qi,D) * (k1+1)) / (TF(qi,D) + k1*(1-b+b*|D|/avgdl))
    # Parametri:
    #   k1 (1.2-2.0): controlează saturarea frecvenței. Mai mare = mai puțină saturare.
    #   b  (0.0-1.0): controlează normalizarea lungimii. 0 = fără normalizare, 1 = normalizare completă.

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # chunk_id -> text original
        self._corpus: dict[str, str] = {}

        # chunk_id -> listă de tokeni
        self._tokenized_corpus: dict[str, list[str]] = {}

        # termen -> {chunk_id -> frecvență}
        self._inverted_index: dict[str, dict[str, int]] = {}

        # termen -> număr de documente care conțin termenul (pentru IDF)
        self._document_frequency: dict[str, int] = {}

        # chunk_id -> număr de tokeni
        self._document_lengths: dict[str, int] = {}

        self._avg_document_length: float = 0.0
        self._num_documents: int = 0

    def build(self, chunks: list[dict]) -> BM25Stats:
        # Construiește indexul din lista de chunk-uri.
        self._reset()

        for chunk in chunks:
            chunk_id = chunk["id"]
            text = chunk["text"]

            tokens = self._tokenize(text)

            self._corpus[chunk_id] = text
            self._tokenized_corpus[chunk_id] = tokens
            self._document_lengths[chunk_id] = len(tokens)

            # construiește indexul inversat
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in self._inverted_index:
                    self._inverted_index[token] = {}
                self._inverted_index[token][chunk_id] = count

        # calculează frecvența documentelor
        for token, doc_dict in self._inverted_index.items():
            self._document_frequency[token] = len(doc_dict)

        self._num_documents = len(chunks)
        total_tokens = sum(self._document_lengths.values())
        self._avg_document_length = total_tokens / max(self._num_documents, 1)

        return BM25Stats(
            num_documents=self._num_documents,
            avg_document_length=self._avg_document_length,
            vocabulary_size=len(self._inverted_index),
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        # Caută query-ul în index și returnează (chunk_id, scor) sortat descrescător.
        if not self._corpus:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            idf = self._compute_idf(token)

            for chunk_id, term_frequency in self._inverted_index[token].items():
                doc_length = self._document_lengths[chunk_id]
                tf_normalized = self._compute_tf(term_frequency, doc_length)

                bm25_score = idf * tf_normalized

                scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _compute_idf(self, token: str) -> float:
        # IDF (Frecvența Inversă a Documentelor) — cât de rar apare termenul în corpus.
        df = self._document_frequency.get(token, 0)
        return math.log(
            (self._num_documents - df + 0.5) / (df + 0.5) + 1
        )

    def _compute_tf(self, term_frequency: int, doc_length: int) -> float:
        # TF normalizat cu saturare și normalizare de lungime.
        normalization = 1 - self.b + self.b * (doc_length / self._avg_document_length)
        return (term_frequency * (self.k1 + 1)) / (term_frequency + self.k1 * normalization)

    def _tokenize(self, text: str) -> list[str]:
        # Tokenizare simplă: minuscule + eliminare caractere speciale.
        text = text.lower()
        text = re.sub(r"[^a-z0-9\săîâțș]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]  # elimină tokeni de 1 caracter

    def _reset(self):
        self._corpus.clear()
        self._tokenized_corpus.clear()
        self._inverted_index.clear()
        self._document_frequency.clear()
        self._document_lengths.clear()
        self._avg_document_length = 0.0
        self._num_documents = 0

    @property
    def is_built(self) -> bool:
        return self._num_documents > 0

    @property
    def stats(self) -> Optional[BM25Stats]:
        if not self.is_built:
            return None
        return BM25Stats(
            num_documents=self._num_documents,
            avg_document_length=self._avg_document_length,
            vocabulary_size=len(self._inverted_index),
        )


class BM25Retriever:
    # Retriever bazat pe BM25 — căutare pe cuvinte cheie (lexicală).
    # - Semantică: înțelege sensul, dar poate rata cuvinte cheie exacte
    # - BM25: găsește cuvinte cheie exacte, dar nu înțelege sinonime
    def __init__(
        self,
        storage_service: StorageService,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.storage_service = storage_service
        self.top_k = top_k
        self._index = BM25Index(k1=k1, b=b)
        self._chunk_lookup: dict[str, dict] = {}  # chunk_id -> dict complet al chunk-ului
        self._is_initialized = False

    def initialize(self) -> BM25Stats:
        # Încarcă toate chunk-urile din DB și construiește indexul BM25.
        # Trebuie apelat înainte de primul retrieve().
        print("[BM25] Loading chunks from database...")
        all_chunks = self.storage_service.get_all_chunks()

        if not all_chunks:
            print("[BM25] No chunks found in database.")
            self._is_initialized = True
            return BM25Stats(0, 0.0, 0)

        # construiește lookup rapid chunk_id -> chunk
        self._chunk_lookup = {chunk["id"]: chunk for chunk in all_chunks}

        stats = self._index.build(all_chunks)
        self._is_initialized = True

        print(f"[BM25] Index built: {stats}")
        return stats

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not self._is_initialized:
            raise RuntimeError(
                "BM25Retriever not initialized. Call initialize() first."
            )

        if not self._index.is_built:
            return []

        raw_results = self._index.search(query, top_k=self.top_k)

        return self._build_retrieved_chunks(raw_results)

    def _build_retrieved_chunks(
        self,
        raw_results: list[tuple[str, float]],
    ) -> list[RetrievedChunk]:
        if not raw_results:
            return []

        # normalizează scorurile între 0 și 1
        max_score = max(score for _, score in raw_results)

        chunks = []
        for chunk_id, raw_score in raw_results:
            chunk_data = self._chunk_lookup.get(chunk_id)
            if chunk_data is None:
                continue

            normalized_score = raw_score / max_score if max_score > 0 else 0.0

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=chunk_data["document_id"],
                    chunk_index=chunk_data["chunk_index"],
                    text=chunk_data["text"],
                    score=normalized_score,
                    retriever_type="bm25",
                    metadata=chunk_data.get("metadata", {}),
                )
            )

        return chunks

    def add_chunks(self, new_chunks: list[dict]) -> None:
        # Adaugă chunk-uri noi în index fără să-l reconstruiești complet.
        # Util după ce un document nou e procesat.
        for chunk in new_chunks:
            self._chunk_lookup[chunk["id"]] = chunk

        all_chunks = list(self._chunk_lookup.values())
        self._index.build(all_chunks)
        print(f"[BM25] Index rebuilt with {len(all_chunks)} total chunks.")

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self._index.is_built