import re
from typing import List
import numpy as np
from models.chunk import Chunk


class SemanticChunker:
    def __init__(
        self,
        similarity_threshold: float = 0.35,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 5,
        context_window: int = 2,
    ):
        if min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be > 0")

        if max_sentences_per_chunk < min_sentences_per_chunk:
            raise ValueError("max_sentences_per_chunk must be >= min_sentences_per_chunk")

        if context_window <= 0:
            raise ValueError("context_window must be > 0")

        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.context_window = context_window

    def split_sentences(self, text: str) -> List[str]:
        text = text.strip()

        if not text:
            return []

        text = re.sub(r"\s+", " ", text)
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)

        sentences = []
        for sentence in raw_sentences:
            cleaned = sentence.strip()
            if cleaned:
                sentences.append(cleaned)

        return sentences

    def chunk_sentences(
        self,
        sentences: List[str],
        sentence_embeddings: np.ndarray,
        document_id: str = "doc_0"
    ) -> List[Chunk]:
        if not sentences:
            return []

        if len(sentences) != len(sentence_embeddings):
            raise ValueError("Number of sentences must match number of embeddings")

        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id=f"{document_id}_chunk_0",
                    document_id=document_id,
                    chunk_index=0,
                    text=sentences[0],
                    start_sentence=0,
                    end_sentence=0,
                    metadata={
                        "num_sentences": 1,
                        "split_reason": "single_sentence",
                    },
                )
            ]

        chunks: List[Chunk] = []
        current_sentences = [sentences[0]]
        current_start = 0
        chunk_index = 0

        for i in range(1, len(sentences)):
            current_embedding = sentence_embeddings[i]

            # luăm doar ultimele N propoziții din chunkul curent
            window_start = max(current_start, i - self.context_window)
            recent_embeddings = sentence_embeddings[window_start:i]
            recent_mean_embedding = np.mean(recent_embeddings, axis=0)

            # similarity față de contextul recent
            similarity = self._cosine_similarity(recent_mean_embedding, current_embedding)

            should_split = (
                similarity < self.similarity_threshold
                and len(current_sentences) >= self.min_sentences_per_chunk
            )

            too_large = len(current_sentences) >= self.max_sentences_per_chunk

            print("Sentence:", sentences[i])
            print("Similarity:", similarity)
            print("Should split:", should_split)
            print("Too large:", too_large)
            print("-" * 50)

            if should_split or too_large:
                split_reason = "semantic_shift" if should_split else "max_sentences"

                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=" ".join(current_sentences).strip(),
                        start_sentence=current_start,
                        end_sentence=i - 1,
                        metadata={
                            "num_sentences": len(current_sentences),
                            "split_reason": split_reason,
                            "similarity_before_split": similarity,
                        },
                    )
                )

                chunk_index += 1
                current_sentences = [sentences[i]]
                current_start = i
            else:
                current_sentences.append(sentences[i])

        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                text=" ".join(current_sentences).strip(),
                start_sentence=current_start,
                end_sentence=len(sentences) - 1,
                metadata={
                    "num_sentences": len(current_sentences),
                    "split_reason": "final_chunk",
                },
            )
        )

        return chunks

    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        if denominator == 0:
            return 0.0

        return float(np.dot(vector1, vector2) / denominator)