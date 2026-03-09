import pytest
import numpy as np
from models.chunk import Chunk
from models.document import Document
from core.retriever import RetrievedChunk
from core.pipeline import PipelineMode


@pytest.fixture
def sample_chunk():
    return Chunk(
        chunk_id="doc_0_chunk_0",
        document_id="doc_0",
        chunk_index=0,
        text="Python este un limbaj de programare popular.",
        start_sentence=0,
        end_sentence=1,
        metadata={"num_sentences": 2, "split_reason": "semantic_shift"},
    )


@pytest.fixture
def sample_chunks():
    texts = [
        "Python este un limbaj de programare popular. Este folosit în web development.",
        "Fotbalul este un sport. Un meci durează 90 de minute.",
        "Machine learning folosește date pentru a antrena modele. NumPy este util.",
        "Baza de date stochează informații. SQL este limbajul de interogare.",
        "Docker containerizează aplicații. Este folosit în DevOps.",
    ]
    return [
        Chunk(
            chunk_id=f"doc_0_chunk_{i}",
            document_id="doc_0",
            chunk_index=i,
            text=text,
            start_sentence=i * 2,
            end_sentence=i * 2 + 1,
            metadata={"num_sentences": 2},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def sample_chunks_dicts():
    texts = [
        "Python este un limbaj de programare popular.",
        "Fotbalul este un sport cunoscut în toată lumea.",
        "Machine learning folosește date pentru antrenare.",
        "Baza de date stochează informații structurate.",
        "Docker containerizează aplicații cu ușurință.",
    ]
    return [
        {
            "id": f"uuid-{i}",
            "document_id": "doc-uuid-0",
            "chunk_index": i,
            "text": text,
            "start_sentence": i,
            "end_sentence": i + 1,
            "metadata": {},
        }
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def sample_retrieved_chunks():
    texts = [
        "Python este un limbaj de programare popular.",
        "Machine learning folosește date pentru antrenare.",
        "Docker containerizează aplicații.",
    ]
    return [
        RetrievedChunk(
            chunk_id=f"uuid-{i}",
            document_id="doc-0",
            chunk_index=i,
            text=text,
            score=0.9 - i * 0.1,
            retriever_type="semantic",
            metadata={},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def random_embeddings():
    np.random.seed(42)
    return np.random.rand(5, 384).astype(np.float32)