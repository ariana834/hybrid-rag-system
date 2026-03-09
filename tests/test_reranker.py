import math
import pytest
from core.reranker import Reranker
from core.retriever import RetrievedChunk

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def make_chunk(i, score=0.9):
    return RetrievedChunk(
        chunk_id=f"chunk-{i}",
        document_id="doc-0",
        chunk_index=i,
        text=f"Text despre topicul {i}.",
        score=score,
        retriever_type="semantic",
        metadata={},
    )

def test_sigmoid_zero():
    assert sigmoid(0) == pytest.approx(0.5, abs=1e-6)

def test_sigmoid_positive():
    assert sigmoid(5) > 0.99

def test_sigmoid_negative():
    assert sigmoid(-5) < 0.01

def test_sigmoid_output_range():
    for x in [-10, -1, 0, 1, 10]:
        s = sigmoid(x)
        assert 0.0 < s < 1.0

def test_sigmoid_monotone():
    assert sigmoid(1) > sigmoid(0) > sigmoid(-1)

def test_rerank_result_count(sample_retrieved_chunks):
    """Reranker trebuie să returneze același număr de chunks."""
    reranker = Reranker()
    results = reranker.rerank("Python programare", sample_retrieved_chunks)
    assert len(results) == len(sample_retrieved_chunks)

def test_rerank_result_fields(sample_retrieved_chunks):
    reranker = Reranker()
    results = reranker.rerank("Python", sample_retrieved_chunks)
    for r in results:
        assert hasattr(r, "chunk")
        assert hasattr(r, "rerank_score")
        assert hasattr(r, "rerank_rank")
        assert hasattr(r, "original_rank")

def test_rerank_scores_normalized(sample_retrieved_chunks):
    reranker = Reranker()
    results = reranker.rerank("Python", sample_retrieved_chunks)
    for r in results:
        assert 0.0 <= r.rerank_score <= 1.0

def test_rerank_ranks_sequential(sample_retrieved_chunks):
    reranker = Reranker()
    results = reranker.rerank("Python", sample_retrieved_chunks)
    ranks = [r.rerank_rank for r in results]
    assert sorted(ranks) == list(range(1, len(results) + 1))

def test_rerank_to_chunks_returns_retrieved_chunks(sample_retrieved_chunks):
    reranker = Reranker()
    chunks = reranker.rerank_to_chunks("Python", sample_retrieved_chunks)
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    assert len(chunks) == len(sample_retrieved_chunks)

def test_rerank_empty_list():
    reranker = Reranker()
    results = reranker.rerank("test", [])
    assert results == []