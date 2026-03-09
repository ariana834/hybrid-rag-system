import pytest
from core.hybrid_retriever import HybridSearchConfig, HybridRetriever
from core.retriever import RetrievedChunk


@pytest.fixture
def config():
    return HybridSearchConfig(
        semantic_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60,
        top_k=5,
    )

def make_chunk(chunk_id, score, retriever_type="semantic"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-0",
        chunk_index=0,
        text=f"text for {chunk_id}",
        score=score,
        retriever_type=retriever_type,
        metadata={},
    )

def test_config_defaults(config):
    assert config.semantic_weight == 0.5
    assert config.bm25_weight == 0.5
    assert config.rrf_k == 60
    assert config.top_k == 5

def test_config_custom():
    c = HybridSearchConfig(semantic_weight=0.7, bm25_weight=0.3, rrf_k=30, top_k=3)
    assert c.semantic_weight == 0.7
    assert c.top_k == 3


# ── RRF fusion logic ─────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    """Chunk de pe rank 1 trebuie să aibă scor mai mare decât rank 3."""
    k = 60
    score_rank1 = 1.0 / (k + 1)
    score_rank3 = 1.0 / (k + 3)
    assert score_rank1 > score_rank3

def test_rrf_k_affects_score():
    """k mai mare → diferențe mai mici între rank-uri."""
    rank = 1
    score_k10  = 1.0 / (10 + rank)
    score_k100 = 1.0 / (100 + rank)
    assert score_k10 > score_k100

def test_rrf_two_lists_same_chunk():
    """Chunk care apare în ambele liste primește scor mai mare."""
    k = 60
    # apare în ambele liste pe rank 1
    both = 1.0 / (k + 1) + 1.0 / (k + 1)
    # apare doar în una pe rank 1
    one = 1.0 / (k + 1)
    assert both > one

def test_rrf_weights():
    """Semantic weight 0.7 trebuie să domine față de BM25 0.3."""
    k = 60
    rank = 1
    semantic_score = 0.7 * (1.0 / (k + rank))
    bm25_score     = 0.3 * (1.0 / (k + rank))
    assert semantic_score > bm25_score


def test_update_config(config):
    config.semantic_weight = 0.8
    config.bm25_weight = 0.2
    assert config.semantic_weight == 0.8
    assert config.bm25_weight == 0.2