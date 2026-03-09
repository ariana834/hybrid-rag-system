import pytest
from core.bm25_retriever import BM25Index, BM25Retriever


@pytest.fixture
def index(sample_chunks_dicts):
    idx = BM25Index()
    idx.build(sample_chunks_dicts)
    return idx


# ── BM25Index ────────────────────────────────────────────────────

def test_build_sets_stats(index, sample_chunks_dicts):
    assert index.stats.docs == len(sample_chunks_dicts)
    assert index.stats.vocab > 0
    assert index.stats.avg_len > 0

def test_search_returns_results(index):
    results = index.search("Python programare", top_k=3)
    assert len(results) > 0

def test_search_top_result_relevant(index):
    results = index.search("Python programare", top_k=5)
    top = results[0]
    assert "Python" in top["text"]

def test_search_scores_normalized(index):
    results = index.search("machine learning date", top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0

def test_search_top_k_respected(index):
    results = index.search("programare", top_k=2)
    assert len(results) <= 2

def test_search_empty_query(index):
    results = index.search("", top_k=5)
    assert isinstance(results, list)

def test_build_empty(sample_chunks_dicts):
    idx = BM25Index()
    idx.build([])
    assert idx.stats.docs == 0

def test_idf_computed(index):
    assert len(index._idf) > 0

def test_search_unknown_term(index):
    results = index.search("xyzabc123", top_k=3)
    assert isinstance(results, list)