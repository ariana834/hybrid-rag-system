import re
import pytest
from unittest.mock import MagicMock, patch
from core.llm_service import LLMService, LLMResponse, CitedSource
from core.retriever import RetrievedChunk

def make_retrieved_chunk(i):
    return RetrievedChunk(
        chunk_id=f"c-{i}",
        document_id="d-0",
        chunk_index=i,
        text=f"Chunk {i}: informație relevantă despre topicul {i}.",
        score=0.9 - i * 0.1,
        retriever_type="hybrid",
        metadata={},
    )

def test_cited_source_fields():
    s = CitedSource(
        citation_number=1,
        chunk_id="c-0",
        document_id="d-0",
        chunk_index=0,
        text_excerpt="text...",
        relevance_score=0.95,
    )
    assert s.citation_number == 1
    assert s.relevance_score == 0.95

def test_llm_response_cost_zero():
    r = LLMResponse(
        question="test?", answer="răspuns", answer_with_citations="răspuns",
        sources=[], model="gpt-4o-mini", elapsed_seconds=1.0,
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
    )
    assert r.cost_usd > 0.0

def test_llm_response_repr():
    r = LLMResponse(
        question="q", answer="a", answer_with_citations="a",
        sources=[], model="gpt-4o-mini", elapsed_seconds=0.5,
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
    )
    assert "gpt-4o-mini" in repr(r)

def test_citation_regex_finds_numbers():
    text = "Salariul este 4050 lei [1][2] conform legii [3]."
    found = list(map(int, re.findall(r'\[(\d+)\]', text)))
    assert found == [1, 2, 3]

def test_citation_regex_empty():
    text = "Nicio citare în acest text."
    found = re.findall(r'\[(\d+)\]', text)
    assert found == []

def test_citation_regex_duplicate():
    text = "Conform [1] și din nou [1]."
    found = list(map(int, re.findall(r'\[(\d+)\]', text)))
    assert found.count(1) == 2

def test_numbered_context_format():
    chunks = [make_retrieved_chunk(i) for i in range(3)]
    parts = [f"[{i+1}] {c.text}" for i, c in enumerate(chunks)]
    context = "\n\n".join(parts)
    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context

def test_numbered_context_single_chunk():
    chunk = make_retrieved_chunk(0)
    context = f"[1] {chunk.text}"
    assert context.startswith("[1]")

def test_llm_service_raises_without_key():
    with pytest.raises(ValueError, match="API key"):
        LLMService(api_key=None)

def test_llm_service_init_with_key():
    svc = LLMService(api_key="sk-test-fake-key-for-testing")
    assert svc.model == "gpt-4o-mini"
    assert svc.temperature == 0.1