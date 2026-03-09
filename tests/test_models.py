import pytest
from models.chunk import Chunk
from models.document import Document
from models.query import Query
from models.response import Response, Source


# chunk

def test_chunk_creation(sample_chunk):
    assert sample_chunk.chunk_id == "doc_0_chunk_0"
    assert sample_chunk.chunk_index == 0
    assert sample_chunk.document_id == "doc_0"
    assert len(sample_chunk.text) > 0

def test_chunk_metadata(sample_chunk):
    assert "num_sentences" in sample_chunk.metadata
    assert "split_reason" in sample_chunk.metadata

def test_chunk_sentence_range(sample_chunk):
    assert sample_chunk.start_sentence <= sample_chunk.end_sentence

# document
def test_document_creation():
    doc = Document(
        id="uuid-123",
        filename="test.pdf",
        file_type="pdf",
        content="Conținut test.",
        num_characters=14,
    )
    assert doc.filename == "test.pdf"
    assert doc.file_type == "pdf"
    assert doc.num_characters == 14

def test_document_id_is_string():
    doc = Document(id="abc", filename="f.txt", file_type="txt", content="x", num_characters=1)
    assert isinstance(doc.id, str)


#query
def test_query_defaults():
    q = Query(text="Ce este Python?")
    assert q.text == "Ce este Python?"
    assert q.top_k == 5
    assert q.filters == {}

def test_query_custom():
    q = Query(text="test", top_k=10, filters={"type": "pdf"})
    assert q.top_k == 10
    assert q.filters["type"] == "pdf"

def test_response_creation():
    r = Response(question="Test?", answer="Răspuns.", sources=[])
    assert r.question == "Test?"
    assert r.answer == "Răspuns."
    assert r.sources == []

def test_source_creation():
    s = Source(
        chunk_id="c-1",
        document_id="d-1",
        chunk_index=0,
        text_excerpt="Un text.",
        score=0.95,
    )
    assert s.score == 0.95
    assert s.chunk_index == 0