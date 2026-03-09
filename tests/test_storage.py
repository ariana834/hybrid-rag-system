import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.storage import StorageService
from models.document import Document
from models.chunk import Chunk


@pytest.fixture
def storage():
    return StorageService()

@pytest.fixture
def sample_document():
    return Document(
        id="550e8400-e29b-41d4-a716-446655440000",
        filename="test.pdf",
        file_type="pdf",
        content="Conținut test.",
        num_characters=14,
    )
@pytest.fixture
def sample_chunks_for_storage(sample_document):
    return [
        Chunk(
            chunk_id=f"doc_chunk_{i}",
            document_id=sample_document.id,
            chunk_index=i,
            text=f"Chunk {i} text.",
            start_sentence=i,
            end_sentence=i + 1,
            metadata={"num_sentences": 2},
        )
        for i in range(3)
    ]

def test_save_mismatched_chunks_embeddings(storage, sample_document, sample_chunks_for_storage):
    wrong_embeddings = np.random.rand(2, 384).astype(np.float32)  # 2 != 3
    with pytest.raises(ValueError, match="must match"):
        storage.save_document_with_chunks(sample_document, sample_chunks_for_storage, wrong_embeddings)

def test_save_returns_none_on_db_error(storage, sample_document, sample_chunks_for_storage):
    embeddings = np.random.rand(3, 384).astype(np.float32)
    with patch("core.storage.SessionLocal") as mock_session:
        mock_session.return_value.__enter__.return_value.add.side_effect = Exception("DB error")
        result = storage.save_document_with_chunks(sample_document, sample_chunks_for_storage, embeddings)
        assert result is None

def test_document_exists_returns_bool(storage):
    with patch("core.storage.SessionLocal") as mock_session:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_ctx
        mock_ctx.execute.return_value.first.return_value = None
        result = storage.document_exists("nonexistent.pdf")
        assert result is False

def test_document_exists_true(storage):
    with patch("core.storage.SessionLocal") as mock_session:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_ctx
        mock_ctx.execute.return_value.first.return_value = MagicMock()
        result = storage.document_exists("exists.pdf")
        assert result is True

def test_get_all_chunks_returns_list(storage):
    with patch("core.storage.SessionLocal") as mock_session:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_ctx
        mock_ctx.execute.return_value.scalars.return_value.all.return_value = []
        result = storage.get_all_chunks()
        assert isinstance(result, list)

def test_get_all_chunks_on_error_returns_empty(storage):
    with patch("core.storage.SessionLocal") as mock_session:
        mock_session.return_value.__enter__.side_effect = Exception("fail")
        result = storage.get_all_chunks()
        assert result == []