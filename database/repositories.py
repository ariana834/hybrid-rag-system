import uuid
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import DocumentORM, ChunkORM
from models.document import Document
from models.chunk import Chunk


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_document(self, document: Document) -> str:
        db_document = DocumentORM(
            id=uuid.UUID(document.id),
            filename=document.filename,
            file_type=document.file_type,
            content=document.content,
            num_characters=document.num_characters,
        )
        self.session.add(db_document)
        self.session.commit()
        return str(db_document.id)


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            db_chunk = ChunkORM(
                id=uuid.uuid4(),
                document_id=uuid.UUID(chunk.document_id),
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                start_sentence=chunk.start_sentence,
                end_sentence=chunk.end_sentence,
                metadata_json=chunk.metadata,
                embedding=embedding.tolist(),
            )
            self.session.add(db_chunk)

        self.session.commit()

    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ChunkORM]:
        statement = (
            select(ChunkORM)
            .order_by(ChunkORM.embedding.cosine_distance(query_embedding.tolist()))
            .limit(top_k)
        )
        return list(self.session.scalars(statement).all())