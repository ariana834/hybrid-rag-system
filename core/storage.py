import numpy as np
from typing import Optional

from database.repositories import DocumentRepository, ChunkRepository
from database.session import SessionLocal
from models.document import Document
from models.chunk import Chunk


class StorageService:
    #responsabil cu persistarea documentelor și chunk-urilor în PostgreSQL.
    def save_document_with_chunks(self,document: Document,  chunks: list[Chunk], embeddings: np.ndarray,) -> Optional[str]:
        #salvează un document împreună cu chunk-urile și embedding-urile sale.

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        with SessionLocal() as session:
            try:
                doc_repo = DocumentRepository(session)
                chunk_repo = ChunkRepository(session)

                document_id = doc_repo.add_document(document)
                chunk_repo.add_chunks(chunks, embeddings)

                print(f"[Storage] Saved document '{document.filename}' "
                      f"with {len(chunks)} chunks. ID: {document_id}")

                return document_id

            except Exception as error:
                session.rollback()
                print(f"[Storage] Error saving document '{document.filename}': {error}")
                return None

    def document_exists(self, filename: str) -> bool:
        #Verifică dacă un document cu același nume există deja în DB.

        with SessionLocal() as session:
            try:
                from sqlalchemy import select
                from database.models import DocumentORM

                result = session.execute(select(DocumentORM).where(DocumentORM.filename == filename)).first()
                return result is not None

            except Exception as error:
                print(f"[Storage] Error checking document existence: {error}")
                return False

    def get_all_chunks(self) -> list:
        #Returnează toate chunk-urile din DB.
        #Folosit de BM25Retriever pentru a-și construi indexul.

        with SessionLocal() as session:
            try:
                from sqlalchemy import select
                from database.models import ChunkORM
                chunks = session.execute(select(ChunkORM)).scalars().all()

                # Detașăm obiectele de sesiune ca să poată fi folosite după închiderea ei
                result = []
                for chunk in chunks:
                    result.append({
                        "id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "start_sentence": chunk.start_sentence,
                        "end_sentence": chunk.end_sentence,
                        "metadata": chunk.metadata_json,
                    })

                return result

            except Exception as error:
                print(f"[Storage] Error fetching chunks: {error}")
                return []

    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        #Căutare semantică (vector similarity) în DB folosind pgvector.
        #Returnează lista de ChunkORM ordonate după distanță cosinus.
        with SessionLocal() as session:
            try:
                chunk_repo = ChunkRepository(session)
                chunks = chunk_repo.semantic_search(query_embedding, top_k=top_k)

                result = []
                for chunk in chunks:
                    result.append({
                        "id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "start_sentence": chunk.start_sentence,
                        "end_sentence": chunk.end_sentence,
                        "metadata": chunk.metadata_json,
                    })

                return result

            except Exception as error:
                print(f"[Storage] Error during semantic search: {error}")
                return []