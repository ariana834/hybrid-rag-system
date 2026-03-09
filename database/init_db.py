from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase
from database.base import Base
from database.session import engine
from database.models import DocumentORM, ChunkORM

def init_db():
    print("ENGINE URL:", engine.url)

    # Activează extensia pgvector
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")
if __name__ == "__main__":
    init_db()