<div align="center">

# documind-ai
**Upload a document. Ask anything. Every answer cites its source.**

</div>

---

**documind-ai** is a full-stack Retrieval-Augmented Generation (RAG) system built entirely from scratch in Python.

Drop in a PDF, DOCX, or TXT file. The system parses it, splits it into semantically coherent chunks, embeds each one, and stores them in a PostgreSQL vector database. When you ask a question, it runs two retrieval strategies in parallel — dense vector search and BM25 keyword matching — fuses the results using Reciprocal Rank Fusion, re-scores the top candidates with a cross-encoder, and generates a precise answer with inline citations.

No black boxes. No wrapper libraries. Every component — the chunker, the BM25 index, the RRF fusion, the reranker — is hand-implemented and fully tested.

---

# Demo

Below is a quick walkthrough of the system in action.

---

## Upload a document

Upload a **PDF, DOCX, or TXT** file through the interface.  
The system parses the document, extracts the text, and splits it into **semantic chunks** that are later embedded and indexed.

<p align="center">
<img src="assets/upload.png" width="450">
</p>

After processing, the UI displays the number of chunks created and the total processing time.

---

## Retrieval modes

The system supports multiple retrieval strategies that can be tested independently.

<p align="center">
<img src="assets/retrieval_modes.png" width="450">
</p>

| Mode | Description |
|-----|-------------|
| `semantic_only` | Dense vector similarity search |
| `bm25_only` | Keyword-based lexical retrieval |
| `hybrid` | Combines semantic and BM25 retrieval using Reciprocal Rank Fusion |
| `hybrid_rerank` | Hybrid retrieval followed by cross-encoder reranking |

Switch between retrieval strategies directly from the sidebar without restarting the application.

---

## Ask questions about your documents

Users can ask natural language questions about the uploaded document.

<p align="center">
<img src="assets/chat.png" width="750">
</p>

The system:

1. Retrieves relevant document chunks using the selected retrieval strategy  
2. Combines semantic and keyword results using **Reciprocal Rank Fusion**  
3. Optionally reranks the top candidates using a **cross-encoder model**  
4. Generates a final answer with **inline citations**

Each response also displays:

- retrieved source passages  
- relevance scores  
- retrieval mode used  
- token usage  
- estimated cost per query  

---

# How it works

```
DOCUMENT PIPELINE

document
 ├─ parse      → PDF / DOCX / TXT
 ├─ chunk      → semantic splitting (cosine similarity, not fixed size)
 ├─ embed      → sentence-transformers/all-MiniLM-L6-v2
 └─ store      → PostgreSQL + pgvector


QUERY PIPELINE

query
 ├─ retrieve   → semantic search + BM25 (hand-written, no libs)
 ├─ fuse       → Reciprocal Rank Fusion
 ├─ rerank     → cross-encoder/ms-marco-MiniLM-L-6-v2
 └─ generate   → GPT-4o-mini → answer with inline citations [1][2]
```

---

# Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | all-MiniLM-L6-v2 (384 dimensions) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM | GPT-4o-mini |
| Infrastructure | Docker + docker-compose |

---

# Retrieval modes

| Mode | Description |
|---|---|
| `semantic_only` | Nearest-neighbor vector search |
| `bm25_only` | TF-IDF keyword matching |
| `hybrid` | Semantic + BM25 fused with RRF |
| `hybrid_rerank` | Hybrid → cross-encoder re-scores top-20 candidates ← **default** |

Switch between modes at runtime from the sidebar — no restart required.

---

# Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/documind-ai
cd documind-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```python
# config.py
DATABASE_URL   = "postgresql+psycopg2://postgres:pass@localhost:5433/documind"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL   = "gpt-4o-mini"
```

### 3. Start the database

```bash
docker-compose up -d
python database/init_db.py
```

### 4. Run

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# Project structure

```
documind-ai/
│
├── core/
│   ├── parser.py            Document extraction (PDF, DOCX, TXT)
│   ├── chunker.py           Semantic chunking via cosine similarity
│   ├── embeddings.py        Sentence-transformers wrapper
│   ├── bm25_retriever.py    BM25 implemented from scratch
│   ├── hybrid_retriever.py  RRF fusion of semantic + BM25
│   ├── reranker.py          Cross-encoder reranking
│   ├── pipeline.py          End-to-end orchestration
│   ├── llm_service.py       GPT with cited sources
│   ├── storage.py           Database read/write layer
│   └── prompts.py           Centralized prompt templates
│
├── database/                SQLAlchemy ORM + pgvector schema
├── models/                  Dataclasses (Chunk, Document, Query, Response)
├── tests/                   Pytest suite — no live database required
└── app.py                   Streamlit UI
```

---

# Tests

```bash
pytest tests/ -v
```

```
test_chunker.py       semantic splitting, cosine similarity
test_bm25.py          index building, scoring, normalization
test_hybrid.py        RRF formula, weight configuration
test_reranker.py      sigmoid normalization, rank ordering
test_models.py        dataclass validation
test_storage.py       mocked DB operations
test_llm_service.py   citation parsing, response structure
```

All tests are mocked — they run without a live database or API keys.

---

# Why from scratch?

Many RAG tutorials rely heavily on high-level frameworks that abstract away most of the underlying retrieval logic. While convenient, this often makes it difficult to understand how each component contributes to the final result.

This project takes a different approach: every major component of the retrieval pipeline is implemented explicitly in Python. The goal is to provide full transparency into how a modern RAG system works internally.

Key components implemented from scratch include:

• **BM25 retrieval** — a full implementation of TF-IDF based scoring with configurable *k1* and *b* parameters, without relying on external IR libraries.  

• **Semantic chunking** — documents are split based on semantic similarity between sentence embeddings, rather than fixed-size windows or naive text separators.  

• **Reciprocal Rank Fusion (RRF)** — a rank-based method for combining heterogeneous retrieval systems (dense and sparse) without requiring score normalization.  

• **Cross-encoder reranking** — candidate passages are re-evaluated using a model that jointly processes the query and document, significantly improving relevance over bi-encoder similarity scores.

This design makes the full retrieval pipeline transparent, modular, and easy to experiment with.