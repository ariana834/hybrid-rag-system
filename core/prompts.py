"""
Toate prompt-urile aplicației centralizate într-un singur loc.
Importă din llm_service.py în loc să folosești stringuri hardcodate acolo.
"""

from string import Template


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = """\
You are a precise document assistant. Your job is to answer questions \
based exclusively on the provided document chunks.

Rules:
- Answer ONLY using information from the provided chunks.
- Cite every claim using inline citation numbers like [1], [2], [3].
- Each citation number corresponds to the chunk number in the context.
- If multiple chunks support a claim, cite all of them: [1][3].
- If the answer is not found in the chunks, say exactly: \
"I could not find this information in the provided documents."
- Be concise and factual. Do not add information not present in the chunks.
- Write in the same language as the question.\
"""

SYSTEM_PROMPT_CONVERSATIONAL = """\
You are a helpful document assistant with memory of the conversation.
You answer questions based on document chunks provided and the conversation history.

Rules:
- Use conversation history to understand follow-up questions.
- Cite sources inline with [1], [2], etc.
- If the answer isn't in the chunks, say so clearly.
- Write in the same language as the question.\
"""

SYSTEM_PROMPT_SUMMARY = """\
You are a document summarization assistant.
Summarize the provided document chunks into a clear, concise summary.
Preserve key facts, numbers, and names. Write in the same language as the document.\
"""

RAG_TEMPLATE = Template("""\
Here are the relevant document chunks:

$chunks

---
Question: $question

Answer with inline citations [1], [2], etc. referring to the chunk numbers above:\
""")

CONVERSATIONAL_TEMPLATE = Template("""\
Conversation history:
$history

Here are the relevant document chunks:

$chunks

---
Current question: $question

Answer with inline citations [1], [2], etc.:\
""")

SUMMARY_TEMPLATE = Template("""\
Here are the document chunks to summarize:

$chunks

---
Provide a concise summary of the above content:\
""")

NO_ANSWER_FOUND = "I could not find this information in the provided documents."

NO_DOCUMENTS_UPLOADED = (
    "No documents have been uploaded yet. "
    "Please upload a PDF, DOCX, or TXT file to get started."
)

DOCUMENT_ALREADY_EXISTS = "This document has already been processed."

def build_rag_prompt(chunks_context: str, question: str) -> str:
    """Construiește promptul standard RAG."""
    return RAG_TEMPLATE.substitute(chunks=chunks_context, question=question)


def build_conversational_prompt(
    chunks_context: str,
    question: str,
    history: list[dict],
) -> str:
    """
    Construiește promptul cu istoric conversație.

    Args:
        history: listă de dict {"role": "user"/"assistant", "content": "..."}
    """
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history
    )
    return CONVERSATIONAL_TEMPLATE.substitute(
        history=history_text or "No previous conversation.",
        chunks=chunks_context,
        question=question,
    )


def build_summary_prompt(chunks_context: str) -> str:
    """Construiește promptul pentru sumarizare document."""
    return SUMMARY_TEMPLATE.substitute(chunks=chunks_context)


def format_chunks_for_prompt(chunks) -> str:
    """
    Formatează chunk-urile ca context numerotat pentru orice prompt.
    Folosit de build_rag_prompt și build_conversational_prompt.
    """
    if not chunks:
        return "No relevant chunks found."
    parts = [f"[{i}] (score: {c.score:.4f})\n{c.text}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join(parts)