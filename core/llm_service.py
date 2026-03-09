import os
import time
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
from core.retriever import RetrievedChunk
from core.pipeline import QueryResult

@dataclass
class CitedSource:
    """O sursă citată în răspuns, cu chunk-ul original și numărul de citare."""
    citation_number: int          # [1], [2], [3] în textul răspunsului
    chunk_id: str
    document_id: str
    chunk_index: int
    text_excerpt: str             # primele 150 de caractere din chunk
    relevance_score: float        # scorul din reranker/retriever

    def __repr__(self):
        return (
            f"[{self.citation_number}] score={self.relevance_score:.4f} "
            f"| {self.text_excerpt[:80]!r}..."
        )


@dataclass
class LLMResponse:
    #raspunsul complet de la LLM cu surse citate.
    question: str
    answer: str                          # răspunsul raw de la GPT
    answer_with_citations: str           # răspunsul cu [1], [2] inline
    sources: list[CitedSource]           # sursele citate
    model: str                           # ex: "gpt-4o-mini"
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metadata: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        pricing = {
            "gpt-4o":              {"input": 2.50,  "output": 10.00},
            "gpt-4o-mini":         {"input": 0.150, "output": 0.600},
            "gpt-4-turbo":         {"input": 10.00, "output": 30.00},
        }
        rates = pricing.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost  = (self.prompt_tokens     / 1_000_000) * rates["input"]
        output_cost = (self.completion_tokens / 1_000_000) * rates["output"]
        return round(input_cost + output_cost, 6)

    def format_sources(self) -> str:
        #formatează sursele ca text readable pentru UI.
        if not self.sources:
            return "No sources found."
        lines = ["**Sources:**\n"]
        for source in self.sources:
            lines.append(
                f"[{source.citation_number}] (score: {source.relevance_score:.4f})\n"
                f"    {source.text_excerpt}...\n"
            )
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"sources={len(self.sources)}, "
            f"cost=${self.cost_usd:.6f}, "
            f"{self.elapsed_seconds:.2f}s)"
        )




SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions \
based exclusively on the provided document chunks.

Rules:
- Answer ONLY using information from the provided chunks.
- Cite every claim using inline citation numbers like [1], [2], [3].
- Each citation number corresponds to the chunk number in the context.
- If multiple chunks support a claim, cite all of them: [1][3].
- If the answer is not found in the chunks, say exactly: "I could not find this information in the provided documents."
- Be concise and factual. Do not add information not present in the chunks.
- Write in the same language as the question."""

CONTEXT_TEMPLATE = """Here are the relevant document chunks:
{chunks}
---
Question: {question}
Answer with inline citations [1], [2], etc. referring to the chunk numbers above:"""





class LLMService:
    # Generează răspunsuri la întrebări folosind chunk-urile din pipeline
    # Fiecare chunk este numerotat ([1], [2], ...) și folosit ca context
    # Modelul citează sursele inline în răspuns (ex: text [1])
    # Se extrag citările și se returnează răspunsul + sursele folosite
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,        # temperatură mică = răspunsuri mai factuale
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key= to LLMService."
            )

        self._client = OpenAI(api_key=resolved_key)

    def answer(self, query_result: QueryResult) -> LLMResponse:
        # Generează un răspuns pentru un QueryResult din pipeline
        # Args: query_result – rezultatul returnat de RAGPipeline.query()
        # Returns: LLMResponse cu răspunsul generat și sursele citate
        if not query_result.chunks:
            return self._empty_response(query_result.query)

        return self._generate(
            question=query_result.query,
            chunks=query_result.chunks,
        )

    def answer_from_chunks( self,question: str, chunks: list[RetrievedChunk]) -> LLMResponse:
        # Generează răspuns direct din chunks, fără QueryResult
        if not chunks:
            return self._empty_response(question)

        return self._generate(question=question, chunks=chunks)

    def _generate(  self, question: str,chunks: list[RetrievedChunk]) -> LLMResponse:
        start = time.time()
        # construim contextul numerotat
        numbered_context = self._build_numbered_context(chunks)
        # construim prompt-ul final
        user_prompt = CONTEXT_TEMPLATE.format(
            chunks=numbered_context,
            question=question,
        )
        # apel OpenAI
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[ {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user",   "content": user_prompt}])

        raw_answer = response.choices[0].message.content or ""
        usage= response.usage

        # identificăm ce surse au fost citate în răspuns
        cited_sources = self._extract_cited_sources(raw_answer, chunks)

        elapsed = time.time() - start

        return LLMResponse(
            question=question,
            answer=raw_answer,
            answer_with_citations=raw_answer,   # răspunsul conține deja [1][2] inline
            sources=cited_sources,
            model=self.model,
            elapsed_seconds=elapsed,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            metadata={
                "num_chunks_provided": len(chunks),
                "num_sources_cited":   len(cited_sources),
            },
        )

    def _build_numbered_context(self, chunks: list[RetrievedChunk]) -> str:
        #construiește contextul numerotat pentru prompt.
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{i}] (score: {chunk.score:.4f})\n{chunk.text}"
            )
        return "\n\n".join(parts)

    def _extract_cited_sources(self,answer: str,  chunks: list[RetrievedChunk]) -> list[CitedSource]:
        import re
        cited_numbers = set(
            int(n) for n in re.findall(r"\[(\d+)\]", answer)
        )
        sources = []
        for citation_number in sorted(cited_numbers):
            chunk_index = citation_number - 1   # [1] → index 0

            if chunk_index < 0 or chunk_index >= len(chunks):
                continue # ignora citările care nu corespund niciunui chunk

            chunk = chunks[chunk_index]
            sources.append(
                CitedSource(
                    citation_number=citation_number,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text_excerpt=chunk.text[:150],
                    relevance_score=chunk.score,
                )
            )
        return sources

    def _empty_response(self, question: str) -> LLMResponse:
        #răspuns gol când nu există chunks relevante
        return LLMResponse(
            question=question,
            answer="I could not find this information in the provided documents.",
            answer_with_citations="I could not find this information in the provided documents.",
            sources=[],
            model=self.model,
            elapsed_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )