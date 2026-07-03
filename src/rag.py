"""
src/rag.py

The basic retrieval-augmented generation loop:
  question -> retrieve top-k chunks -> build prompt -> LLM -> answer

Deliberately no agents here. This is the substrate the LangGraph
agents (Step 4) sit on top of. Keeping it standalone means we can
test retrieval quality and generation quality independently of
the orchestration layer.
"""
from dataclasses import dataclass
from typing import List

from src.vectorstore.base import VectorStore, SearchResult
from src.llm.base import LLMClient


SYSTEM_PROMPT = """You are a regulatory compliance analyst.

Answer the user's question using ONLY the context excerpts provided.
Each excerpt is labelled with its source page number.

RULES:
1. If the context does not contain the information needed to answer,
   respond exactly: "The provided SOP does not cover this."
   Do not guess. Do not fill gaps from general knowledge.
2. When you cite a fact, include the page number in square brackets,
   e.g. [p.12].
3. Be concise. Regulatory answers should be direct, not padded."""


@dataclass
class RagAnswer:
    """Structured answer with the chunks used, for auditability."""
    question: str
    answer: str
    sources: List[SearchResult]


def build_prompt(question: str, chunks: List[SearchResult]) -> str:
    """
    Assemble the user-role prompt: labelled context blocks, then the
    question at the bottom.

    Why context first, question last: models attend most strongly to
    the start and end of the input ('lost in the middle' effect).
    Putting the question at the bottom keeps it in the model's
    immediate attention when generating.
    """
    context_parts = []
    for c in chunks:
        # Labelling each chunk with its page lets the model produce
        # accurate [p.N] citations in its answer.
        context_parts.append(f"[Excerpt from page {c.page}]\n{c.text}")
    context = "\n\n---\n\n".join(context_parts)

    return (
        f"CONTEXT:\n{context}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the context above. Cite page numbers as [p.N]."
    )


def answer_question(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    top_k: int = 3,
) -> RagAnswer:
    """
    End-to-end RAG: retrieve, prompt, generate.
    Works with any VectorStore (FAISS or Azure) and any LLMClient.
    """
    chunks = store.search(question, top_k=top_k)
    if not chunks:
        # Empty index — usually means someone forgot to run add().
        return RagAnswer(
            question=question,
            answer="No documents have been indexed yet.",
            sources=[],
        )

    prompt = build_prompt(question, chunks)
    answer_text = llm.generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,  # deterministic for compliance
    )

    return RagAnswer(
        question=question,
        answer=answer_text.strip(),
        sources=chunks,
    )


if __name__ == "__main__":
    # Smoke test the full RAG loop against both backends.
    # We do FAISS first (fastest, no network), then Azure to prove
    # the answer is stable across vector stores.
    from pathlib import Path
    from src.ingest import load_and_chunk
    from src.vectorstore.faiss_store import FaissStore
    from src.llm.groq_client import GroqClient

    questions = [
        "How quickly must adverse events be reported to the IRB?",
        "What must be documented when obtaining informed consent?",
        "What is the process for reporting protocol deviations?",
        # A deliberately out-of-scope question — the model should
        # refuse to answer rather than hallucinate.
        "What is the maximum allowable dose of paracetamol for adults?",
    ]

    print("=== Building FAISS index ===")
    pdf = Path(__file__).parent.parent / "data" / "SOP.pdf"
    chunks = load_and_chunk(str(pdf))
    store = FaissStore()
    store.add(chunks)
    print(f"Indexed {len(chunks)} chunks.\n")

    llm = GroqClient()
    print(f"LLM: {llm.model}\n")

    for q in questions:
        result = answer_question(q, store, llm, top_k=3)
        print(f"Q: {result.question}")
        print(f"A: {result.answer}")
        print(f"Sources: pages {sorted({s.page for s in result.sources})}")
        print("-" * 70)