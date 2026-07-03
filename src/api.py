"""
src/api.py

FastAPI service wrapping the compliance graph.

Endpoints:
  GET  /health              - liveness probe
  POST /check-compliance    - main endpoint: PDF upload or raw text -> report
  GET  /docs                - auto-generated Swagger UI (free from FastAPI)

The vector store backend is chosen at boot via VECTOR_STORE_BACKEND
(faiss | azure). The API code is backend-agnostic — that's the point
of the VectorStore protocol from Step 2.
"""
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field

from src.ingest import load_and_chunk, chunk_pages
from src.vectorstore.base import VectorStore
from src.vectorstore.faiss_store import FaissStore
from src.llm.base import LLMClient
from src.llm.groq_client import GroqClient
from src.checklist import DEFAULT_CHECKLIST
from src.agents.graph import run_compliance_check
from src.agents.schemas import WorkflowRule, CheckResult, ComplianceReport


# --- Backend selection ---------------------------------------------------

def _build_store() -> VectorStore:
    """
    Pick the vector store backend from env. Falls back to FAISS which
    has no external dependencies and works offline — sensible default
    for dev and CI.
    """
    backend = os.getenv("VECTOR_STORE_BACKEND", "faiss").lower()
    if backend == "azure":
        # Lazy import so the API can run without azure-search-documents
        # installed if the operator picks the FAISS backend.
        from src.vectorstore.azure_store import AzureSearchStore
        return AzureSearchStore()
    if backend == "faiss":
        return FaissStore()
    raise ValueError(
        f"Unknown VECTOR_STORE_BACKEND '{backend}'. "
        f"Expected 'faiss' or 'azure'."
    )


# --- Lifespan: load expensive resources once ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the embedding model, vector store, and LLM client at boot
    and stashes them on app.state. First request doesn't pay the
    cold-start cost. Clean shutdown on the yield's downside.
    """
    print("[startup] Loading vector store and LLM client...")
    app.state.store = _build_store()
    app.state.llm = GroqClient()
    print(f"[startup] Backend: {type(app.state.store).__name__} | "
          f"LLM: {app.state.llm.model}")
    yield
    print("[shutdown] Bye.")


app = FastAPI(
    title="SOP Compliance Agent",
    description=(
        "Multi-agent RAG system that translates regulatory SOPs into "
        "structured workflow rules and checks them against a compliance "
        "checklist. Built with LangGraph, FAISS/Azure AI Search, Groq, "
        "and FastAPI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- Response models ----------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    backend: str = Field(examples=["FaissStore"])
    model: str = Field(examples=["llama-3.1-8b-instant"])


class ComplianceResponse(BaseModel):
    """Top-level response for /check-compliance."""
    sop_id: str
    n_chunks_indexed: int
    extracted_rules: List[WorkflowRule]
    checks: List[CheckResult]
    summary: dict = Field(
        description="Counts of satisfied / partial / missing for a "
                    "quick scan by callers.",
        examples=[{"satisfied": 5, "partial": 1, "missing": 0}],
    )


# --- Helpers ------------------------------------------------------------

def _summarise(checks: List[CheckResult]) -> dict:
    """Roll up check statuses to a small dict for scannable UI display."""
    counts = {"satisfied": 0, "partial": 0, "missing": 0}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    return counts


def _chunks_from_pdf_bytes(pdf_bytes: bytes) -> list:
    """
    Save the PDF bytes to a temp file (PyMuPDF wants a filepath),
    ingest, then clean up. Using tempfile means we're safe against
    concurrent requests colliding on the same filename.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.close()
        return load_and_chunk(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _chunks_from_text(text: str) -> list:
    """
    Wrap raw text as a single 'page' and run it through the same
    chunker used for PDFs. Keeps the ingestion path uniform.
    """
    return chunk_pages([{"page": 1, "text": text}])


# --- Endpoints ----------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe. Also surfaces which backend is active."""
    return HealthResponse(
        status="ok",
        backend=type(app.state.store).__name__,
        model=app.state.llm.model,
    )


@app.post(
    "/check-compliance",
    response_model=ComplianceResponse,
    tags=["compliance"],
    summary="Extract workflow rules from an SOP and check compliance.",
)
async def check_compliance(
    sop_id: str = Form(
        default="uploaded-sop",
        description="Identifier for this SOP; useful when multiple "
                    "SOPs share the same index.",
    ),
    pdf: Optional[UploadFile] = File(
        default=None,
        description="Optional PDF upload. Provide either pdf OR text.",
    ),
    text: Optional[str] = Form(
        default=None,
        description="Optional raw SOP text. Provide either pdf OR text.",
    ),
) -> ComplianceResponse:
    """
    Ingest an SOP, index it, run the multi-agent compliance graph,
    return extracted rules + compliance check results.

    Accepts either a PDF upload OR raw text. Exactly one is required.
    """
    if (pdf is None) == (text is None):
        # Both None or both provided — same error, same message.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of 'pdf' or 'text'.",
        )

    # --- Ingest ---
    try:
        if pdf is not None:
            if not pdf.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file must be a .pdf",
                )
            chunks = _chunks_from_pdf_bytes(await pdf.read())
        else:
            if not text or not text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="'text' cannot be empty.",
                )
            chunks = _chunks_from_text(text)
    except HTTPException:
        raise
    except Exception as e:
        # Ingestion failures usually mean a corrupt PDF. Don't leak
        # internal traceback to the client.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest input: {e.__class__.__name__}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content could be extracted from the input.",
        )

    # --- Index ---
    # We build a fresh store per request rather than sharing one. The
    # extra ~1 sec per request is worth the isolation: no cross-tenant
    # bleed, no stale index from a previous SOP. For Azure this would
    # be more expensive (index-per-request would spam the service);
    # in a real deployment we'd namespace documents by sop_id in a
    # shared index instead. Left as a followup in the README.
    store = FaissStore() if isinstance(app.state.store, FaissStore) else _build_store()
    store.add(chunks)

    # --- Run graph ---
    try:
        report: ComplianceReport = run_compliance_check(
            store=store,
            llm=app.state.llm,
            checklist=DEFAULT_CHECKLIST,
        )
    except Exception as e:
        # Graph failure could be a rate limit, malformed LLM output,
        # or a network blip. Surface the class name, not the message,
        # to avoid leaking API-key-adjacent strings.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Compliance graph failed: {e.__class__.__name__}",
        )

    return ComplianceResponse(
        sop_id=sop_id,
        n_chunks_indexed=len(chunks),
        extracted_rules=report.extracted_rules,
        checks=report.checks,
        summary=_summarise(report.checks),
    )