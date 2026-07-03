"""
src/agents/graph.py

The LangGraph wiring: retrieve -> extract -> check -> emit.

Even for two agents this is more scaffolding than raw sequential
Python would need. The payoff is that state is explicit, transitions
are typed, and adding a third node later (e.g. a human-in-the-loop
review step, a rewriter for ambiguous rules) is a graph edge, not
a refactor.
"""
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from src.llm.base import LLMClient
from src.vectorstore.base import VectorStore, SearchResult
from src.agents.schemas import WorkflowRule, CheckResult, ComplianceReport
from src.agents.extractor import extract_rules
from src.agents.checker import check_all


class ComplianceState(TypedDict):
    """
    The shared state passed between nodes. Every node reads what it
    needs and writes what it produces. Making state explicit is what
    makes the graph replayable and inspectable — you can rerun any
    single node given the state that entered it.
    """
    # Inputs
    queries: List[str]           # what to retrieve for
    checklist: List[str]         # what to check against
    # Intermediates
    retrieved_chunks: List[SearchResult]
    extracted_rules: List[WorkflowRule]
    # Output
    checks: List[CheckResult]


def build_graph(store: VectorStore, llm: LLMClient, top_k: int = 3):
    """
    Assemble the graph. The store and llm are captured by closures
    inside each node — a lightweight dependency-injection pattern
    that keeps the node functions pure w.r.t. state.
    """

    def retrieve_node(state: ComplianceState) -> ComplianceState:
        """
        Run each query against the vector store, dedupe by chunk_id,
        and cap the total number of chunks sent downstream. This keeps
        the extractor's prompt within Groq's free-tier TPM limit
        (6000 tokens/min on llama-3.1-8b-instant).
        """
        seen: dict[int, SearchResult] = {}
        for q in state["queries"]:
            for hit in store.search(q, top_k=top_k):
                if hit.chunk_id not in seen or hit.score > seen[hit.chunk_id].score:
                    seen[hit.chunk_id] = hit
        # Rank by score, keep the top 8 across all queries. At ~500
        # words × ~1.3 tokens/word that's ~5200 tokens of context,
        # comfortably under Groq's 6000 TPM cap.
        top_chunks = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:6]
        state["retrieved_chunks"] = top_chunks
        return state

    def extract_node(state: ComplianceState) -> ComplianceState:
        state["extracted_rules"] = extract_rules(state["retrieved_chunks"], llm)
        return state

    def check_node(state: ComplianceState) -> ComplianceState:
        state["checks"] = check_all(
            state["checklist"],
            state["extracted_rules"],
            llm,
        )
        return state

    graph = StateGraph(ComplianceState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract", extract_node)
    graph.add_node("check", check_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "check")
    graph.add_edge("check", END)

    return graph.compile()


# Standard queries used to seed retrieval. These deliberately map
# 1-to-1 onto the compliance checklist topics — so the extractor
# sees chunks relevant to what the checker will ask about.
DEFAULT_QUERIES = [
    "adverse event reporting timeframe",
    "informed consent documentation",
    "protocol deviation reporting",
    "drug and device accountability",
    "investigator responsibilities and delegation",
    "protocol amendments and IRB approval",
]


def run_compliance_check(
    store: VectorStore,
    llm: LLMClient,
    checklist: List[str],
    queries: List[str] = None,
    top_k: int = 3,
) -> ComplianceReport:
    """End-to-end: build the graph, run it, return the final report."""
    graph = build_graph(store, llm, top_k=top_k)
    initial: ComplianceState = {
        "queries": queries or DEFAULT_QUERIES,
        "checklist": checklist,
        "retrieved_chunks": [],
        "extracted_rules": [],
        "checks": [],
    }
    final = graph.invoke(initial)
    return ComplianceReport(
        extracted_rules=final["extracted_rules"],
        checks=final["checks"],
    )


if __name__ == "__main__":
    # Full end-to-end smoke test: PDF -> index -> graph -> report.
    from pathlib import Path
    from src.ingest import load_and_chunk
    from src.vectorstore.faiss_store import FaissStore
    from src.llm.groq_client import GroqClient
    from src.checklist import DEFAULT_CHECKLIST

    print("=== Building FAISS index ===")
    pdf = Path(__file__).parent.parent.parent / "data" / "SOP.pdf"
    chunks = load_and_chunk(str(pdf))
    store = FaissStore()
    store.add(chunks)
    print(f"Indexed {len(chunks)} chunks.\n")

    llm = GroqClient()

    print("=== Running compliance graph ===\n")
    report = run_compliance_check(
        store=store,
        llm=llm,
        checklist=DEFAULT_CHECKLIST,
    )

    print(f"--- Extracted {len(report.extracted_rules)} rules ---")
    for r in report.extracted_rules:
        deadline = f" [{r.deadline}]" if r.deadline else ""
        who = f" ({r.responsible_party})" if r.responsible_party else ""
        page = f" p.{r.source_page}" if r.source_page else ""
        print(f"  {r.rule_id}: {r.trigger} -> {r.action}{deadline}{who}{page}")

    print(f"\n--- Compliance check ({len(report.checks)} items) ---")
    for c in report.checks:
        icon = {"satisfied": "✓", "partial": "~", "missing": "✗"}[c.status]
        rules_note = f" [via {', '.join(c.evidence_rule_ids)}]" if c.evidence_rule_ids else ""
        print(f"  {icon} {c.status.upper()}: {c.requirement}")
        print(f"      {c.reasoning}{rules_note}")