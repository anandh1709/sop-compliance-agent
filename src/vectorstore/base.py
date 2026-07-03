"""
src/vectorstore/base.py

The VectorStore contract. Everything downstream (RAG, agents, API) talks
to this abstract interface, not to FAISS or Azure directly. That's how we
keep the vector backend swappable.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SearchResult:
    """One retrieved chunk with its similarity score and source metadata."""
    chunk_id: int
    text: str
    page: int
    score: float  # higher = more similar (cosine similarity, 0..1)


class VectorStore(ABC):
    """
    Abstract vector store. Concrete implementations (FaissStore,
    AzureSearchStore) inherit from this and implement add + search.

    We use ABC rather than typing.Protocol because we want the interface
    enforced at subclass definition time — if someone forgets to implement
    search(), Python raises immediately, not at runtime.
    """

    @abstractmethod
    def add(self, chunks: List[Dict]) -> None:
        """
        Embed and store a list of chunk dicts (from ingest.load_and_chunk).
        Each chunk must have keys: chunk_id (int), text (str), page (int).
        """
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Return the top_k chunks most similar to `query`, ranked by score
        descending.
        """
        ...