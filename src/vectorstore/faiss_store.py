"""
src/vectorstore/faiss_store.py

Local FAISS-backed VectorStore. Good for dev and single-machine use.

Design notes:
  - Embedding model: sentence-transformers/all-MiniLM-L6-v2. Fast, small,
    good enough for English regulatory text. Swap to bge-large or a
    biomedical-specific model if retrieval quality proves insufficient.

  - Index type: IndexFlatIP with L2-normalised vectors. Inner product on
    unit vectors == cosine similarity. Exact search (no approximation),
    which at our scale (thousands of chunks) is milliseconds.

  - Persistence: index + metadata written to disk so the API doesn't have
    to re-embed on every restart. FAISS only holds vectors and integer
    ids; the chunk text + page number live in a separate JSON sidecar.
"""
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.vectorstore.base import VectorStore, SearchResult


class FaissStore(VectorStore):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = "data/faiss_index",
    ):
        # Loading the model downloads ~90MB on first run, then caches locally.
        self.model = SentenceTransformer(model_name)
        # The model tells us its own embedding dimension — don't hardcode 384,
        # so this class still works if we swap the model.
        self.dim = self.model.get_embedding_dimension()

        # Inner-product index. Combined with normalised vectors below,
        # this gives us cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)

        # Parallel list holding chunk metadata keyed by FAISS row index.
        # FAISS itself only stores vectors + integer positions, not text.
        self.metadata: List[Dict] = []

        self.index_dir = Path(index_dir)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings and L2-normalise them.
        Normalising means inner product == cosine similarity — which is
        what we want for semantic retrieval.
        """
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # normalize_L2 works in place on the numpy array.
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")  # FAISS requires float32

    def add(self, chunks: List[Dict]) -> None:
        """Embed and index a batch of chunks."""
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vecs = self._embed(texts)
        self.index.add(vecs)
        # Store metadata in the same order as we added vectors, so the
        # FAISS row index i maps to metadata[i].
        self.metadata.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve the top_k most similar chunks."""
        if self.index.ntotal == 0:
            return []
        query_vec = self._embed([query])
        # scores: (1, top_k) cosine similarities
        # ids:    (1, top_k) row indices into self.metadata
        scores, ids = self.index.search(query_vec, top_k)

        results: List[SearchResult] = []
        for score, row_id in zip(scores[0], ids[0]):
            # FAISS returns -1 for empty slots when top_k > ntotal
            if row_id == -1:
                continue
            meta = self.metadata[row_id]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                page=meta["page"],
                score=float(score),
            ))
        return results

    # --- Persistence ---
    # Kept as instance methods (not part of the base VectorStore protocol)
    # because Azure AI Search persists server-side — persistence isn't
    # part of the abstract contract, only of this local implementation.

    def save(self) -> None:
        """Write index + metadata to disk so the API boots instantly."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self) -> bool:
        """
        Load index + metadata from disk if they exist.
        Returns True on success, False if no persisted index found.
        """
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.json"
        if not (index_path.exists() and meta_path.exists()):
            return False
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True


if __name__ == "__main__":
    # Smoke test: ingest the SOP, build the index, run a couple of queries,
    # save to disk, load fresh, query again to confirm persistence works.
    from src.ingest import load_and_chunk

    pdf = Path(__file__).parent.parent.parent / "data" / "SOP.pdf"
    chunks = load_and_chunk(str(pdf))
    print(f"Ingested {len(chunks)} chunks.")

    store = FaissStore()
    print(f"Embedding model: {store.model} (dim={store.dim})")
    store.add(chunks)
    print(f"Index now holds {store.index.ntotal} vectors.\n")

    # Two representative queries against the SOP.
    for q in [
        "How quickly must adverse events be reported?",
        "What are the requirements for informed consent?",
    ]:
        print(f"Query: {q}")
        for r in store.search(q, top_k=3):
            print(f"  [chunk {r.chunk_id} p.{r.page} score={r.score:.3f}] "
                  f"{r.text[:120]}...")
        print()

    # Persist and reload to prove save/load works.
    store.save()
    print("Saved index to data/faiss_index/")

    fresh = FaissStore()
    assert fresh.load(), "Failed to load persisted index"
    print(f"Reloaded index: {fresh.index.ntotal} vectors, "
          f"{len(fresh.metadata)} metadata entries.")