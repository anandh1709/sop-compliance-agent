"""
src/vectorstore/azure_store.py

Azure AI Search-backed VectorStore. Same public interface as FaissStore
(add/search) — the rest of the codebase can't tell them apart.

Design notes:
  - Bring-your-own embeddings. We embed client-side with MiniLM (same
    model as FaissStore) so retrieval quality is identical across
    backends. Server-side integrated vectorisation is a prod upgrade
    but needs Azure OpenAI, which isn't on the Free tier.

  - HNSW vector search. Approximate nearest neighbours; the accuracy
    hit is invisible at our scale and speed is the point.

  - Idempotent index creation. ensure_index() creates the schema if
    missing, no-ops if it exists. Prototype deployment speed matters.

  - Schema future-proofed with a sop_id field so multi-SOP tenants
    work without a migration.
"""
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX,
    require,
)
from src.vectorstore.base import VectorStore, SearchResult


class AzureSearchStore(VectorStore):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sop_id: str = "default",
    ):
        # Same embedding model as FaissStore — swap-in retrieval parity.
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.sop_id = sop_id

        # Fail fast with clear message if env vars aren't set.
        self.endpoint = require("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT)
        self.api_key = require("AZURE_SEARCH_KEY", AZURE_SEARCH_KEY)
        self.index_name = AZURE_SEARCH_INDEX

        credential = AzureKeyCredential(self.api_key)

        # Two clients: one for index management (create/delete schemas),
        # one for document CRUD (add/search vectors). Separation of
        # concerns — Azure SDK models it this way too.
        self._index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=credential
        )
        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=credential,
        )

    # --- Index management ---

    def ensure_index(self) -> None:
        """
        Create the index if it doesn't exist. Idempotent — safe to call
        on every startup. This is what prototype-deployment-speed looks
        like: no manual portal clicks.
        """
        try:
            self._index_client.get_index(self.index_name)
            return  # Already exists, nothing to do.
        except ResourceNotFoundError:
            pass  # Fall through to create it.

        # HNSW config — the vector search algorithm. Parameters are
        # Azure's defaults; tune only if retrieval quality demands it.
        hnsw_config = HnswAlgorithmConfiguration(name="hnsw-default")

        # A profile ties an algorithm config to a field. Lets you
        # define multiple algorithms and swap per-field.
        vector_profile = VectorSearchProfile(
            name="vector-profile-default",
            algorithm_configuration_name="hnsw-default",
        )

        vector_search = VectorSearch(
            algorithms=[hnsw_config],
            profiles=[vector_profile],
        )

        fields = [
            # Azure requires the key field to be a string.
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            # Multi-SOP support baked into the schema from day one.
            SimpleField(
                name="sop_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Integer field so we can filter/sort by page in future.
            SimpleField(
                name="page",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            # Full-text searchable — enables future hybrid search
            # (vector + BM25) without a schema migration.
            SearchableField(name="text", type=SearchFieldDataType.String),
            # The actual embedding vector.
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.dim,
                vector_search_profile_name="vector-profile-default",
            ),
        ]

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )
        self._index_client.create_index(index)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts. No L2 normalisation — HNSW cosine handles it."""
        vecs = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.tolist()

    # --- VectorStore interface ---

    def add(self, chunks: List[Dict]) -> None:
        """
        Embed chunks and upload as documents to the Azure index.
        Batched at 100 per call to stay well within rate limits.
        """
        if not chunks:
            return
        self.ensure_index()

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        documents = [
            {
                "chunk_id": f"{self.sop_id}-{c['chunk_id']}",  # unique per SOP
                "sop_id": self.sop_id,
                "page": c["page"],
                "text": c["text"],
                "embedding": emb,
            }
            for c, emb in zip(chunks, embeddings)
        ]

        BATCH = 100
        for i in range(0, len(documents), BATCH):
            self._search_client.upload_documents(documents=documents[i : i + BATCH])

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Vector similarity search against the Azure index."""
        query_vec = self._embed([query])[0]

        vector_query = VectorizedQuery(
            vector=query_vec,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        # search_text=None → pure vector search, no BM25 component.
        # (Hybrid search would set search_text=query, which combines
        # keyword and vector scoring — a great prod upgrade.)
        results_iter = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["chunk_id", "sop_id", "page", "text"],
            top=top_k,
        )

        results: List[SearchResult] = []
        for hit in results_iter:
            # "chunk_id" here is the composite "sop_id-N" we stored.
            # Extract the numeric part for the SearchResult contract.
            composite = hit["chunk_id"]
            numeric_id = int(composite.rsplit("-", 1)[-1])
            results.append(SearchResult(
                chunk_id=numeric_id,
                text=hit["text"],
                page=hit["page"],
                score=float(hit["@search.score"]),
            ))
        return results


if __name__ == "__main__":
    # Smoke test: same shape as faiss_store's — ingest, index, query.
    # Proves the two backends are drop-in interchangeable.
    from pathlib import Path
    from src.ingest import load_and_chunk

    pdf = Path(__file__).parent.parent.parent / "data" / "SOP.pdf"
    chunks = load_and_chunk(str(pdf))
    print(f"Ingested {len(chunks)} chunks.\n")

    store = AzureSearchStore(sop_id="johns-hopkins-irb-sop")
    print(f"Endpoint: {store.endpoint}")
    print(f"Index: {store.index_name}")
    print("Ensuring index exists...")
    store.ensure_index()
    print("Uploading documents...")
    store.add(chunks)
    print(f"Uploaded {len(chunks)} chunks.\n")

    # Azure's indexer needs a beat to make new docs searchable.
    # A short sleep on first run avoids empty results.
    import time
    time.sleep(3)

    for q in [
        "How quickly must adverse events be reported?",
        "What are the requirements for informed consent?",
    ]:
        print(f"Query: {q}")
        for r in store.search(q, top_k=3):
            print(f"  [chunk {r.chunk_id} p.{r.page} score={r.score:.3f}] "
                  f"{r.text[:120]}...")
        print()