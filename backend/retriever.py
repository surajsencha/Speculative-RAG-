"""
Retriever — combines EmbeddingModel + VectorStore.

Provides a single interface to:
  1. Ingest chunks  (embed  ->  add to FAISS)
  2. Answer queries (embed query  ->  search FAISS  ->  return top-K chunks)
  3. Persist / reload the index
"""

from backend.embeddings import EmbeddingModel
from backend.vector_store import VectorStore


class Retriever:
    """High-level retrieval interface."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ):
        """
        Args:
            embedding_model: If *None*, a default ``all-MiniLM-L6-v2`` model
                             is created automatically.
            vector_store:    If *None*, an empty store is created.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore(
            dimension=self.embedding_model.dimension
        )

    # ── ingest ───────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> int:
        """
        Embed *chunks* and add them to the vector store.

        Args:
            chunks: List of chunk dicts (must have a ``"text"`` key).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add(embeddings, chunks)
        return len(chunks)

    # ── query ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 9) -> list[dict]:
        """
        Return the *top_k* chunks most relevant to *query*.

        Each returned dict contains the original chunk fields plus a
        ``score`` key (higher = more relevant).
        """
        query_emb = self.embedding_model.encode_query(query)
        return self.vector_store.search(query_emb, top_k)

    # ── persistence ──────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save the FAISS index + metadata to disk."""
        self.vector_store.save(directory)

    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        self.vector_store.load(directory)

    # ── helpers ──────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently indexed."""
        return self.vector_store.size

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self.vector_store.clear()
