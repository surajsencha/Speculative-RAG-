"""
FAISS vector store.

Stores chunk embeddings in a flat inner-product index and provides
add / search / save / load operations.
"""

import json
import os

import faiss
import numpy as np


class VectorStore:
    """In-memory FAISS index with parallel metadata list."""

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Embedding vector size (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # inner product
        self.chunks: list[dict] = []  # chunks[i] matches index row i

    # ── mutate ───────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and their chunk metadata to the store.

        Args:
            embeddings: (N, dimension) float32 array.
            chunks:     List of chunk dicts (same length as embeddings).
        """
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self.index.reset()
        self.chunks.clear()

    # ── query ────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 9) -> list[dict]:
        """
        Return the top_k most similar chunks.

        Args:
            query_embedding: (1, dimension) float32 array.
            top_k:           Number of results to return.

        Returns:
            List of chunk dicts, each with an added ``score`` field,
            ordered from most to least similar.
        """
        # Clamp top_k to the number of stored vectors
        top_k = min(top_k, self.index.ntotal)
        if top_k == 0:
            return []

        scores, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for missing results
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── persistence ──────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Write the FAISS index and chunk metadata to *directory*."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self, directory: str) -> None:
        """Load a previously saved index + metadata from *directory*."""
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # ── helpers ──────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self.index.ntotal
