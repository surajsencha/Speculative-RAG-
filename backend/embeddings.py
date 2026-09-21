"""
Sentence-transformer embeddings.

Encodes text into 384-dimensional vectors using the all-MiniLM-L6-v2 model.
Vectors are L2-normalized so that dot-product == cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model ID. Default produces 384-dim vectors.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (N, dimension), dtype float32.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        ).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: The question / search string.

        Returns:
            numpy array of shape (1, dimension), dtype float32.
        """
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")
