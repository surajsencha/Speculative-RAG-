"""
Centralized configuration for the SpecRAG Assistant.
All tuneable settings live here — change them in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM API ──────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "llama-3.1-8b-instant")
    VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "llama-3.3-70b-versatile")

    # ── Paths ────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")
    EXPERIMENTS_PATH = os.path.join(BASE_DIR, "experiments", "results.json")

    # ── Chunking ─────────────────────────────────────────────
    CHUNK_SIZE = 500          # characters per chunk
    CHUNK_OVERLAP = 100       # overlap between consecutive chunks

    # ── Embeddings ───────────────────────────────────────────
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    # ── Retrieval ────────────────────────────────────────────
    TOP_K = 9                 # number of chunks to retrieve

    # ── Speculative RAG ──────────────────────────────────────
    NUM_DRAFTS = 3            # number of parallel drafts

    # ── Flask ────────────────────────────────────────────────
    FLASK_PORT = 5000
    FLASK_DEBUG = True
