"""
Test Phase 3 -- Embeddings + FAISS Vector Store + Retriever.

Loads the Speculative RAG paper, chunks it, embeds chunks into FAISS,
runs sample queries, and tests save/load persistence.
"""

import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.document_loader import extract_text_from_pdf
from backend.chunker import chunk_documents
from backend.retriever import Retriever
from backend.config import Config


def main():
    test_pdf = os.path.join("data", "uploads", "speculative_rag_paper.pdf")
    if not os.path.exists(test_pdf):
        print("[ERROR] PDF not found:", test_pdf)
        return

    # ---- Load & chunk (Phase 1+2, already tested) ----
    print("Loading and chunking PDF...")
    pages = extract_text_from_pdf(test_pdf)
    chunks = chunk_documents(pages, chunk_size=500, overlap=100)
    print(f"  {len(pages)} pages -> {len(chunks)} chunks\n")

    # ---- Phase 3: Embed + Index ----
    print("=" * 60)
    print("PHASE 3 -- Embeddings + FAISS + Retriever")
    print("=" * 60)

    print("\n[1/4] Creating embedding model (first run downloads ~90 MB)...")
    t0 = time.time()
    retriever = Retriever()  # auto-creates EmbeddingModel + VectorStore
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Embedding dimension: {retriever.embedding_model.dimension}")

    print(f"\n[2/4] Embedding {len(chunks)} chunks and adding to FAISS...")
    t0 = time.time()
    added = retriever.add_documents(chunks)
    embed_time = time.time() - t0
    print(f"  Added {added} chunks in {embed_time:.1f}s")
    print(f"  FAISS index size: {retriever.total_chunks}")

    # ---- Test queries ----
    print("\n[3/4] Running test queries...")
    test_queries = [
        "What is speculative RAG?",
        "How does the drafter model generate answers?",
        "What datasets were used for evaluation?",
        "What are the limitations of this approach?",
    ]

    for query in test_queries:
        t0 = time.time()
        results = retriever.retrieve(query, top_k=5)
        query_time = time.time() - t0

        print(f"\n  Q: \"{query}\"")
        print(f"  Retrieved {len(results)} chunks in {query_time*1000:.0f}ms")
        for i, r in enumerate(results[:3]):  # show top 3
            src = f"{r['metadata']['filename']}, p{r['metadata']['page_number']}"
            print(f"    [{i+1}] score={r['score']:.3f} | {src}")
            print(f"        {r['text'][:80]}...")

    # ---- Test save/load ----
    print(f"\n[4/4] Testing save/load persistence...")
    save_dir = Config.VECTOR_STORE_PATH
    retriever.save(save_dir)
    print(f"  Saved to {save_dir}")

    # Create a new retriever and load from disk
    retriever2 = Retriever()
    retriever2.load(save_dir)
    print(f"  Loaded: {retriever2.total_chunks} chunks in index")

    # Verify search still works after reload
    results2 = retriever2.retrieve("speculative RAG", top_k=3)
    print(f"  Search after reload: {len(results2)} results, "
          f"top score={results2[0]['score']:.3f}")

    match = (retriever.total_chunks == retriever2.total_chunks and
             len(results2) == 3)
    print(f"  Persistence check: {'PASS' if match else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Phase 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
