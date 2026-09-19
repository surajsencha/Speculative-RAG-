"""
Test Phase 1 (document_loader) and Phase 2 (chunker).

Uses the Speculative RAG paper PDF from the cloned repo as test input.
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.document_loader import extract_text_from_pdf
from backend.chunker import chunk_documents, get_chunking_stats


def main():
    # Use the Speculative RAG paper as test data
    test_pdf = os.path.join(
        os.path.dirname(__file__),
        "data", "uploads", "speculative_rag_paper.pdf"
    )

    if not os.path.exists(test_pdf):
        print("[ERROR] Test PDF not found at:", test_pdf)
        print("  Place any PDF in the project folder and update the path.")
        return

    # ---- Phase 1: PDF Extraction ----
    print("=" * 60)
    print("PHASE 1 -- PDF Text Extraction")
    print("=" * 60)

    pages = extract_text_from_pdf(test_pdf)

    print(f"  File:            {os.path.basename(test_pdf)}")
    print(f"  Pages extracted: {len(pages)}")
    print()

    # Show first page preview
    if pages:
        p = pages[0]
        print(f"  Page 1 metadata: {p['metadata']}")
        print(f"  Page 1 preview:  {p['text'][:150]}...")
        print()

    # ---- Phase 2: Chunking ----
    print("=" * 60)
    print("PHASE 2 -- Text Chunking")
    print("=" * 60)

    chunks = chunk_documents(pages, chunk_size=500, overlap=100)
    stats = get_chunking_stats(chunks)

    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Avg chunk length: {stats['avg_chunk_length']} chars")
    print(f"  Min chunk length: {stats['min_chunk_length']} chars")
    print(f"  Max chunk length: {stats['max_chunk_length']} chars")
    print()

    # Show first 3 chunks
    print("  First 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"    [{i}] id={chunk['chunk_id']}")
        print(f"        source: {chunk['metadata']['filename']}, "
              f"page {chunk['metadata']['page_number']}")
        print(f"        text:   {chunk['text'][:80]}...")
        print()

    # Verify overlap between consecutive chunks
    if len(chunks) >= 2:
        c0_end = chunks[0]["text"][-50:]
        c1_start = chunks[1]["text"][:50]
        has_overlap = c0_end[-30:] in chunks[1]["text"][:150]
        print(f"  Overlap check (chunk 0 end -> chunk 1 start): "
              f"{'PASS' if has_overlap else 'CHECK MANUALLY'}")
        print()

    print("=" * 60)
    print("Phase 1 + Phase 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
