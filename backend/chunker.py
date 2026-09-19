"""
Text chunking with overlap.

Splits extracted page texts into smaller, overlapping chunks while
preserving source metadata (filename, page number) for citations.
"""


def chunk_documents(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:
    """
    Split page-level texts into overlapping chunks.

    Uses a sliding window so that sentences at chunk boundaries
    appear in at least one chunk in full.

    Args:
        pages:      List of page dicts from document_loader.
        chunk_size: Target characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts::

            [
                {
                    "chunk_id": "paper1.pdf_p1_c0",
                    "text": "chunk content...",
                    "metadata": {
                        "filename": "paper1.pdf",
                        "page_number": 1,
                        "chunk_index": 0
                    }
                },
                ...
            ]
    """
    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        chunk_index = 0
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Don't create tiny trailing fragments (< 50 chars)
            if len(chunk_text) < 50 and chunks:
                # Merge with previous chunk from the same page
                if chunks[-1]["metadata"]["filename"] == metadata["filename"]:
                    chunks[-1]["text"] += " " + chunk_text
                break

            chunk_id = (
                f"{metadata['filename']}_p{metadata['page_number']}_c{chunk_index}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "filename": metadata["filename"],
                    "page_number": metadata["page_number"],
                    "chunk_index": chunk_index
                }
            })

            # Slide the window forward (chunk_size - overlap)
            start += chunk_size - overlap
            chunk_index += 1

    return chunks


def get_chunking_stats(chunks: list[dict]) -> dict:
    """
    Return summary statistics about the chunks.

    Useful for verifying chunking is working as expected.
    """
    if not chunks:
        return {"total_chunks": 0}

    lengths = [len(c["text"]) for c in chunks]
    filenames = set(c["metadata"]["filename"] for c in chunks)

    return {
        "total_chunks": len(chunks),
        "total_documents": len(filenames),
        "documents": list(filenames),
        "avg_chunk_length": round(sum(lengths) / len(lengths)),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }
