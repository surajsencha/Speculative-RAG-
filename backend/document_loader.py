"""
PDF text extraction using PyMuPDF.

Takes a PDF file path, reads every page, and returns a list of
page-text dicts with metadata (filename, page number).
"""

import os
import pymupdf  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one entry per page.

    Args:
        file_path: Absolute or relative path to a PDF file.

    Returns:
        List of dicts, one per non-empty page::

            [
                {
                    "text": "page content...",
                    "metadata": {
                        "filename": "paper1.pdf",
                        "page_number": 1
                    }
                },
                ...
            ]
    """
    doc = pymupdf.open(file_path)
    filename = os.path.basename(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text:  # skip empty pages
            continue

        pages.append({
            "text": text,
            "metadata": {
                "filename": filename,
                "page_number": page_num + 1  # 1-indexed
            }
        })

    doc.close()
    return pages


def extract_text_from_multiple_pdfs(file_paths: list[str]) -> list[dict]:
    """
    Extract text from multiple PDFs.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of page dicts from all PDFs.
    """
    all_pages = []
    for path in file_paths:
        pages = extract_text_from_pdf(path)
        all_pages.extend(pages)
    return all_pages
