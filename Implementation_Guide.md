# Speculative RAG — Complete Implementation Guide

> **Project:** SpecRAG Assistant — Document Research Assistant  
> **Paper:** Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting (ICLR 2025)  
> **Difficulty:** 4/10 | **Estimated Time:** 7–10 days  
> **Hardware:** Any laptop (no GPU required)

---

## Table of Contents

1. [Tech Stack & API Choice](#1-tech-stack--api-choice)
2. [Project Structure](#2-project-structure)
3. [Phase 0 — Environment Setup](#phase-0--environment-setup-30-min)
4. [Phase 1 — PDF Ingestion & Text Extraction](#phase-1--pdf-ingestion--text-extraction-day-1)
5. [Phase 2 — Text Chunking](#phase-2--text-chunking-day-1)
6. [Phase 3 — Embeddings & FAISS Vector Store](#phase-3--embeddings--faiss-vector-store-day-2)
7. [Phase 4 — Standard RAG Baseline](#phase-4--standard-rag-baseline-day-2-3)
8. [Phase 5 — Evidence Partitioning](#phase-5--evidence-partitioning-day-3)
9. [Phase 6 — Parallel Drafting](#phase-6--parallel-drafting-day-3-4)
10. [Phase 7 — Verification](#phase-7--verification-day-4)
11. [Phase 8 — Flask API Backend](#phase-8--flask-api-backend-day-5)
12. [Phase 9 — Web Frontend](#phase-9--web-frontend-day-5-6)
13. [Phase 10 — Experiments & Evaluation](#phase-10--experiments--evaluation-day-6-7)
14. [Phase 11 — Polish & Demo Prep](#phase-11--polish--demo-prep-day-7-8)
15. [Testing Checkpoints](#testing-checkpoints)
16. [Troubleshooting](#troubleshooting)

---

## 1. Tech Stack & API Choice

### Recommended Free Option: **Groq API**

Groq offers a generous free tier with fast inference — perfect for a student project.

| Role | Model | Why |
|------|-------|-----|
| **Drafter** (smaller, cheaper) | `llama-3.1-8b-instant` | Fast, small model — generates candidate drafts |
| **Verifier** (stronger, smarter) | `llama-3.3-70b-versatile` | Larger model — evaluates and picks the best draft |

**How to get a Groq API key (free):**
1. Go to https://console.groq.com
2. Sign up with Google/GitHub
3. Go to **API Keys** → **Create API Key**
4. Copy the key — it starts with `gsk_...`

> **Tip:** Groq free tier gives ~30 requests/minute and ~14,400 requests/day — more than enough for this project. If you hit rate limits, just add a small delay between requests.

### Full Tech Stack

| Component | Technology | Install Size |
|-----------|-----------|:------------:|
| Language | Python 3.10+ | — |
| PDF Processing | PyMuPDF (fitz) | ~30 MB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | ~90 MB |
| Vector Store | faiss-cpu | ~20 MB |
| LLM API | Groq (free) | ~1 MB |
| Async Execution | Python asyncio + aiohttp | built-in |
| Backend | Flask + Flask-CORS | ~2 MB |
| Frontend | HTML + CSS + JavaScript | — |
| Charts | Chart.js (CDN) | — |

---

## 2. Project Structure

```
SpecRAG-Assistant/
│
├── backend/
│   ├── app.py                  ← Flask API entry point
│   ├── config.py               ← All configurable settings
│   ├── document_loader.py      ← PDF text extraction (PyMuPDF)
│   ├── chunker.py              ← Text chunking with overlap
│   ├── embeddings.py           ← Sentence-transformer embeddings
│   ├── vector_store.py         ← FAISS index (add, search, save, load)
│   ├── retriever.py            ← Top-K chunk retrieval
│   ├── partitioner.py          ← Split chunks into evidence groups
│   ├── drafter.py              ← Parallel draft generation (Groq)
│   ├── verifier.py             ← Verification + best-draft selection (Groq)
│   ├── standard_rag.py         ← Standard RAG pipeline
│   ├── speculative_rag.py      ← Full Speculative RAG pipeline
│   └── evaluation.py           ← Metrics tracking (latency, quality)
│
├── frontend/
│   ├── index.html              ← Main web page
│   ├── style.css               ← Styling
│   └── script.js               ← Frontend logic (upload, query, display)
│
├── data/
│   ├── uploads/                ← Uploaded PDF files
│   └── vector_store/           ← Saved FAISS index + metadata
│
├── experiments/
│   └── results.json            ← Experiment history
│
├── requirements.txt
├── .env                        ← API keys (never commit this)
├── .gitignore
└── README.md
```

---

## Phase 0 — Environment Setup (30 min)

### Step 0.1 — Create the project folder

```bash
cd "c:\Users\sench\Desktop\Sem-5\AI\AI project"
mkdir SpecRAG-Assistant
cd SpecRAG-Assistant
mkdir backend frontend data data\uploads data\vector_store experiments
```

### Step 0.2 — Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 0.3 — Create `requirements.txt`

```
# PDF Processing
PyMuPDF>=1.24.0

# Embeddings
sentence-transformers>=2.2.0

# Vector Store
faiss-cpu>=1.7.4

# LLM API
groq>=0.9.0
aiohttp>=3.9.0

# Backend
flask>=3.0.0
flask-cors>=4.0.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.26.0
```

### Step 0.4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the `all-MiniLM-L6-v2` model (~90 MB). This happens automatically when you first create embeddings.

### Step 0.5 — Create `.env`

```dotenv
GROQ_API_KEY=gsk_your_key_here
DRAFTER_MODEL=llama-3.1-8b-instant
VERIFIER_MODEL=llama-3.3-70b-versatile
```

### Step 0.6 — Create `.gitignore`

```
venv/
__pycache__/
data/uploads/*
data/vector_store/*
experiments/results.json
.env
*.pyc
```

---

## Phase 1 — PDF Ingestion & Text Extraction (Day 1)

### Goal
Upload a PDF → Extract text with page-level metadata.

### File: `backend/document_loader.py`

**What it does:**
1. Takes a PDF file path
2. Uses PyMuPDF to read every page
3. Returns a list of `{ text, metadata: { filename, page_number } }` objects

**Key functions to implement:**

```python
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Input: path to a PDF file
    Output: list of dicts, one per page:
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
    doc = fitz.open(file_path)
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
```

**Test it:**
```python
# test_document_loader.py
from backend.document_loader import extract_text_from_pdf

pages = extract_text_from_pdf("test.pdf")
print(f"Extracted {len(pages)} pages")
print(f"Page 1 preview: {pages[0]['text'][:200]}")
print(f"Metadata: {pages[0]['metadata']}")
```

### ✅ Checkpoint
- [ ] Can load a PDF and extract text from every page
- [ ] Each page has correct filename and page_number metadata
- [ ] Empty pages are skipped

---

## Phase 2 — Text Chunking (Day 1)

### Goal
Split extracted text into smaller, overlapping chunks suitable for embedding.

### File: `backend/chunker.py`

**What it does:**
1. Takes the list of page-text dicts from Phase 1
2. Splits text into chunks of ~500 characters with ~100 character overlap
3. Each chunk retains its source metadata (filename + page number)

**Key function:**

```python
def chunk_documents(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Input: list of page dicts from document_loader
    Output: list of chunk dicts:
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
            
            # Don't create very small trailing chunks
            if len(chunk_text) < 50 and chunks:
                # Merge with previous chunk
                chunks[-1]["text"] += " " + chunk_text
                break
            
            chunk_id = f"{metadata['filename']}_p{metadata['page_number']}_c{chunk_index}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "filename": metadata["filename"],
                    "page_number": metadata["page_number"],
                    "chunk_index": chunk_index
                }
            })
            
            start += chunk_size - overlap  # slide window with overlap
            chunk_index += 1
    
    return chunks
```

**Why overlapping?** Overlapping chunks ensure that sentences at the boundaries aren't lost. If a key sentence spans two chunks, the overlap guarantees it appears fully in at least one chunk.

**Test it:**
```python
from backend.document_loader import extract_text_from_pdf
from backend.chunker import chunk_documents

pages = extract_text_from_pdf("test.pdf")
chunks = chunk_documents(pages, chunk_size=500, overlap=100)
print(f"Created {len(chunks)} chunks from {len(pages)} pages")
print(f"Chunk 0: {chunks[0]['text'][:100]}...")
print(f"Chunk 0 metadata: {chunks[0]['metadata']}")
```

### ✅ Checkpoint
- [ ] 10-page PDF produces ~20-40 chunks (depends on content density)
- [ ] Each chunk is roughly 400-500 characters
- [ ] Each chunk has correct source metadata
- [ ] Consecutive chunks overlap by ~100 characters

---

## Phase 3 — Embeddings & FAISS Vector Store (Day 2)

### Goal
Convert text chunks to vector embeddings and store them in FAISS for fast similarity search.

### File: `backend/embeddings.py`

**What it does:**
1. Loads the `all-MiniLM-L6-v2` sentence transformer model
2. Encodes a list of text strings into 384-dimensional vectors
3. Returns a numpy array of embeddings

**Implementation:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into vectors. Returns shape (N, 384)."""
        return self.model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=True
        ).astype('float32')
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Returns shape (1, 384)."""
        return self.model.encode(
            [query], 
            normalize_embeddings=True
        ).astype('float32')
```

> **Note:** `normalize_embeddings=True` is important — it makes cosine similarity equivalent to dot product, which is what FAISS `IndexFlatIP` uses.

### File: `backend/vector_store.py`

**What it does:**
1. Creates a FAISS inner-product index
2. Adds chunk embeddings with metadata
3. Searches for top-K similar chunks given a query embedding
4. Saves/loads the index to/from disk

**Implementation:**

```python
import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        self.chunks = []  # parallel list: chunks[i] matches index row i
    
    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """Add embeddings and their chunk data to the store."""
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 9) -> list[dict]:
        """
        Search for top_k most similar chunks.
        Returns list of chunk dicts with added 'score' field.
        """
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        
        return results
    
    def save(self, directory: str):
        """Save FAISS index + metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
    
    def load(self, directory: str):
        """Load FAISS index + metadata from disk."""
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
    
    @property
    def size(self) -> int:
        return self.index.ntotal
```

### File: `backend/retriever.py`

**Combines `EmbeddingModel` + `VectorStore` into a single interface.**

```python
from backend.embeddings import EmbeddingModel
from backend.vector_store import VectorStore

class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def add_documents(self, chunks: list[dict]):
        """Embed and store all chunks."""
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add(embeddings, chunks)
    
    def retrieve(self, query: str, top_k: int = 9) -> list[dict]:
        """Get top_k chunks most relevant to the query."""
        query_emb = self.embedding_model.encode_query(query)
        return self.vector_store.search(query_emb, top_k)
    
    def save(self, directory: str):
        self.vector_store.save(directory)
    
    def load(self, directory: str):
        self.vector_store.load(directory)
```

**Test it:**
```python
from backend.document_loader import extract_text_from_pdf
from backend.chunker import chunk_documents
from backend.embeddings import EmbeddingModel
from backend.vector_store import VectorStore
from backend.retriever import Retriever

# Load & chunk
pages = extract_text_from_pdf("test.pdf")
chunks = chunk_documents(pages)

# Embed & store
emb_model = EmbeddingModel()
store = VectorStore()
retriever = Retriever(emb_model, store)
retriever.add_documents(chunks)

# Search
results = retriever.retrieve("What are the limitations?", top_k=5)
for r in results:
    print(f"[Score: {r['score']:.3f}] {r['text'][:80]}...")
```

### ✅ Checkpoint
- [ ] Embedding model loads and encodes text into 384-dim vectors
- [ ] FAISS index can add and search vectors
- [ ] Retriever returns relevant chunks with scores
- [ ] Index can be saved to and loaded from disk

---

## Phase 4 — Standard RAG Baseline (Day 2-3)

### Goal
Build the basic RAG pipeline: retrieve chunks → send everything to one LLM → get answer. This is your **baseline** for comparison.

### File: `backend/standard_rag.py`

**What it does:**
1. Takes a question and retriever
2. Retrieves top-K chunks
3. Concatenates all chunks into one prompt
4. Sends to the **verifier** model (since standard RAG uses the strongest model)
5. Returns the answer with sources and latency

**Implementation:**

```python
import time
from groq import Groq

class StandardRAG:
    def __init__(self, retriever, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.retriever = retriever
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Build the standard RAG prompt with all chunks."""
        docs = ""
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            docs += f"[{i}] ({meta['filename']}, Page {meta['page_number']})\n"
            docs += f"{chunk['text']}\n\n"
        
        return f"""You are a research assistant. Answer the question based ONLY on the provided documents.
If the documents don't contain enough information, say so.

Documents:
{docs}

Question: {question}

Provide a detailed answer with citations. Format citations as (Source: filename, Page X)."""
    
    def answer(self, question: str, top_k: int = 9) -> dict:
        """
        Standard RAG pipeline.
        Returns dict with answer, sources, metrics, mode.
        """
        # Step 1: Retrieve
        t0 = time.time()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - t0
        
        # Step 2: Generate
        prompt = self._build_prompt(question, chunks)
        
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )
        generation_time = time.time() - t0
        
        answer_text = response.choices[0].message.content
        total_time = retrieval_time + generation_time
        
        # Extract unique sources
        sources = []
        seen = set()
        for chunk in chunks:
            key = (chunk["metadata"]["filename"], chunk["metadata"]["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": chunk["metadata"]["filename"],
                    "page_number": chunk["metadata"]["page_number"]
                })
        
        return {
            "answer": answer_text,
            "sources": sources,
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3)
            },
            "mode": "standard_rag"
        }
```

**Test it:**
```python
standard = StandardRAG(retriever, api_key="gsk_...")
result = standard.answer("What are the main contributions of this paper?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Latency: {result['metrics']['total_time']:.2f}s")
```

### ✅ Checkpoint
- [ ] Standard RAG returns a coherent answer grounded in the documents
- [ ] Answer includes source citations
- [ ] Latency metrics are recorded
- [ ] Works with 1 and multiple PDFs

---

## Phase 5 — Evidence Partitioning (Day 3)

### Goal
Split the retrieved top-K chunks into N groups (evidence subsets) for parallel drafting.

### File: `backend/partitioner.py`

**Implementation:**

```python
def partition_evidence(chunks: list[dict], num_groups: int = 3) -> list[list[dict]]:
    """
    Split chunks into num_groups evidence subsets using round-robin.
    
    Round-robin ensures diversity — each group gets a mix of 
    high-relevance and lower-relevance chunks.
    
    Example with 9 chunks and 3 groups:
        Group A gets chunks ranked 1st, 4th, 7th
        Group B gets chunks ranked 2nd, 5th, 8th
        Group C gets chunks ranked 3rd, 6th, 9th
    """
    groups = [[] for _ in range(num_groups)]
    
    for i, chunk in enumerate(chunks):
        groups[i % num_groups].append(chunk)
    
    # Remove empty groups (if fewer chunks than groups)
    groups = [g for g in groups if len(g) > 0]
    
    return groups
```

**Test it:**
```python
from backend.partitioner import partition_evidence

chunks = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(9)]
groups = partition_evidence(chunks, num_groups=3)
for i, group in enumerate(groups):
    print(f"Group {chr(65+i)}: {[c['chunk_id'] for c in group]}")
# Group A: ['c0', 'c3', 'c6']
# Group B: ['c1', 'c4', 'c7']
# Group C: ['c2', 'c5', 'c8']
```

### ✅ Checkpoint
- [ ] 9 chunks correctly split into 3 groups of 3
- [ ] Round-robin ensures diversity
- [ ] Handles uneven chunk counts gracefully
- [ ] Works with different `num_groups` (2, 3, 4)

---

## Phase 6 — Parallel Drafting (Day 3-4)

### Goal
Generate multiple candidate answers in parallel, one per evidence group, using the **drafter** model.

### File: `backend/drafter.py`

**Implementation:**

```python
import asyncio
import json
import time
from groq import AsyncGroq

class Drafter:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
    
    def _build_draft_prompt(self, question: str, evidence_group: list[dict]) -> str:
        """Build prompt for a single drafter."""
        docs = ""
        for i, chunk in enumerate(evidence_group, 1):
            meta = chunk["metadata"]
            docs += f"[{i}] ({meta['filename']}, Page {meta['page_number']})\n"
            docs += f"{chunk['text']}\n\n"
        
        return f"""You are a research assistant analyzing a specific subset of evidence documents.

Evidence Documents:
{docs}

Question: {question}

Based ONLY on the evidence above, provide your answer in this exact JSON format:
{{
    "answer": "Your detailed answer here, citing evidence as (Source: filename, Page X)",
    "evidence_used": [1, 3],
    "confidence": 0.82
}}

If the evidence doesn't contain relevant information, set confidence to a low value.
Respond with ONLY the JSON, no other text."""
    
    async def generate_draft(self, question: str, evidence_group: list[dict], draft_id: str) -> dict:
        """Generate a single draft from one evidence group."""
        client = AsyncGroq(api_key=self.api_key)
        prompt = self._build_draft_prompt(question, evidence_group)
        
        t0 = time.time()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        latency = time.time() - t0
        
        raw = response.choices[0].message.content
        
        # Parse JSON response
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            try:
                start = raw.index('{')
                end = raw.rindex('}') + 1
                parsed = json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                parsed = {
                    "answer": raw,
                    "evidence_used": [],
                    "confidence": 0.5
                }
        
        return {
            "draft_id": draft_id,
            "answer": parsed.get("answer", raw),
            "evidence_used": parsed.get("evidence_used", []),
            "confidence": parsed.get("confidence", 0.5),
            "latency": round(latency, 3)
        }
    
    async def generate_all_drafts(self, question: str, evidence_groups: list[list[dict]]) -> list[dict]:
        """Generate all drafts in parallel using asyncio.gather."""
        tasks = [
            self.generate_draft(question, group, chr(65 + i))
            for i, group in enumerate(evidence_groups)
        ]
        return await asyncio.gather(*tasks)
```

**Running async code from sync context:**
```python
import asyncio

drafter = Drafter(api_key="gsk_...")
drafts = asyncio.run(drafter.generate_all_drafts(question, evidence_groups))
```

### ✅ Checkpoint
- [ ] Single draft generation works with one evidence group
- [ ] Parallel drafting produces 3 drafts simultaneously
- [ ] Each draft has structured JSON output (answer + confidence)
- [ ] Parallel is faster than sequential (check timing)
- [ ] Handles JSON parsing errors gracefully

---

## Phase 7 — Verification (Day 4)

### Goal
Send all drafts + their evidence to the stronger **verifier** model, which selects/refines the best answer.

### File: `backend/verifier.py`

**Implementation:**

```python
import json
import time
from groq import Groq

class Verifier:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def _build_verification_prompt(self, question: str, drafts: list[dict], 
                                    evidence_groups: list[list[dict]]) -> str:
        """Build prompt for the verifier."""
        draft_sections = ""
        for draft, group in zip(drafts, evidence_groups):
            draft_sections += f"\n--- Draft {draft['draft_id']} (Confidence: {draft['confidence']}) ---\n"
            draft_sections += f"Answer: {draft['answer']}\n"
            draft_sections += f"Evidence used:\n"
            for chunk in group:
                meta = chunk["metadata"]
                draft_sections += f"  - ({meta['filename']}, Page {meta['page_number']}): {chunk['text'][:150]}...\n"
        
        return f"""You are a senior research verifier. You have received multiple draft answers 
to the same question, each based on a different subset of evidence.

Your job is to:
1. Evaluate each draft for accuracy, completeness, and evidence grounding
2. Select the best draft or combine the best parts from multiple drafts
3. Provide the final, refined answer with proper source citations

Question: {question}

{draft_sections}

Respond in this exact JSON format:
{{
    "final_answer": "Your refined, evidence-grounded answer with citations like (Source: filename, Page X)",
    "selected_draft": "A",
    "reasoning": "Brief explanation of why this draft was selected and how the answer was refined",
    "sources": [
        {{"filename": "paper1.pdf", "page_number": 7}},
        {{"filename": "paper2.pdf", "page_number": 3}}
    ]
}}"""
    
    def verify(self, question: str, drafts: list[dict], 
               evidence_groups: list[list[dict]]) -> dict:
        """Verify and select/refine the best draft."""
        prompt = self._build_verification_prompt(question, drafts, evidence_groups)
        
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )
        latency = time.time() - t0
        
        raw = response.choices[0].message.content
        
        # Parse JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                start = raw.index('{')
                end = raw.rindex('}') + 1
                parsed = json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                parsed = {
                    "final_answer": raw,
                    "selected_draft": "A",
                    "reasoning": "Could not parse verification response",
                    "sources": []
                }
        
        return {
            "final_answer": parsed.get("final_answer", raw),
            "selected_draft": parsed.get("selected_draft", "A"),
            "reasoning": parsed.get("reasoning", ""),
            "sources": parsed.get("sources", []),
            "latency": round(latency, 3)
        }
```

### File: `backend/speculative_rag.py`

**Full Speculative RAG pipeline orchestrator:**

```python
import asyncio
import time
from backend.partitioner import partition_evidence

class SpeculativeRAG:
    def __init__(self, retriever, drafter, verifier):
        self.retriever = retriever
        self.drafter = drafter
        self.verifier = verifier
    
    def answer(self, question: str, top_k: int = 9, num_drafts: int = 3) -> dict:
        """Full Speculative RAG pipeline."""
        
        # Step 1: Retrieve
        t0 = time.time()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - t0
        
        # Step 2: Partition
        t0 = time.time()
        groups = partition_evidence(chunks, num_groups=num_drafts)
        partitioning_time = time.time() - t0
        
        # Step 3: Draft (parallel)
        t0 = time.time()
        drafts = asyncio.run(self.drafter.generate_all_drafts(question, groups))
        drafting_time = time.time() - t0
        
        # Step 4: Verify
        t0 = time.time()
        result = self.verifier.verify(question, drafts, groups)
        verification_time = time.time() - t0
        
        total_time = retrieval_time + partitioning_time + drafting_time + verification_time
        
        return {
            "answer": result["final_answer"],
            "selected_draft": result["selected_draft"],
            "verification_reasoning": result["reasoning"],
            "sources": result["sources"],
            "drafts": [
                {
                    "draft_id": d["draft_id"],
                    "answer": d["answer"],
                    "confidence": d["confidence"],
                    "latency": d["latency"]
                }
                for d in drafts
            ],
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "partitioning_time": round(partitioning_time, 3),
                "drafting_time": round(drafting_time, 3),
                "verification_time": round(verification_time, 3),
                "total_time": round(total_time, 3)
            },
            "mode": "speculative_rag"
        }
```

**Test the full pipeline:**
```python
spec_rag = SpeculativeRAG(retriever, drafter, verifier)
result = spec_rag.answer("What are the limitations of this approach?")

print(f"Answer: {result['answer']}")
print(f"Selected Draft: {result['selected_draft']}")
print(f"Sources: {result['sources']}")
print(f"\nDrafts generated:")
for d in result['drafts']:
    print(f"  Draft {d['draft_id']}: confidence={d['confidence']}, latency={d['latency']:.2f}s")
print(f"\nTotal time: {result['metrics']['total_time']:.2f}s")
```

### ✅ Checkpoint
- [ ] Verifier correctly receives all drafts and evidence
- [ ] Verifier selects the best draft and optionally refines it
- [ ] Final answer includes source citations
- [ ] Full pipeline: retrieve → partition → draft (parallel) → verify works end-to-end
- [ ] Metrics are tracked at each stage

---

## Phase 8 — Flask API Backend (Day 5)

### Goal
Wrap everything in a Flask REST API.

### File: `backend/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DRAFTER_MODEL = os.getenv("DRAFTER_MODEL", "llama-3.1-8b-instant")
    VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "llama-3.3-70b-versatile")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
    VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K = 9
    NUM_DRAFTS = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### File: `backend/app.py`

**API Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/upload` | Upload PDF(s), extract, chunk, embed, store |
| `POST` | `/api/ask` | Ask a question (standard or speculative RAG) |
| `GET`  | `/api/documents` | List uploaded documents |
| `DELETE` | `/api/documents` | Clear all documents and reset index |
| `GET`  | `/api/experiments` | Get experiment history |

**Skeleton:**

```python
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from backend.config import Config
from backend.document_loader import extract_text_from_pdf
from backend.chunker import chunk_documents
from backend.embeddings import EmbeddingModel
from backend.vector_store import VectorStore
from backend.retriever import Retriever
from backend.drafter import Drafter
from backend.verifier import Verifier
from backend.standard_rag import StandardRAG
from backend.speculative_rag import SpeculativeRAG

app = Flask(__name__)
CORS(app)

# Global state
embedding_model = None
retriever = None
standard_rag = None
speculative_rag = None
uploaded_docs = []

def initialize_models():
    """Lazy initialization of models."""
    global embedding_model, retriever, standard_rag, speculative_rag
    
    if embedding_model is None:
        embedding_model = EmbeddingModel(Config.EMBEDDING_MODEL)
        store = VectorStore(dimension=embedding_model.dimension)
        retriever = Retriever(embedding_model, store)
        
        drafter = Drafter(api_key=Config.GROQ_API_KEY, model=Config.DRAFTER_MODEL)
        verifier = Verifier(api_key=Config.GROQ_API_KEY, model=Config.VERIFIER_MODEL)
        
        standard_rag = StandardRAG(retriever, api_key=Config.GROQ_API_KEY)
        speculative_rag = SpeculativeRAG(retriever, drafter, verifier)

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    initialize_models()
    # Accept files, extract text, chunk, embed, store
    # Return document info
    pass

@app.route('/api/ask', methods=['POST'])
def ask_question():
    initialize_models()
    data = request.json
    question = data.get("question")
    mode = data.get("mode", "speculative")  # "speculative" or "standard"
    top_k = data.get("top_k", Config.TOP_K)
    num_drafts = data.get("num_drafts", Config.NUM_DRAFTS)
    
    if mode == "standard":
        result = standard_rag.answer(question, top_k=top_k)
    else:
        result = speculative_rag.answer(question, top_k=top_k, num_drafts=num_drafts)
    
    return jsonify(result)

@app.route('/api/documents', methods=['GET'])
def list_documents():
    return jsonify({"documents": uploaded_docs})

@app.route('/api/documents', methods=['DELETE'])
def clear_documents():
    # Reset vector store and uploaded docs
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Test with curl:**
```bash
# Upload a PDF
curl -X POST -F "file=@test.pdf" http://localhost:5000/api/upload

# Ask a question (speculative mode)
curl -X POST -H "Content-Type: application/json" ^
  -d "{\"question\": \"What are the main findings?\", \"mode\": \"speculative\"}" ^
  http://localhost:5000/api/ask
```

### ✅ Checkpoint
- [ ] `/api/upload` accepts and processes PDFs
- [ ] `/api/ask` returns answers in both modes
- [ ] `/api/documents` lists uploaded files
- [ ] Error handling works (no PDF, empty question, API failure)
- [ ] CORS is enabled for frontend access

---

## Phase 9 — Web Frontend (Day 5-6)

### Goal
Build a clean, functional web UI for the document assistant.

### File: `frontend/index.html`

**Layout sections:**
1. **Header** — App title + description
2. **Document Upload Area** — Drag & drop or file picker for PDFs
3. **Uploaded Documents List** — Shows loaded files with chunk counts
4. **Question Input** — Text input + mode selector (Standard/Speculative) + Ask button
5. **Answer Display** — Final answer with sources
6. **Draft Comparison Panel** — Shows all 3 drafts side by side (speculative mode only)
7. **Metrics Panel** — Latency breakdown bar chart (use Chart.js CDN)
8. **Experiment History** — Table of past queries with metrics

**UI Wireframe:**

```
┌────────────────────────────────────────────────────────────┐
│  🔬 SpecRAG Assistant                                     │
│  Speculative RAG Document Research Assistant               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📄 Documents                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Drop PDFs here or click to upload                   │ │
│  └──────────────────────────────────────────────────────┘ │
│  ✅ paper1.pdf (12 pages, 24 chunks)                     │
│  ✅ paper2.pdf (8 pages, 16 chunks)                      │
│                                                            │
│  ❓ Ask a Question                                        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ What are the limitations of this proposed approach?  │ │
│  └──────────────────────────────────────────────────────┘ │
│  Mode: (•) Speculative RAG  ( ) Standard RAG     [ASK]   │
│  Top-K: [9]  Drafts: [3]                                  │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  📝 Answer                                                │
│  The proposed method has three major limitations...       │
│                                                            │
│  Sources:                                                  │
│  • paper1.pdf — Page 7                                    │
│  • paper2.pdf — Page 9                                    │
│                                                            │
│  Selected Draft: B                                        │
├────────────────────────────────────────────────────────────┤
│  📊 Draft Comparison  (speculative mode only)             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                    │
│  │ Draft A │ │ Draft B │ │ Draft C │                    │
│  │ conf:72%│ │ conf:91%│ │ conf:65%│                    │
│  │  ...    │ │  ★ ...  │ │  ...    │                    │
│  └─────────┘ └─────────┘ └─────────┘                    │
├────────────────────────────────────────────────────────────┤
│  ⏱️ Latency Metrics                                       │
│  Retrieval:     ████ 0.41s                                │
│  Drafting:      ██████████ 1.32s                          │
│  Verification:  ██████ 0.87s                              │
│  Total:         2.60s                                     │
└────────────────────────────────────────────────────────────┘
```

### File: `frontend/style.css`

Design guidelines:
- Dark theme with a professional color palette
- Card-based layout with subtle shadows
- Color coding: drafts in different accent colors (blue, green, purple)
- Responsive design (works on different screen sizes)
- Loading spinner while processing
- Smooth transitions for result reveal

### File: `frontend/script.js`

**Key functions to implement:**

```javascript
// Upload PDFs
async function uploadFiles(files) { ... }

// Ask question
async function askQuestion() { ... }

// Display answer with sources
function displayAnswer(result) { ... }

// Display draft comparison cards
function displayDrafts(drafts) { ... }

// Display latency chart (Chart.js)
function displayMetrics(metrics) { ... }

// Show/hide loading spinner
function setLoading(isLoading) { ... }
```

**Include Chart.js from CDN in index.html:**
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

### ✅ Checkpoint
- [ ] PDFs can be uploaded via drag-and-drop or file picker
- [ ] Uploaded documents are listed with page/chunk counts
- [ ] Questions can be asked in both modes
- [ ] Answer is displayed with source citations
- [ ] Draft comparison panel shows all 3 drafts (speculative mode)
- [ ] Latency chart renders correctly
- [ ] Loading spinner shows while processing

---

## Phase 10 — Experiments & Evaluation (Day 6-7)

### Goal
Run systematic experiments comparing Standard RAG vs Speculative RAG.

### File: `backend/evaluation.py`

```python
import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, results_path: str = "experiments/results.json"):
        self.results_path = results_path
        self.results = self._load()
    
    def _load(self) -> list:
        if os.path.exists(self.results_path):
            with open(self.results_path) as f:
                return json.load(f)
        return []
    
    def record(self, question: str, result: dict):
        """Save one experiment run."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "mode": result["mode"],
            "answer_preview": result["answer"][:200],
            "metrics": result["metrics"]
        }
        self.results.append(entry)
        self._save()
    
    def _save(self):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def compare(self) -> dict:
        """Compare Standard RAG vs Speculative RAG."""
        standard = [r for r in self.results if r["mode"] == "standard_rag"]
        speculative = [r for r in self.results if r["mode"] == "speculative_rag"]
        
        def avg_metric(runs, key):
            vals = [r["metrics"].get(key, 0) for r in runs]
            return round(sum(vals) / len(vals), 3) if vals else 0
        
        std_avg = avg_metric(standard, "total_time")
        spec_avg = avg_metric(speculative, "total_time")
        
        return {
            "standard_rag": {
                "avg_total_time": std_avg,
                "num_runs": len(standard)
            },
            "speculative_rag": {
                "avg_total_time": spec_avg,
                "avg_drafting_time": avg_metric(speculative, "drafting_time"),
                "avg_verification_time": avg_metric(speculative, "verification_time"),
                "num_runs": len(speculative)
            },
            "speedup": f"{std_avg / spec_avg:.2f}x" if spec_avg > 0 else "N/A"
        }
```

### Experiments to Run

| # | Experiment | What to Vary | Measure |
|---|-----------|-------------|---------|
| 1 | Standard vs Speculative | Mode | Total latency, answer quality |
| 2 | Number of drafts | 2, 3, 4 drafts | Latency, answer quality |
| 3 | Top-K retrieval | 3, 5, 9, 12 chunks | Retrieval quality, latency |
| 4 | Ablation | Remove components | Which components matter most |

**Prepare 10 test questions that you'll run through both modes.** Use questions of varying complexity — simple factual, comparison, multi-document synthesis.

### ✅ Checkpoint
- [ ] At least 10 queries tested on both modes
- [ ] Latency comparison data collected
- [ ] Results saved to experiments/results.json
- [ ] Comparison API endpoint works

---

## Phase 11 — Polish & Demo Prep (Day 7-8)

### Improvements to add:

1. **Error handling** — API key missing, PDF fails, Groq rate limit, empty responses
2. **Better prompts** — Test and refine drafter/verifier prompts for consistent JSON
3. **Caching** — Don't re-embed already-indexed documents
4. **UI polish** — Loading animations, toast notifications, keyboard shortcuts
5. **README.md** — Project description, setup instructions, screenshots, architecture diagram

### Demo Script (for presentation)

```
1. Start the app: python backend/app.py
2. Open browser: http://localhost:5000
3. Upload 2-3 research papers
4. Ask: "What are the major limitations across these papers?"
5. Show the Speculative RAG answer → point out sources + selected draft
6. Switch to Standard RAG mode → same question
7. Compare latency side-by-side
8. Show the experiment comparison chart
9. Discuss findings
```

---

## Testing Checkpoints Summary

Complete these in order. Each phase builds on the previous one:

```
Phase 0  [ ] Environment ready, dependencies installed
Phase 1  [ ] PDF → extracted text with page metadata
Phase 2  [ ] Text → overlapping chunks with source tracking
Phase 3  [ ] Chunks → embeddings → FAISS → similarity search works
Phase 4  [ ] Standard RAG: retrieve → generate → answer with sources
Phase 5  [ ] Evidence partitioning: 9 chunks → 3 groups of 3
Phase 6  [ ] Parallel drafting: 3 drafts generated simultaneously
Phase 7  [ ] Verification: best draft selected, final answer refined
Phase 8  [ ] Flask API: all endpoints working
Phase 9  [ ] Web UI: upload → ask → display answer → show metrics
Phase 10 [ ] Experiments: Standard vs Speculative comparison data
Phase 11 [ ] Polished, demo-ready
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: fitz` | Run `pip install PyMuPDF` (not `pip install fitz`) |
| FAISS import error on Windows | Use `pip install faiss-cpu` (not `faiss-gpu`) |
| Groq rate limit (429 error) | Add `time.sleep(2)` between API calls, or use exponential backoff |
| JSON parsing error from LLM | Wrap in try/except, extract JSON from response with regex if needed |
| Embedding model download slow | First run downloads ~90 MB; subsequent runs use cache |
| `asyncio.run()` error in Flask | Use `loop = asyncio.new_event_loop()` instead in threaded Flask |
| CORS error in browser | Ensure `CORS(app)` is set in Flask |
| Empty answer from drafter | Check if chunks actually contain relevant content; try increasing `top_k` |
| Port 5000 in use | Change to `app.run(port=5001)` |

---

## Quick Command Reference

```bash
# Activate environment
cd "c:\Users\sench\Desktop\Sem-5\AI\AI project\SpecRAG-Assistant"
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python backend/app.py

# Open in browser
# http://localhost:5000
```

---

> **Golden Rule:** Build incrementally. Get each phase working and tested before moving to the next. Don't try to build everything at once!
