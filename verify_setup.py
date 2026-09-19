"""Quick sanity check - verify all dependencies are importable."""
import pymupdf  # PyMuPDF (new import name)
import sentence_transformers
import faiss
import groq
import flask
import flask_cors
import numpy
import dotenv

print("[OK] All imports successful!")
print(f"   PyMuPDF:       {pymupdf.VersionBind}")
print(f"   SentenceTrans: {sentence_transformers.__version__}")
print(f"   FAISS:         OK")
print(f"   Groq:          {groq.__version__}")
print(f"   Flask:         {flask.__version__}")
print(f"   NumPy:         {numpy.__version__}")
print(f"   torch:         ", end="")
try:
    import torch
    print(torch.__version__)
except ImportError:
    print("not installed (optional on Windows)")

print("")
print("Phase 0 complete - environment is ready!")
