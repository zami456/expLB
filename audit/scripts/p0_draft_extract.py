"""Phase 0: extract text + metadata from the candidate draft PDF (read-only).

Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_draft_extract.py
Writes audit/phase0/draft_text.txt (per-page text with page markers) and prints metadata.
"""
import hashlib
import os
import sys

from pypdf import PdfReader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(REPO, "full paper draft for this experiments", "main (1).pdf")
OUT = os.path.join(REPO, "audit", "phase0", "draft_text.txt")

with open(PDF, "rb") as fh:
    data = fh.read()
print("path:", PDF)
print("bytes:", len(data))
print("md5:", hashlib.md5(data).hexdigest())

r = PdfReader(PDF)
print("pages:", len(r.pages))
meta = r.metadata or {}
for k, v in dict(meta).items():
    print("meta", k, "=", v)

with open(OUT, "w", encoding="utf-8") as out:
    for i, p in enumerate(r.pages, 1):
        t = p.extract_text() or ""
        out.write(f"\n===== PAGE {i} =====\n")
        out.write(t)
        out.write("\n")
print("wrote", OUT)
sys.exit(0)
