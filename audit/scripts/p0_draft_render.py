"""Phase 0: render selected pages of the draft PDF to PNG so figures can be inspected (read-only on the PDF).

Uses pypdfium2 installed with `pip install --target audit/tools/pylib pypdfium2` (no repo venv touched).
Usage: python audit/scripts/p0_draft_render.py [OUTDIR] [pages...]
Default OUTDIR: audit/phase0/draft_pages ; default pages: the five figure pages 2 6 9 10 11.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "audit", "tools", "pylib"))
import pypdfium2 as pdfium  # noqa: E402

PDF = os.path.join(REPO, "full paper draft for this experiments", "main (1).pdf")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "audit", "phase0", "draft_pages")
pages = [int(x) for x in sys.argv[2:]] or [2, 6, 9, 10, 11]
os.makedirs(out, exist_ok=True)
doc = pdfium.PdfDocument(PDF)
for p in pages:
    img = doc[p - 1].render(scale=2.0).to_pil()
    path = os.path.join(out, f"page{p:02d}.png")
    img.save(path)
    print("wrote", path, img.size)
