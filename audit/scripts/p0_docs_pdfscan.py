"""Phase 0: catalogue PDFs (read-only). For each tracked PDF outside submodules: pages, bytes, git dates,
PDF metadata; for the non-paper draft PDF also the first ~40 lines of text and counts of F-ids / Q-ids /
'Claim N'. Paper PDFs in papers/ are NOT read beyond metadata + page count.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_pdfscan.py"""
import os, re, subprocess
import pypdf
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8").stdout
pdfs = [f for f in git("ls-files", "*.pdf").splitlines() if not re.match(r"(mask|pacchiardi|soft-trigger|assistant_axis)/", f)]
pdfs = [f for f in pdfs if not f.startswith("figures/")]
for f in pdfs:
    r = pypdf.PdfReader(f)
    meta = r.metadata or {}
    log = git("log", "--format=%h %ad %an", "--date=short", "--", f).splitlines()
    print(f"{f}\t{len(r.pages)}p\t{os.path.getsize(f)}B\tfirst={log[-1] if log else ''}\tlast={log[0] if log else ''}\t"
          f"title={meta.get('/Title','')!s:.60}\tcreator={meta.get('/Creator','')!s:.30}\tcreated={meta.get('/CreationDate','')}")
    if f.startswith("full paper draft"):
        txt = "\n".join((p.extract_text() or "") for p in r.pages)
        print("  --- draft text: chars", len(txt))
        print("  F-ids:", sorted({int(x) for x in re.findall(r"\bF-(\d{1,3})\b", txt)}))
        print("  Q-ids:", sorted({int(x) for x in re.findall(r"\bQ(\d{1,2})\b", txt)}))
        print("  Claim N:", sorted(set(re.findall(r"\bClaim\s+\d+", txt))), " Result N:", sorted(set(re.findall(r"\bResult\s+\d+", txt))))
        print("  section headings (lines starting with a number + capital):")
        for l in txt.splitlines():
            if re.match(r"^\s*(\d+(\.\d+)*)\s+[A-Z][A-Za-z].{2,80}$", l) and len(l) < 90:
                print("    ", l.strip())
        print("  --- first 40 lines:")
        for l in txt.splitlines()[:40]:
            print("    |", l[:160])
