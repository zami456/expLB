"""F-70 - novelty audit record. Checks the checkable parts:
  (a) docs/NOVELTY_ASSESSMENT.md really says 2 of 10 claims were called novel by BOTH reviewers,
      and names the same disagreement pair the ledger reports;
  (b) the MASK / Liars' Bench attribution correction, from the two PDFs in papers/;
  (c) whether the three papers F-70's caveat says are "not in papers/" are there now, and since when.
"""
import re, subprocess, sys
from pypdf import PdfReader

print("=== (a) NOVELTY_ASSESSMENT.md, the rows the ledger summarises ===")
txt = open("docs/NOVELTY_ASSESSMENT.md", encoding="utf-8").read()
for pat in [r"Safe to present as contributions.*", r"\*\*5 \(empirical half\)\*\*.*", r"\*\*8\*\*.*",
            r"Must be demoted.*", r"\*\*7\*\* .*", r"\*\*6\*\* .*", r"\*\*10\*\* .*", r"\*\*1\*\* .*", r"\*\*9\*\* .*",
            r"\*\*Claim 3\*\*.*", r"\*\*Claim 4\*\*.*"]:
    for m in re.finditer(pat, txt):
        print("  " + " ".join(m.group(0).split())[:300])

print("\n=== (b) authorship, first page of each PDF ===")
for f, n in [("papers/The MASK Benchmark Disentangling.pdf", 700),
             ("papers/2511.16035v2 (1).pdf", 700)]:
    t = PdfReader(f).pages[0].extract_text()
    print(f"\n-- {f}")
    print("  " + " ".join(t[:n].split()))

print("\n=== (c) the three papers F-70's caveat calls absent ===")
import os, glob
want = {"2511.17408": "The Impact of Off-Policy Training Data.pdf",
        "2605.27958": "Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf",
        "2609.02893": "Probe Generalization as Subspace Selection for OOD Deception Detectio.pdf"}
for arx, fn in want.items():
    p = os.path.join("papers", fn)
    ex = os.path.exists(p)
    add = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %cs", "--", p],
                         capture_output=True).stdout.decode().strip().splitlines()
    print(f"  {arx}: present={ex}  added={add[:1]}")
    if ex:
        t = PdfReader(p).pages[0].extract_text() or ""
        ids = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", t))
        print(f"      first-page arXiv ids: {sorted(ids)}")
