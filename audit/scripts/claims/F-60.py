"""F-60 - "the paper evaluates MASK, not HarmPressureChoice".

Re-counts, from the PDF itself (papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf):
  * mentions of harm-pressure-knowledge-report / HarmPressureKnowledge*  (ledger: 0)
  * mentions of HarmPressureChoice                                        (ledger: 2, both in Table 1)
  * mentions of Mask / MASK                                               (ledger: 33)
  * which dataset labels occur on the figure axes                         (ledger: ConvGame InstrDec Mask AILiar Roleplay)
Prints every hit with its page number so "both inside Table 1" can be judged.
"""
import re, sys
from pypdf import PdfReader

PDF = "papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf"
r = PdfReader(PDF)
pages = [p.extract_text() or "" for p in r.pages]
full = "\n".join(pages)
print(f"{PDF}: {len(r.pages)} pages, {len(full)} chars of extracted text")

def report(name, pat, flags=0):
    print(f"\n--- {name}   pattern={pat!r} ---")
    tot = 0
    for i, t in enumerate(pages, start=1):
        for m in re.finditer(pat, t, flags):
            tot += 1
            s = max(0, m.start()-70); e = min(len(t), m.end()+70)
            ctx = " ".join(t[s:e].split())
            print(f"  p{i:>2}  ...{ctx}...")
    print(f"  TOTAL = {tot}")
    return tot

n_kr  = report("HP-KR (harm.pressure.knowledge / HarmPressureKnowledge)",
               r"(harm[-_ ]?pressure[-_ ]?knowledge|HarmPressureKnowledge|knowledge[-_ ]?report)", re.I)
n_hpc = report("HarmPressureChoice / harm-pressure-choice",
               r"(HarmPressureChoice|harm[-_ ]?pressure[-_ ]?choice|HPC\b)")
n_mask = report("Mask (word-boundary, any case)", r"\bMASK\b|\bMask\b")
print("\n--- lowercase 'mask' occurrences (for completeness) ---")
print("  count =", len(re.findall(r"\bmask\b", full)))

print("\n--- dataset-label runs that look like figure axis labels ---")
for i, t in enumerate(pages, start=1):
    for m in re.finditer(r"ConvGame[^\n]{0,120}", t):
        print(f"  p{i:>2}  {' '.join(m.group(0).split())}")

print("\n--- lines mentioning 'Table 1' ---")
for i, t in enumerate(pages, start=1):
    for m in re.finditer(r"Table 1", t):
        s = max(0, m.start()-90); e = min(len(t), m.end()+120)
        print(f"  p{i:>2}  ...{' '.join(t[s:e].split())}...")
