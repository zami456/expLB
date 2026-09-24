"""Map every DOC_FIXES entry to priority + the documents it touches.

Mechanical: parses audit/DOC_FIXES.md, splits on '### DF-NN', pulls the priority
tag from the header when present, and regex-extracts every file path mentioned in
the body. Output: audit/phase6/df_triage.csv
"""
import re, csv, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "audit" / "DOC_FIXES.md"
OUT = ROOT / "audit" / "phase6" / "df_triage.csv"

text = SRC.read_text(encoding="utf-8")
parts = re.split(r"^### (DF-\d+)", text, flags=re.M)
# parts[0] = preamble, then alternating id, body

PATH_RE = re.compile(
    r"`?((?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.(?:md|py|ps1|json|csv|html|yaml|pt|jsonl|pdf))"
)
# documents the owner would EDIT (prose), vs evidence files she would not
PROSE = (".md", ".html")

rows = []
for i in range(1, len(parts), 2):
    did = parts[i]
    body = parts[i + 1]
    header = body.split("\n", 1)[0]
    m = re.search(r"·\s*(P[123])\s*·", header)
    prio = m.group(1) if m else ""
    title = re.sub(r"\*\*|\[lead QA[^\]]*\]|·\s*P[123]\s*·", " ", header)
    title = re.sub(r"\s+", " ", title).strip(" ·-")

    paths = []
    for p in PATH_RE.findall(body):
        if p not in paths:
            paths.append(p)
    docs = [p for p in paths if p.endswith(PROSE)]
    # ledger / memory / skill mentions that are not full paths
    if re.search(r"\bF-\d+\b", body) and "kd/docs/FINDINGS.md" not in docs:
        if re.search(r"F-\d+(?:'s)?\s+(?:entry|item|caveat|says|writes|quotes|states|calls|prints)", body) \
           or re.search(r"\bthe F-\d+ entry\b", body):
            docs.insert(0, "kd/docs/FINDINGS.md")
    if re.search(r"memory\s+`|memory note|`MEMORY\.md", body) and "MEMORY.md" not in docs:
        docs.append("MEMORY.md (auto-memory)")
    applied = bool(re.search(r"\*\*APPLIED\b|\*\*Done:\*\*", body))
    rows.append(
        dict(df=did, priority=prio, applied="YES" if applied else "",
             n_docs=len(docs), docs="; ".join(docs[:6]), title=title[:150])
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["df", "priority", "applied", "n_docs", "docs", "title"])
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} entries -> {OUT}")
from collections import Counter
print("priority:", Counter(r["priority"] or "(none)" for r in rows).most_common())
print("marked applied:", [r["df"] for r in rows if r["applied"]])
doc_count = Counter()
for r in rows:
    for d in r["docs"].split("; "):
        if d:
            doc_count[d] += 1
print("\ntop target documents:")
for d, c in doc_count.most_common(25):
    print(f"  {c:3d}  {d}")
