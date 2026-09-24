"""Extend audit/claims_inventory.csv with claim-like statements from the two sources the first pass skipped:
  H-*  the collaborator's 17 result notes on origin/feat/persona-pc-ood (read with `git show`, no checkout)
  O-*  the owner's other documents: docs/thesis/*, COLLABORATOR_BRIEFING, ROADMAP_TO_DEFENCE, NOVELTY_ASSESSMENT,
       REVIEW_REQUEST_POOLING, kd/docs/REPORT.md, kd/docs/NEXT.md, cap/docs/DESIGN.md
  M-*  the owner's memory notes (Q1-Q11 and the rest) — DOC-ONLY, they are leads, not evidence

A line counts as claim-like when it carries a number AND a comparative/verdict word. Capped per document so the
inventory stays usable; the cap and the rule are recorded in the `status` column as CANDIDATE.
"""
import csv, os, re, subprocess

CMP = re.compile(r"\b(beats?|higher|lower|better|worse|clears?|survives?|inverts?|fails?|holds?|vs\.?|than|"
                 r"improve[sd]?|drops?|rises?|falls?|gains?|loses?|significant|median|mean|AUROC|recall|accuracy)\b", re.I)
NUM = re.compile(r"\d\.\d{2,4}|\b\d{1,3}\s?%|[+-]\d\.\d+")
CAP = 40


def lines_of(text):
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip().lstrip("|>-*# ").strip()
        if 40 <= len(s) <= 400 and NUM.search(s) and CMP.search(s):
            yield i, re.sub(r"\s+", " ", s)


rows = []
his = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/feat/persona-pc-ood"],
                     capture_output=True, text=True).stdout.splitlines()
his_md = [p for p in his if p.endswith(".md") and "/results/" in p]
n = 0
for p in his_md:
    txt = subprocess.run(["git", "show", f"origin/feat/persona-pc-ood:{p}"], capture_output=True).stdout.decode("utf-8", "replace")
    for k, (ln, s) in enumerate(lines_of(txt)):
        if k >= CAP: break
        n += 1
        rows.append(dict(id=f"H-{n:03d}", source="collaborator note", source_ref=f"{p}:{ln}",
                         claim_short=s[:200], numbers="; ".join(NUM.findall(s)[:6]), layer="", dataset_hint="", status="CANDIDATE"))

owner_docs = ["docs/COLLABORATOR_BRIEFING.md", "docs/ROADMAP_TO_DEFENCE.md", "docs/NOVELTY_ASSESSMENT.md",
              "docs/REVIEW_REQUEST_POOLING.md", "kd/docs/REPORT.md", "kd/docs/NEXT.md", "cap/docs/DESIGN.md",
              "cap/docs/SARCASM_DESIGN.md"] + [f"docs/thesis/{f}" for f in os.listdir("docs/thesis") if f.endswith(".md")]
m = 0
for p in owner_docs:
    if not os.path.exists(p): continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    for k, (ln, s) in enumerate(lines_of(txt)):
        if k >= CAP: break
        m += 1
        rows.append(dict(id=f"O-{m:03d}", source="owner doc", source_ref=f"{p}:{ln}",
                         claim_short=s[:200], numbers="; ".join(NUM.findall(s)[:6]), layer="", dataset_hint="", status="CANDIDATE"))

MEM = os.path.expanduser("~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory")
q = 0
for f in sorted(os.listdir(MEM)):
    if not f.endswith(".md") or f == "MEMORY.md": continue
    txt = open(os.path.join(MEM, f), encoding="utf-8", errors="replace").read()
    for k, (ln, s) in enumerate(lines_of(txt)):
        if k >= 8: break
        q += 1
        rows.append(dict(id=f"M-{q:03d}", source="memory note", source_ref=f"memory/{f}:{ln}",
                         claim_short=s[:200], numbers="; ".join(NUM.findall(s)[:6]), layer="", dataset_hint="", status="CANDIDATE"))

path = "audit/claims_inventory.csv"
existing = list(csv.DictReader(open(path, encoding="utf-8")))
keep = [r for r in existing if not r["id"].startswith(("H-", "O-", "M-"))]
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "source", "source_ref", "claim_short", "numbers", "layer", "dataset_hint", "status"])
    w.writeheader()
    for r in keep + rows: w.writerow(r)
print(f"inventory now {len(keep) + len(rows)} rows (was {len(existing)}): +{n} collaborator-note, +{m} owner-doc, +{q} memory candidates")
