"""Build audit/claims/<id>.md from the batch CSV row plus the pasted script outputs.

Usage: python _mknote.py <id> <heading text file> <output file> [<output file> ...]
The heading file holds the free-text part (claim, method, verdict-with-reason); each output file
is pasted verbatim under its own fenced block, so the note carries the actual script output.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.abspath(os.path.join(HERE, "..", "..", "phase2", "claims_batch_P3-kdbank-cap.csv"))
OUTDIR = os.path.abspath(os.path.join(HERE, "..", "..", "claims"))
os.makedirs(OUTDIR, exist_ok=True)

cid, head = sys.argv[1], sys.argv[2]
outs = sys.argv[3:]
row = next(r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["id"] == cid)

parts = [open(head, encoding="utf-8").read().rstrip(), "", "## The ledger row", ""]
for k in ("source_ref", "evidence_file", "evidence_key", "version_used", "verdict",
          "confidence", "importance", "script"):
    parts.append(f"- **{k}**: {row[k]}")
parts += ["", "## Script output (verbatim)", ""]
for o in outs:
    parts += [f"### `{os.path.basename(o)}`", "", "```", open(o, encoding="utf-8").read().rstrip(), "```", ""]
open(os.path.join(OUTDIR, f"{cid}.md"), "w", encoding="utf-8").write("\n".join(parts) + "\n")
print("wrote", os.path.join(OUTDIR, f"{cid}.md"))
