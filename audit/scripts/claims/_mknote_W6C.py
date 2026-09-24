"""Generate audit/claims/<id>.md for every row of claims_batch_W6C-briefing.csv,
pasting the relevant slice of the script output that produced the verdict."""
import csv
import io
import os

OUT = "audit/phase2/out/W6C"

# id -> list of (output file, first line, last line) slices to paste, 1-based inclusive
SLICES = {
 "O-008": [("W6C_collab.txt", 1, 34)],
 "O-010": [("../P12/P12_lab.txt", 213, 242)],
 "O-013": [("../P12/P12_lab.txt", 213, 270), ("W6C_eleven.txt", 1, 20)],
 "O-016": [("W6C_premise.txt", 2, 18)],
 "O-017": [("W6C_premise.txt", 2, 18)],
 "O-018": [("W6C_o018.txt", 1, 52)],
 "O-019": [("W6C_collab.txt", 7, 40)],
 "O-020": [("W6C_premise.txt", 19, 52)],
 "O-021": [("W6C_premise.txt", 19, 52)],
 "O-023": [("W6C_collab.txt", 57, 67)],
 "O-024": [],
 "O-025": [("W6C_misc.txt", 21, 38)],
 "O-026": [("W6C_premise.txt", 53, 67)],
 "O-027": [("W6C_premise.txt", 53, 67)],
 "O-028": [("W6C_premise.txt", 68, 84)],
 "O-029": [("W6C_premise.txt", 68, 84)],
 "O-030": [("W6C_misc.txt", 1, 20)],
 "O-031": [("W6C_eleven.txt", 1, 20)],
 "O-032": [("W6C_premise.txt", 53, 67)],
 "O-033": [("W6C_misc.txt", 38, 71)],
 "O-034": [("W6C_premise.txt", 53, 67)],
 "O-035": [("W6C_collab.txt", 41, 56)],
 "O-036": [("W6C_collab2.txt", 1, 42), ("W6C_premise.txt", 28, 42)],
}

# extra pasted material that is not a script slice (git/PDF reads done in-session)
EXTRA = {
 "O-018": """### `git show aea9f53:docs/COLLABORATOR_BRIEFING.md` (the withdrawn version), lines 198-205
```
| comparison at L40 | margin |
|---|---:|
| 0.916 vs single-source 0.732 *(the abstract)* | **+0.184** |
| 0.916 vs pooled LODO 0.868 | +0.048 |
| **0.916 vs pooled, row-budget matched 0.893** | **+0.023** |

The headline survives, but **the margin goes from +0.184 to +0.023** - and that is against *your*
0.916. On our own grid the same contrast on the same six at L40 reads **0.811**, against pooled
row-matched at **0.893**, so like-for-like on our data the persona contrast **loses by 0.082**.
```
and lines 168-172 of the same version:
```
| on the same six datasets | **0.811** |
| on all eleven | **0.733** |
| cost of adding the other five | **-0.079** |
```
""",
 "O-024": """### `git show aea9f53:docs/COLLABORATOR_BRIEFING.md` line 247 (and 1a482b7 line 227)
```
**Token aggregation alone is worth up to +0.103.** Add the six-dataset subset effect (+0.079,
section 5a) and 0.811 -> 0.916 is fully accounted for without anyone being wrong.
```

### `git show f386a72:src/probes/deception-detection/deception_detection/scripts/pool_activations.py`
```
$ ... | grep -n "mean(0)"
84:        pooled.append(acts[row_mask].float().mean(0).squeeze(0).numpy())

$ ... | sed -n '1,4p'
\"\"\"Build per-example mean-pooled activations from the on-disk activation cache.

The persona method (arXiv 2605.09391) works on one vector per example -- the mean over the
assistant response tokens -- whereas the deployed probe works per token. This reads the cache
```
At branch head 45b16d9 the same statement has moved to line 123 and reads
`pooled.append(example.acts[example.mask].float().mean(0).squeeze(0).numpy())`.

### where +0.103 comes from
```
$ grep -rn "0\\.103" docs/thesis/persona-coordinates-70b-results.md
770:| 20 | 0.695 | 0.741 | 0.681 | 0.775 | 0.798 | 0.650 | **+0.103** |
1048:| 7 | Q8, onboarding | Zero-shot top-10% beats mean, "best +0.053" | +0.103 (L20), +0.064
     (L22), +0.053 (L28), <= +0.014 at 40-48; best-vs-best +0.042 | Reword |
```
""",
 "O-033": """### the ledger entry the claim restates (`kd/docs/FINDINGS.md`, F-94, lines 4172-4180)
```
**Clear counts over the full grid: content clears at 3 of 21 layers (L0, L12, L36); presence
clears at 15 of 21 - and at every single layer from L28 to L76**, thirteen consecutive.
...
| pooling | content effect | presence effect | ratio |
| mean (F-94 above) | +0.0356 | +0.0730 | ~2:1 |
| **last token** | **+0.0346** | **+0.0373** | **~1:1** |
...
**No ratio should be quoted at all** - it was wrong at 3 layers, corrected at 16 and 21,
and now moves again with the pooling choice.
```
""",
}


def slice_file(rel, a, b):
    path = os.path.join(OUT, rel)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    return "\n".join(lines[a - 1:b])


rows = list(csv.DictReader(io.open("audit/phase2/claims_batch_W6C-briefing.csv",
                                   encoding="utf-8")))
os.makedirs("audit/claims", exist_ok=True)
for r in rows:
    body = []
    body.append("# %s - batch W6C-briefing (docs/COLLABORATOR_BRIEFING.md)\n" % r["id"])
    body.append("**Source:** `%s`.  **Importance: %s**\n" % (r["source_ref"], r["importance"]))
    body.append("## The claim, restated\n%s\n" % r["claim_plain"])
    body.append("Stated numbers: %s\n" % r["stated_numbers"])
    body.append("## Evidence used\n- file(s): %s\n- key: %s\n- version: %s\n"
                % (r["evidence_file"], r["evidence_key"], r["version_used"]))
    body.append("## Level 1 (document vs file)\n%s\n\nMatch: **%s**\n"
                % (r["level1_value"], r["level1_match"]))
    body.append("## Level 2 (recomputed)\n%s\n\nMethod: %s\n"
                % (r["level2_value"], r["level2_method"]))
    if r["id"] in EXTRA:
        body.append("## Pasted evidence\n" + EXTRA[r["id"]])
    sl = SLICES.get(r["id"], [])
    if sl:
        body.append("## Script output (pasted)\n")
        for rel, a, b in sl:
            nice = rel.replace("../P12/", "audit/phase2/out/P12/")
            if not nice.startswith("audit/"):
                nice = "audit/phase2/out/W6C/" + nice
            body.append("`%s` lines %d-%d:\n\n```\n%s\n```\n" % (nice, a, b, slice_file(rel, a, b)))
    body.append("## Robustness\n%s\n" % r["robustness"])
    body.append("## Verdict\n**%s**\n\nConfidence: %s\n\nReason / notes: %s\n"
                % (r["verdict"], r["confidence"], r["notes"]))
    body.append("Script: `%s`\n" % r["script"])
    with io.open("audit/claims/%s.md" % r["id"], "w", encoding="utf-8") as f:
        f.write("\n".join(body))
    print("wrote audit/claims/%s.md" % r["id"])
