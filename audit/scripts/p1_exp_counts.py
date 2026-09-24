"""Phase 1 Task A: summary counts over audit/experiments.csv + audit/phase1/exp_work/experiments_computed.tsv.
Prints: rows by status x who; orphan candidates (no F-id and 0 doc mentions); experiments with lab files
rewritten in the 09-22 wave; experiments whose lab files are (partly) missing from Drive.
Usage (repo root): PYTHONIOENCODING=utf-8 python audit/scripts/p1_exp_counts.py
"""
import collections, csv, os

AUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = list(csv.DictReader(open(os.path.join(AUD, "experiments.csv"), encoding="utf-8")))
comp = {r["exp_id"]: r for r in csv.DictReader(open(os.path.join(AUD, "phase1", "exp_work", "experiments_computed.tsv"), encoding="utf-8"), delimiter="\t")}


def who_group(w):
    w = w.lower()
    if w.startswith("collaborator"):
        return "collaborator"
    if w.startswith("owner"):
        return "owner"
    if w.startswith("upstream"):
        return "upstream"
    return "unknown"


def kind(eid):
    return "print-only analysis" if eid.startswith("AN") else ("background" if eid.startswith("BG") else "run")


print("TOTAL rows", len(rows))
ct = collections.Counter((who_group(r["who"]), r["status"]) for r in rows)
whos = sorted({k[0] for k in ct}); stats = sorted({k[1] for k in ct})
print("who \\ status: " + " | ".join(stats) + " | total")
for w in whos:
    print(f"{w}: " + " | ".join(str(ct[(w, s)]) for s in stats) + f" | {sum(ct[(w, s)] for s in stats)}")
print("by kind:", dict(collections.Counter(kind(r["exp_id"]) for r in rows)))
print("rows with >=1 F-id:", sum(1 for r in rows if r["f_ids"].strip()))
allf = set()
for r in rows:
    allf.update(r["f_ids"].split())
print("distinct F-ids referenced:", len(allf), "missing from F-01..F-109:", sorted(set(f"F-{i:02d}" if i < 100 else f"F-{i}" for i in range(1, 110)) - allf))

orph = [r for r in rows if not r["f_ids"].strip() and comp[r["exp_id"]]["doc_hits"] in ("0", "")]
docsonly = [r for r in rows if not r["f_ids"].strip() and comp[r["exp_id"]]["doc_hits"] not in ("0", "")]
print("\nn orphan candidates:", len(orph), " n docs-only (no F-id):", len(docsonly))
print("premise rows (O78-O85) lab files:", sum(int(comp[r["exp_id"]]["lab_n"] or 0) for r in rows if r["exp_id"][:3] in ("O78", "O79", "O80", "O81", "O82", "O83", "O84", "O85")))
print("\nORPHAN CANDIDATES (no F-id, 0 doc mentions by the precise keys):")
for r in rows:
    c = comp[r["exp_id"]]
    if not r["f_ids"].strip() and c["doc_hits"] in ("0", ""):
        print(f"  {r['exp_id']}  n={r['n_result_files']}  status={r['status']}  loc={r['results_location'][:110]}")
print("\nNO F-id BUT mentioned in docs (docs only):")
for r in rows:
    c = comp[r["exp_id"]]
    if not r["f_ids"].strip() and c["doc_hits"] not in ("0", ""):
        print(f"  {r['exp_id']}  doc_hits={c['doc_hits']}  ({c['doc_hit_names'][:150]})")
print("\nLAB FILES REWRITTEN IN THE 09-22 WAVE (lab_wave_n > 0):")
for r in rows:
    c = comp[r["exp_id"]]
    if c["lab_wave_n"] not in ("", "0"):
        print(f"  {r['exp_id']}  wave={c['lab_wave_n']} of lab={c['lab_n']}")
print("\nLAB FILES WITHOUT AN IDENTICAL DRIVE COPY (lab_on_drive < lab_n):")
for r in rows:
    c = comp[r["exp_id"]]
    if c["lab_n"] not in ("", "0") and int(c["lab_on_drive_n"] or 0) < int(c["lab_n"]):
        print(f"  {r['exp_id']}  on_drive={c['lab_on_drive_n']} of lab={c['lab_n']}")
print("\nSTATUS != complete:")
for r in rows:
    if r["status"] != "complete":
        print(f"  {r['exp_id']}  {r['status']}: {r['status_evidence'][:160]}")
