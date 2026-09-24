"""P17 merge: fold the Wave-1 outputs (audit/p17/) into the audit's shared tables.

Idempotent: rows are replaced or appended by id, and a P17 note is added only once.
The *.before-p17 backups were taken in Step 0 (see audit/BRANCH_PINS.md). This script refuses to
run without them.

  claims.csv          176 D- rows re-pinned (note + version_used suffix; D-189 verdict revised with
                      the old verdict kept), 12 other D- rows annotated, +26 D2- rows, +17 L2- rows
  claims_inventory    +26 D2- rows (derived from the claims) + 17 L2- rows (Agent C)
  fmap_all.csv        +18 rows F-110..F-127 (Agent C)
  experiments.csv     +5 rows for work newer than the 09-22 census (evidence in each row)

Usage: python audit/scripts/p17_merge.py [--dry-run]
"""
import csv, pathlib, sys

A = pathlib.Path(__file__).resolve().parents[1]
P = A / "p17"
DRY = "--dry-run" in sys.argv
TAG = "[P17 2026-09-24]"


def read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def write(p, fields, rows):
    bak = pathlib.Path(str(p) + ".before-p17")
    assert bak.exists(), f"missing backup {bak}"
    if DRY:
        return
    with open(p, "w", encoding="utf-8", newline="") as fh:   # csv default lineterminator = CRLF, as the originals
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def upsert(rows, new, key="id"):
    idx = {r[key]: i for i, r in enumerate(rows)}
    added = replaced = 0
    for n in new:
        if n[key] in idx:
            rows[idx[n[key]]] = n; replaced += 1
        else:
            rows.append(n); added += 1
    return added, replaced


# ---------------- claims.csv ----------------
cf, claims = read(A / "claims.csv")
byid = {r["id"]: r for r in claims}
rep = {r["id"]: r for r in read(P / "d_rows_repinned.csv")[1]}
n_note = n_verdict = 0
for i, x in rep.items():
    r = byid[i]
    if TAG in r["notes"]:
        continue
    src = x["source_sha_of_stored"]
    r["version_used"] += f" {TAG} re-pinned: stored value read at {src}; verdict at origin/paper@16def2de = {x['verdict_at_pin']}"
    r["notes"] = (r["notes"] + f" {TAG} P17 re-pin (audit/p17/d_rows_repinned.csv): "
                  f"@ea1ce52b={x['value_at_ea1ce52b'][:160]} | @16def2de={x['value_at_16def2de'][:160]}").strip()
    n_note += 1
    if x["changed"] == "y" and x["verdict_at_pin"] != r["verdict"]:
        r["verdict"] = f"{x['verdict_at_pin']} {TAG} (was: {r['verdict']}; changed at origin/paper@2ad758fb, evidence: {x['evidence'][:200]})"
        n_verdict += 1
oth = read(P / "d_rows_other_affected.csv")[1]
n_oth = 0
for x in oth:
    r = byid.get(x["id"])
    if r is None or TAG in r["notes"]:
        continue
    r["notes"] = (r["notes"] + f" {TAG} values checked 45b16d90->16def2de (audit/p17/d_rows_other_affected.csv): "
                  f"changed={x['changed']}, affected={x['affected']}; {x['evidence'][:220]}").strip()
    n_oth += 1
_, d2 = read(P / "paper_new_claims.csv")
_, l2 = read(P / "ledger_new_claims.csv")
a1, r1 = upsert(claims, d2 + l2)
write(A / "claims.csv", cf, claims)
print(f"claims.csv: re-pinned {n_note} rows, verdicts revised {n_verdict}, other D- annotated {n_oth}, "
      f"new rows added {a1} replaced {r1}, total {len(claims)}")

# ---------------- claims_inventory.csv ----------------
inf, inv = read(A / "claims_inventory.csv")
_, l2inv = read(P / "ledger_new_inventory.csv")
d2inv = [{"id": r["id"], "source": "paper tex (origin/paper@16def2de)", "source_ref": r["source_ref"],
          "claim_short": r["claim_plain"][:200], "numbers": r["stated_numbers"], "layer": "",
          "dataset_hint": "", "status": "CANDIDATE"} for r in d2]
a2, r2 = upsert(inv, d2inv + l2inv)
write(A / "claims_inventory.csv", inf, inv)
print(f"claims_inventory.csv: added {a2} replaced {r2}, total {len(inv)}")

# ---------------- fmap_all.csv ----------------
ff, fmap = read(A / "phase1" / "fmap_all.csv")
_, fnew = read(P / "fmap_F110_F127.csv")
a3, r3 = upsert(fmap, fnew, key="f_id")
write(A / "phase1" / "fmap_all.csv", ff, fmap)
print(f"fmap_all.csv: added {a3} replaced {r3}, total {len(fmap)}")

# ---------------- experiments.csv ----------------
ef, exps = read(A / "experiments.csv")
DD = "src/probes/deception-detection"
new_e = [
    dict(exp_id="C21-depth-sweep-16L",
         what_it_tests="Persona study (persona_ood.py) at every 4th layer 20..79 (16 layers): ranking, zero-shot, base-probe inversion vs depth.",
         who="collaborator", when_first="2026-09-23 23:46", when_last="2026-09-24 16:47",
         when_source="git commit dates on origin/paper (first ea1ce52b, results unchanged after 2ad758fb per audit/p17/csv_drift.csv)",
         branch="origin/paper", commit="ea1ce52b 2026-09-23 23:46 / last result change 2ad758fb 2026-09-24 15:34 (pin 16def2de)",
         script="DD/deception_detection/scripts/persona_ood.py; layer_sweep.py; persona_ood_figures.py",
         config_or_args="--layer {20..76 step 4, 79} (UNKNOWN beyond folder names; run args not committed)",
         scheduled_task="(his machine)",
         results_location=f"origin/paper@16def2de {DD}/results/persona_ood_ext_L{{24..79}}/ (611 files) + results/persona_depth/ (3 files, depth.csv)",
         n_result_files="614", status="complete",
         status_evidence="committed CSVs; depth.csv agrees with per-layer files cell for cell (audit/p17/A_REPORT.md)",
         depends_on="his activation cache per layer (not in git)", f_ids="F-117 F-120 F-122",
         notes=f"{TAG} added by P17; paper claims on it: claims.csv D2-001..D2-026."),
    dict(exp_id="C22-unified-axis-arms",
         what_it_tests="Unified axis (13v12) / uni PC arms in the transfer matrices and zero-shot tables, all 16 layers.",
         who="collaborator", when_first="2026-09-23 23:46", when_last="2026-09-24 15:34",
         when_source="git commit dates: 52 matrix_*uni_* files (L24..L79) at ea1ce52b, 12 (L20/L40/L44) added at 2ad758fb (QA F-note, audit/p17/QA_REPORT.md)", branch="origin/paper", commit="ea1ce52b 2026-09-23 23:46 / 2ad758fb 2026-09-24 15:34",
         script="DD/deception_detection/scripts/persona_ood.py", config_or_args="UNKNOWN (not committed)",
         scheduled_task="(his machine)",
         results_location=f"origin/paper@16def2de {DD}/results/persona_ood_ext_L*/matrix_*uni_*.csv (64 files) + zero_shot.csv cols auroc_uni_axis/auroc_uni_pc1",
         n_result_files="64", status="complete", status_evidence="committed CSVs (audit/p17/csv_drift.csv)",
         depends_on="C21-depth-sweep-16L", f_ids="", notes=f"{TAG} added by P17; provenance corrected after QA (was: first at 2ad758fb)."),
    dict(exp_id="C23-gender-secret-ndif",
         what_it_tests="Gender-Secret via NDIF: does the knowledge-denial direction read the lie when both classes deny.",
         who="collaborator", when_first="2026-09-22 15:40", when_last="2026-09-22 15:40",
         when_source="git commit date 572095cb", branch="origin/paper", commit="572095cb 2026-09-22 15:40",
         script="DD/deception_detection/scripts/gender_secret.py", config_or_args="UNKNOWN (not committed)",
         scheduled_task="(his machine)", results_location=f"origin/paper@16def2de {DD}/results/gender_secret/ (2 files)",
         n_result_files="2", status="complete", status_evidence="committed files",
         depends_on="NDIF; gender-secret model", f_ids="F-118", notes=f"{TAG} added by P17; reported in COLLAB_PAPER_BRANCH.md."),
    dict(exp_id="C24-paper-build",
         what_it_tests="Paper 'Lying in Knowledge-Denier Persona Coordinates': every number generated from committed results by make_numbers.py.",
         who="collaborator", when_first="2026-09-22 13:58", when_last="2026-09-24 16:47",
         when_source="git commit dates of paper/ on origin/paper", branch="origin/paper",
         commit="8fd071b2 2026-09-22 13:58 / last 16def2de 2026-09-24 16:47",
         script="DD/paper/make_numbers.py; DD/paper/make_figures.py; tectonic main.tex",
         config_or_args="per paper/README.md", scheduled_task="(his machine)",
         results_location=f"origin/paper@16def2de {DD}/paper/ (34 files; generated/numbers.tex 1,176 macros)",
         n_result_files="34", status="complete (evolving)", status_evidence="committed; numbers re-derived in audit/p17/paper_delta_values_16def2de.csv",
         depends_on="C21 C22 and earlier persona_ood results", f_ids="F-119 F-127",
         notes=f"{TAG} added by P17; residual overstatement introduction.tex:68-70 (claims.csv D2 DOES NOT HOLD)."),
    dict(exp_id="O88-kd-scale-ladder",
         what_it_tests="kd bank run at 3B and 8B (first time), compared with the 26-persona bank: scale ladder of the persona gain.",
         who="owner", when_first="2026-09-23 23:57", when_last="2026-09-24 08:51",
         when_source="kd/docs/FINDINGS.md F-121 (ledger prose; lab logs not re-read)",
         branch="lab-only (not in git)", commit="n/a (separate tree C:/Users/user4/expLB-resplit)",
         script="C:/Users/user4/expLB-resplit/run_kd_scale.ps1, run_kdscale_crossgen.ps1; audit/scripts/p12_kd_scale_ladder.py",
         config_or_args="UNKNOWN (see .ps1 on lab)", scheduled_task="UNKNOWN",
         results_location="C:/Users/user4/expLB-resplit/run-crossgen/kdscale_*/matrices/mean_L*_s42.json",
         n_result_files="UNKNOWN", status="complete",
         status_evidence="17/17 ledger numbers re-derived from lab files (audit/p17/fmap_F110_F127.csv F-121)",
         depends_on="O-scale rungs (run-crossgen/scale_*)", f_ids="F-121",
         notes=f"{TAG} added by P17; LAB-ONLY evidence."),
]
a4, r4 = upsert(exps, new_e, key="exp_id")
write(A / "experiments.csv", ef, exps)
print(f"experiments.csv: added {a4} replaced {r4}, total {len(exps)}")
print("DRY RUN - nothing written" if DRY else "written")
