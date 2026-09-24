"""F-104 - "four script fixes existed only in the repo and never reached the lab PC;
a SHA-256 audit of 160 local vs 151 remote scripts found all of them; all four deployed and hash-matched".

Re-derived, entirely from the audit's own Phase-1 census (lab, 2026-09-22 20:56, ~18 h after F-104)
and from git / the working tree - no new lab call:
  1. count of .py/.ps1 under cap/scripts, kd/scripts, scale/scripts, src/assistant_axis_ndif
     on the lab and locally  (ledger: 151 remote / 160 local, on 2026-09-21 ~20:30);
  2. per-file md5 (line-ending normalised) lab vs git HEAD for every one of them, and in
     particular for the four F-104 names;
  3. the set difference: local-only and remote-only files (ledger: 17 and 8).
"""
import csv, hashlib, os, subprocess, sys

DIRS = ["cap/scripts", "kd/scripts", "scale/scripts", "src/assistant_axis_ndif"]
EXT = (".py", ".ps1")

def md5_lf(b):
    return hashlib.md5(b.replace(b"\r\n", b"\n")).hexdigest()

# --- lab side, from the Phase 1 census (eol-normalised md5) -------------------------------
lab = {}
with open("audit/phase1/labpc/labpc_census_eol.tsv", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        if row["root"] != r"C:\Users\user4\expLB":
            continue
        rel = row["relpath"].replace("\\", "/")
        d = os.path.dirname(rel)
        if d in DIRS and rel.endswith(EXT):
            lab[rel] = row["md5_lf"]

# --- local side: git HEAD blobs + untracked files in those dirs ---------------------------
loc = {}
for d in DIRS:
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        rel = f"{d}/{fn}"
        if os.path.isfile(rel) and rel.endswith(EXT):
            loc[rel] = md5_lf(open(rel, "rb").read())

print(f"script dirs: {DIRS}")
print(f"LAB  (.py/.ps1, census 2026-09-22 20:56): {len(lab)}")
print(f"LOCAL(.py/.ps1, working tree now)      : {len(loc)}")

only_loc = sorted(set(loc) - set(lab))
only_lab = sorted(set(lab) - set(loc))
both = sorted(set(loc) & set(lab))
diff = [p for p in both if loc[p] != lab[p]]
print(f"\ncommon={len(both)}  local-only={len(only_loc)}  remote-only={len(only_lab)}  DIVERGENT(md5_lf)={len(diff)}")

print("\n--- LOCAL-ONLY (ledger: 17) ---")
for p in only_loc: print("   ", p)
print("\n--- REMOTE-ONLY (ledger: 8) ---")
for p in only_lab: print("   ", p)
print("\n--- DIVERGENT content (ledger after deployment: 0) ---")
for p in diff: print("   ", p, "local", loc[p], "lab", lab[p])

print("\n--- the four files F-104 names ---")
FOUR = ["cap/scripts/run_cap_controls.ps1", "src/assistant_axis_ndif/15_persona_count.py",
        "src/assistant_axis_ndif/19_figures.py", "cap/scripts/run_sarc_pilot.ps1"]
sizes = {}
with open("audit/phase1/labpc/labpc_census.tsv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["root"] != r"C:\Users\user4\expLB": continue
        p = row["relpath"].replace("\\", "/")
        if p in FOUR:
            sizes[p] = (row["bytes"], row["mtime"], row["md5"])
for p in FOUR:
    lb, lc = lab.get(p), loc.get(p)
    st = os.stat(p) if os.path.exists(p) else None
    print(f"  {p}")
    print(f"     lab  md5_lf={lb}  bytes/mtime/md5={sizes.get(p)}")
    print(f"     local md5_lf={lc}  bytes={st.st_size if st else None}")
    print(f"     MATCH={lb == lc}")
