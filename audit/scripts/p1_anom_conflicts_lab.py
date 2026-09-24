"""Phase 1 Task B, list 4 (conflicting versions). RUNS ON THE LAB PC; reads ONLY the audit's census files in
C:/Users/user4/audit_scratch/census/ and PRINTS (writes nothing). Stdlib + audit module p1_anom_common only.

A "logical file" = one canonical lab-style path (p1_anom_common.canon: Drive crossgen/ -> run-crossgen/,
kd/crossgen11/ -> run-crossgen/kd11/, paper16/run/ and paper16/ -> run-paper16/, own100/run/ and own100/ -> run/, ...).
A conflict = the same logical file exists in >= 2 places (lab, one or more Drive paths, asix HEAD, his branch,
laptop-ignored) with DIFFERENT content. Content keys as in p1_anom_dupes_lab.py (md5_lf; Drive/laptop raw md5
translated via the lab's raw->lf map). Lab files > 200 MB are compared by byte size only.
Only result-type units (p1_anom_common.unit) are reported in detail; code/doc conflicts are counted.
Usage: python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_anom_conflicts_lab.py "ARGS=" --out <file>
"""
import collections, csv, gzip, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import canon, unit

csv.field_size_limit(10**9)
C = "C:/Users/user4/audit_scratch/census/"
P = collections.defaultdict(list)   # canon path -> [(src, key, where, time, bytes)]
raw2lf = {}
with gzip.open(C + "joined_labpc.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"] == "root":
            continue
        cp = canon("LAB", r["relpath"], r["root"])
        big = r["md5"] in ("SKIP_BIG", "ERR")
        if not big:
            raw2lf[r["md5"]] = r["md5_lf"]
        P[cp].append(("LAB", ("size:" + r["bytes"]) if big else r["md5_lf"], "lab", r["mtime"], int(r["bytes"])))
with gzip.open(C + "drive_census.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        cp = canon("DRIVE", r["path"])
        P[cp].append(("DRIVE", raw2lf.get(r["md5"], r["md5"]), "drive:" + r["path"], r["modtime"][:19] + "Z", int(r["bytes"] or 0)))
for src, fn in (("ASIX", "census_asix_git.tsv"), ("PCOOD", "census_pcood.tsv")):
    with open(C + fn, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            P[r["path"]].append((src, r["md5"], src.lower(), "", int(r["bytes"] or 0)))
with open(C + "census_laptop_ignored.tsv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        P[r["path"]].append(("LAPTOP_IGN", raw2lf.get(r["md5"], r["md5"]), "laptop", r["mtime"], int(r["bytes"] or 0)))


def same(a, b):
    """content equality, with the size fallback for unhashed big lab files."""
    ka, kb = a[1], b[1]
    if ka.startswith("size:") or kb.startswith("size:"):
        return a[4] == b[4]
    return ka == kb


conf = {}
for cp, ms in P.items():
    if len(ms) < 2:
        continue
    ref = ms[0]
    groups = []
    for m in ms:
        for g in groups:
            if same(g[0], m):
                g.append(m)
                break
        else:
            groups.append([m])
    if len(groups) > 1:
        conf[cp] = groups

print("logical files present in >=2 places:", sum(1 for v in P.values() if len(v) > 1))
print("logical files with >=2 DIFFERENT contents:", len(conf))
by_unit = collections.defaultdict(list)
nores = collections.Counter()
for cp, gs in conf.items():
    u, k = unit(cp)
    if u is None:
        nores[k] += 1
        top = cp.split("/")[0]
        by_unit["(non-result) " + ("/".join(cp.split("/")[:2]) if top in ("src", "kd", "cap", "docs", "scale") else top)].append((cp, gs))
        continue
    by_unit[u].append((cp, gs))
print("non-result (code/doc/input) conflicts by reason:", dict(nores))

pat = collections.Counter()
for u, items in sorted(by_unit.items()):
    for cp, gs in items:
        sig = " vs ".join(sorted("+".join(sorted({m[0] for m in g})) for g in gs))
        pat[sig] += 1
print("\nsource pattern of the differing versions (each side = sources sharing one version):")
for s, c in pat.most_common():
    print(f"   {c:6d}  {s}")

print("\nper unit (count; then up to 6 examples with every version: sources/where/time/bytes/key8):")
for u, items in sorted(by_unit.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    sigs = collections.Counter(" vs ".join(sorted("+".join(sorted({m[0] for m in g})) for g in gs)) for _, gs in items)
    print(f"\n## {u}  ({len(items)} logical files)  patterns: {dict(sigs)}")
    for cp, gs in sorted(items)[:6]:
        print("   ", cp)
        for g in gs:
            print("        " + " | ".join(f"{m[0]}:{m[2] if m[0] == 'DRIVE' else ''} {m[3]} {m[4]}B {m[1][:8]}" for m in g))
