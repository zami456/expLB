"""Phase 1 Task B, list 3 (duplicates). RUNS ON THE LAB PC (the laptop lacks RAM); reads ONLY the audit's own census
files in C:/Users/user4/audit_scratch/census/ and PRINTS its results (writes nothing). Imports only the stdlib and the
audit module p1_anom_common (canon/unit path helpers; no owner code).

Content key: lab files -> md5_lf (md5 after CRLF->LF, = what git stores); Drive and laptop files -> their raw md5,
translated to the lab md5_lf when the raw md5 equals a lab file's raw md5; git blobs (asix, his branch) -> blob md5
(already LF). Lab files > 200 MB were not hashed (SKIP_BIG); they are matched to Drive by canonical path + byte size
instead (reported separately, INFERRED identity).
Usage (from the laptop):
  python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_anom_dupes_lab.py "ARGS=" --out <local file>
"""
import collections, csv, gzip, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import canon, unit

csv.field_size_limit(10**9)
C = "C:/Users/user4/audit_scratch/census/"
EMPTY = "d41d8cd98f00b204e9800998ecf8427e"
M = collections.defaultdict(list)          # key -> [(src, canon_path, bytes)]
raw2lf = {}
big = []
n_src = collections.Counter()
with gzip.open(C + "joined_labpc.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"] == "root":
            continue
        cp = canon("LAB", r["relpath"], r["root"])
        n_src["LAB"] += 1
        if r["md5"] in ("SKIP_BIG", "ERR"):
            big.append((cp, int(r["bytes"]), r["md5"]))
            continue
        raw2lf[r["md5"]] = r["md5_lf"]
        M[r["md5_lf"]].append(("LAB", cp, int(r["bytes"])))
drive_by_cp = {}
with gzip.open(C + "drive_census.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        cp = canon("DRIVE", r["path"])
        n_src["DRIVE"] += 1
        k = raw2lf.get(r["md5"], r["md5"])
        M[k].append(("DRIVE", "drive:" + r["path"] if cp.startswith("drive:") else cp + "  [drive:" + r["path"] + "]", int(r["bytes"] or 0)))
        drive_by_cp.setdefault(cp, []).append((int(r["bytes"] or 0), r["path"]))
for src, fn in (("ASIX", "census_asix_git.tsv"), ("PCOOD", "census_pcood.tsv")):
    with open(C + fn, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n_src[src] += 1
            M[r["md5"]].append((src, r["path"], int(r["bytes"] or 0)))
if os.path.exists(C + "census_laptop_ignored.tsv"):
    with open(C + "census_laptop_ignored.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n_src["LAPTOP_IGN"] += 1
            M[raw2lf.get(r["md5"], r["md5"])].append(("LAPTOP_IGN", r["path"], int(r["bytes"] or 0)))
print("files per source:", dict(n_src))


def cpath(m):
    return m[1].split("  [drive:")[0]


def unit_of(m):
    p = cpath(m)
    if m[0] == "DRIVE" and p.startswith("drive:"):
        p = p
    u, k = unit(p)
    return u


groups = {k: v for k, v in M.items() if len(v) > 1}
print("content keys:", len(M), " duplicate groups (>=2 copies anywhere):", len(groups),
      " files in them:", sum(len(v) for v in groups.values()))
empty = groups.pop(EMPTY, [])
print("empty-file group:", len(empty), "files (excluded below)")
tiny = {k: v for k, v in groups.items() if max(b for _, _, b in v) < 100}
for k in tiny:
    groups.pop(k)
print("tiny-content groups (<100 B):", len(tiny), "groups,", sum(len(v) for v in tiny.values()), "files (excluded below)")

single, multi = {}, {}
for k, v in groups.items():
    paths = {cpath(m) for m in v}
    (single if len(paths) == 1 else multi)[k] = v
print("\n== A. groups whose copies all share ONE canonical path (backups / git copies of the same file):", len(single))
pat = collections.Counter()
for v in single.values():
    pat["+".join(f"{s}{n}" for s, n in sorted(collections.Counter(s for s, _, _ in v).items()))] += 1
for p, c in pat.most_common(30):
    print(f"   {c:7d}  {p}")

print("\n== B. groups with >=2 DIFFERENT canonical paths:", len(multi))
cls = collections.Counter()
cls_files = collections.Counter()
cross = collections.defaultdict(lambda: [0, 0, None, 0])
same_unit = collections.defaultdict(lambda: [0, 0, None])
pc_pairs = collections.defaultdict(lambda: [0, None])
nores = 0
for k, v in multi.items():
    units = {unit_of(m) for m in v}
    units_res = {u for u in units if u}
    if any(s == "PCOOD" for s, _, _ in v):
        others = sorted({(unit_of(m) or cpath(m).rsplit("/", 1)[0]) for m in v if m[0] != "PCOOD"})
        mine = sorted({cpath(m).rsplit("/", 1)[0] for m in v if m[0] == "PCOOD"})
        key = (" | ".join(mine), " | ".join(others) or "(PCOOD only)")
        pc_pairs[key][0] += 1
        pc_pairs[key][1] = pc_pairs[key][1] or [cpath(m) + f" ({m[0]})" for m in v][:4]
    if not units_res:
        nores += 1
        continue
    paths = sorted({cpath(m) for m in v})
    if all(p.startswith("run-crossgen/") and p.endswith(".csv") for p in paths):
        c = "crossgen per-cell CSV"
    elif len(units_res) == 1 and None not in units:
        c = "same unit, different file names"
        u = next(iter(units_res))
        same_unit[u][0] += 1
        same_unit[u][1] += len(paths)
        same_unit[u][2] = same_unit[u][2] or paths[:3]
    else:
        c = "different units"
        key = " <> ".join(sorted(str(u) for u in units))
        cross[key][0] += 1
        cross[key][1] += len(paths)
        cross[key][2] = cross[key][2] or paths[:4]
        cross[key][3] = max(cross[key][3], max(b for _, _, b in v))
    cls[c] += 1
    cls_files[c] += len(v)
print("   groups with no result-unit member (code/doc/input copies):", nores)
for c, n in cls.most_common():
    print(f"   {n:7d} groups / {cls_files[c]:7d} files  {c}")

print("\n== B1. same unit, different file names (top 40 units):")
for u, (n, npth, ex) in sorted(same_unit.items(), key=lambda kv: -kv[1][0])[:40]:
    print(f"   {n:6d} groups {npth:6d} paths  {u}   e.g. {ex}")

print("\n== B2. DIFFERENT units share identical content (all, sorted by groups):")
for key, (n, npth, ex, mx) in sorted(cross.items(), key=lambda kv: -kv[1][0]):
    print(f"   {n:6d} groups {npth:6d} paths max {mx:>11d} B  {key}\n          e.g. {ex}")

print("\n== C. groups involving his branch (PCOOD):")
for (mine, others), (n, ex) in sorted(pc_pairs.items(), key=lambda kv: -kv[1][0]):
    print(f"   {n:5d}  his {mine}  ==  {others}\n          e.g. {ex}")

print("\n== D. lab files > 200 MB (not hashed) vs Drive by canonical path + size:")
st = collections.Counter()
for cp, b, why in big:
    d = drive_by_cp.get(cp, [])
    if any(db == b for db, _ in d):
        st["same path+size on Drive (INFERRED identical)"] += 1
    elif d:
        st["same path on Drive, DIFFERENT size"] += 1
        print("   size differs:", cp, b, d[:2])
    else:
        st["no Drive file at that path"] += 1
        print("   not on Drive:", cp, b, why)
print("  ", dict(st), " total", len(big))
