"""Phase 1 Task B helper: folder-level aggregation of every census source (streamed; small memory).
Read-only. Inputs: audit/phase1/labpc/joined_labpc.tsv.gz, drive_census.tsv.gz, audit/phase1/census_asix_git.tsv,
census_pcood.tsv, census_laptop_ignored.tsv.
Output: audit/phase1/anomalies_detail/folders.tsv  one row per (source, folder prefix at depth 1..5):
  source folder depth n_files n_result bytes mtime_min mtime_max top_ext [lab only: on_drive in_asix in_pcood asix_diff drive_diff skip_big]
Folder naming: LAB paths under C:/Users/user4/expLB are repo-relative; other lab roots are prefixed '@<root>/'.
DRIVE paths are relative to tamara:expLB/. ASIX/PCOOD/LAPTOP paths are repo-relative.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_folders.py
"""
import collections, csv, gzip, os, sys

csv.field_size_limit(10**9)
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P1 = os.path.join(A, "phase1")
OUT = os.path.join(P1, "anomalies_detail")
os.makedirs(OUT, exist_ok=True)
RES = (".json", ".jsonl", ".csv", ".pt", ".npz", ".npy", ".log", ".status", ".parquet", ".pkl", ".txt", ".yaml", ".yml")
MAXD = 5
BS = chr(92)

agg = {}


def add(source, path, nbytes, mtime, extra=None):
    parts = path.split("/")
    dirs = parts[:-1]
    ext = os.path.splitext(parts[-1])[1].lower()
    for d in range(1, min(len(dirs), MAXD) + 1):
        key = (source, "/".join(dirs[:d]))
        a = agg.get(key)
        if a is None:
            a = agg[key] = {"n": 0, "nres": 0, "bytes": 0, "mmin": "", "mmax": "", "ext": collections.Counter(),
                            "on_drive": 0, "in_asix": 0, "in_pcood": 0, "asix_diff": 0, "drive_diff": 0, "skip_big": 0}
        a["n"] += 1
        a["nres"] += ext in RES
        a["bytes"] += nbytes
        if mtime:
            if not a["mmin"] or mtime < a["mmin"]:
                a["mmin"] = mtime
            if mtime > a["mmax"]:
                a["mmax"] = mtime
        if len(a["ext"]) < 40 or ext in a["ext"]:
            a["ext"][ext or "(none)"] += 1
        if extra:
            for k, v in extra.items():
                a[k] += v


def lab_path(root, rel):
    root = root.replace(BS, "/")
    if root.endswith("/expLB"):
        return rel
    return "@" + root.split("/")[-1] + "/" + rel


with gzip.open(os.path.join(P1, "labpc", "joined_labpc.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"] == "root":
            continue
        add("LAB", lab_path(r["root"], r["relpath"]), int(r["bytes"] or 0), r["mtime"],
            {"on_drive": int(r["on_drive"]), "in_asix": int(r["in_asix"]), "in_pcood": int(r["in_pcood"]),
             "asix_diff": int(bool(r["asix_same_name_diff_content"])), "drive_diff": int(bool(r["drive_same_name_diff_content"])),
             "skip_big": int(r["md5"] == "SKIP_BIG")})
with gzip.open(os.path.join(P1, "labpc", "drive_census.tsv.gz"), "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        add("DRIVE", r["path"], int(r["bytes"] or 0), r["modtime"][:19].replace("T", " "))
for src, fn in (("ASIX", "census_asix_git.tsv"), ("PCOOD", "census_pcood.tsv")):
    with open(os.path.join(P1, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            add(src, r["path"], int(r["bytes"] or 0), "")
with open(os.path.join(P1, "census_laptop_ignored.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        add("LAPTOP_IGN", r["path"], int(r["bytes"] or 0), r["mtime"])

with open(os.path.join(OUT, "folders.tsv"), "w", encoding="utf-8", newline="\n") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["source", "folder", "depth", "n_files", "n_result", "bytes", "mtime_min", "mtime_max", "top_ext",
                "on_drive", "in_asix", "in_pcood", "asix_diff", "drive_diff", "skip_big"])
    for (s, fo), a in sorted(agg.items()):
        w.writerow([s, fo, fo.count("/") + 1, a["n"], a["nres"], a["bytes"], a["mmin"], a["mmax"],
                    ",".join(f"{e}:{c}" for e, c in a["ext"].most_common(4)),
                    a["on_drive"], a["in_asix"], a["in_pcood"], a["asix_diff"], a["drive_diff"], a["skip_big"]])
print("folders:", len(agg), "->", os.path.join(OUT, "folders.tsv"))
