"""Phase 1 census join. Runs ON the lab PC (the laptop lacks RAM). Reads only audit_scratch files.
Inputs (in audit_scratch/census/): labpc_census.tsv, labpc_census_eol.tsv, drive_census.tsv.gz,
  census_asix_git.tsv (asix HEAD blobs), census_pcood.tsv (his branch blobs under src/probes/deception-detection).
Matching is by content: raw md5, or md5 after CRLF->LF for text files (git stores LF blobs).
Outputs (gzipped TSV, same folder):
  joined_labpc.tsv.gz  one row per lab file: where identical copies exist (drive / asix / pcood), and whether a
                       same-named file exists elsewhere with DIFFERENT content (a version conflict)
  drive_not_on_lab.tsv.gz  Drive files whose content matches no lab file (older versions, ghosts, Drive-only)
  asix_results_not_on_lab.tsv.gz  asix result-type blobs whose content matches no lab file
  join_summary.txt     counts per top-level folder
"""
import collections, csv, gzip, os
C = "C:/Users/user4/audit_scratch/census/"
csv.field_size_limit(10**9)
RESULT_EXT = (".json", ".jsonl", ".csv", ".pt", ".npz", ".npy", ".log", ".yaml", ".yml", ".txt", ".status", ".parquet", ".md")

def rd(path, gz=False):
    f = gzip.open(path, "rt", encoding="utf-8") if gz else open(path, encoding="utf-8")
    return csv.DictReader(f, delimiter="\t")

eol = {(r["root"], r["relpath"]): r["md5_lf"] for r in rd(C + "labpc_census_eol.tsv")}
lab = []
for r in rd(C + "labpc_census.tsv"):
    r["md5_lf"] = eol.get((r["root"], r["relpath"]), r["md5"])
    r["top"] = r["root"].split("\\")[-1] + "/" + (r["relpath"].split("/")[0] if "/" in r["relpath"] else "<loose>")
    lab.append(r)
lab_by_md5 = collections.defaultdict(list)
for r in lab:
    lab_by_md5[r["md5"]].append(r)
    if r["md5_lf"] != r["md5"]:
        lab_by_md5[r["md5_lf"]].append(r)

drive = list(rd(C + "drive_census.tsv.gz", gz=True))
drive_by_md5 = collections.defaultdict(list)
drive_by_name = collections.defaultdict(list)
for d in drive:
    drive_by_md5[d["md5"]].append(d["path"])
    drive_by_name[os.path.basename(d["path"])].append(d)

def load_git(fn, label):
    by_md5, by_name = collections.defaultdict(list), collections.defaultdict(list)
    rows = list(rd(C + fn))
    for g in rows:
        by_md5[g["md5"]].append(g["path"])
        by_name[os.path.basename(g["path"])].append(g)
    return rows, by_md5, by_name
asix, asix_md5, asix_name = load_git("census_asix_git.tsv", "asix")
pc, pc_md5, pc_name = load_git("census_pcood.tsv", "pcood")

summ = collections.defaultdict(collections.Counter)
with gzip.open(C + "joined_labpc.tsv.gz", "wt", encoding="utf-8", newline="\n") as out:
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(["root", "relpath", "bytes", "mtime", "md5", "md5_lf", "on_drive", "drive_paths", "in_asix", "asix_paths",
                "in_pcood", "pcood_paths", "asix_same_name_diff_content", "drive_same_name_diff_content"])
    for r in lab:
        keys = {r["md5"], r["md5_lf"]} - {"SKIP_BIG", "ERR"}
        dp = sorted({p for k in keys for p in drive_by_md5.get(k, [])})
        ap = sorted({p for k in keys for p in asix_md5.get(k, [])})
        pp = sorted({p for k in keys for p in pc_md5.get(k, [])})
        name = os.path.basename(r["relpath"])
        # same relative path in asix but different content = version conflict (only when a git path ends with relpath)
        a_conf = [g["path"] for g in asix_name.get(name, []) if g["path"].endswith(r["relpath"]) and g["md5"] not in keys]
        d_conf = [d["path"] for d in drive_by_name.get(name, []) if d["md5"] not in keys and d["path"].split("/", 1)[-1] == r["relpath"].split("/", 1)[-1]]
        w.writerow([r["root"], r["relpath"], r["bytes"], r["mtime"], r["md5"], r["md5_lf"], int(bool(dp)), "|".join(dp[:3]),
                    int(bool(ap)), "|".join(ap[:3]), int(bool(pp)), "|".join(pp[:3]), "|".join(a_conf[:3]), "|".join(d_conf[:3])])
        s = summ[r["top"]]
        s["files"] += 1; s["bytes"] += int(r["bytes"])
        s["on_drive"] += bool(dp); s["in_asix"] += bool(ap); s["in_pcood"] += bool(pp)
        s["asix_conflict"] += bool(a_conf); s["drive_conflict"] += bool(d_conf)
        s["skip_big"] += r["md5"] == "SKIP_BIG"

lab_keys = set(lab_by_md5)
with gzip.open(C + "drive_not_on_lab.tsv.gz", "wt", encoding="utf-8", newline="\n") as out:
    out.write("path\tbytes\tmodtime\tmd5\n"); nd = collections.Counter()
    for d in drive:
        if d["md5"] and d["md5"] not in lab_keys:
            out.write(f"{d['path']}\t{d['bytes']}\t{d['modtime']}\t{d['md5']}\n"); nd[d["path"].split("/")[0]] += 1
with gzip.open(C + "asix_results_not_on_lab.tsv.gz", "wt", encoding="utf-8", newline="\n") as out:
    out.write("path\tbytes\tmd5\n"); na = collections.Counter()
    for g in asix:
        top = g["path"].split("/")[0]
        if top in ("run-crossgen", "run-coords", "run-paper16", "run-cap", "kd", "cap", "scale", "figures", "run-hf275") and g["path"].endswith(RESULT_EXT):
            if g["md5"] not in lab_keys:
                out.write(f"{g['path']}\t{g['bytes']}\t{g['md5']}\n"); na["/".join(g["path"].split("/")[:2])] += 1

with open(C + "join_summary.txt", "w", encoding="utf-8") as f:
    f.write("LAB FILES per top folder: files, GB, on_drive, in_asix(identical), in_pcood, asix_same_name_diff_content, drive_same_name_diff_content, skip_big\n")
    for k, s in sorted(summ.items()):
        f.write(f"{k:45s} {s['files']:7d} {s['bytes']/1e9:8.3f} drive={s['on_drive']:7d} asix={s['in_asix']:6d} pcood={s['in_pcood']:3d} asixDIFF={s['asix_conflict']:5d} driveDIFF={s['drive_conflict']:5d} big={s['skip_big']}\n")
    f.write(f"\nDRIVE files whose content matches no lab file: {sum(nd.values())} (per Drive top folder)\n")
    for k, v in sorted(nd.items()):
        f.write(f"  {k:30s} {v}\n")
    f.write(f"\nASIX result-type blobs (persona-work folders) whose content matches no lab file: {sum(na.values())}\n")
    for k, v in sorted(na.items()):
        f.write(f"  {k:40s} {v}\n")
print(open(C + "join_summary.txt", encoding="utf-8").read())
