"""Owner-authorised backup (RF-10): select lab files that exist ONLY on the lab PC — not on Drive (by md5, or by
path+size for >200 MB files that were not hashed), not in git on asix or his branch. Excludes binaries of tools
(rclone.exe), lock files and the audit's own scratch. Writes audit/fix/backup_list.tsv (root, relpath, bytes, md5)."""
import collections, csv, gzip, os
csv.field_size_limit(10**9)
os.makedirs("audit/fix", exist_ok=True)
drive_size = {}
with gzip.open("audit/phase1/labpc/drive_census.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        drive_size[(os.path.basename(r["path"]), r["bytes"])] = r["path"]
sel, groups = [], collections.defaultdict(lambda: [0, 0])
with gzip.open("audit/phase1/labpc/joined_labpc.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["on_drive"] == "1" or r["in_asix"] == "1" or r["in_pcood"] == "1":
            continue
        rel = r["relpath"]; root = r["root"]
        if r["md5"] == "SKIP_BIG" and (os.path.basename(rel), r["bytes"]) in drive_size:
            continue  # big unhashed file present on Drive with same name+size
        if rel.endswith(".lock") or rel.startswith("rclone-") or root.split(chr(92))[-1] == "tools" or rel.startswith("tools/"):
            continue
        sel.append((root, rel, int(r["bytes"]), r["md5"]))
        top = root.split(chr(92))[-1] + "/" + (rel.split("/")[0] if "/" in rel else "<loose>")
        groups[top][0] += 1; groups[top][1] += int(r["bytes"])
with open("audit/fix/backup_list.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("root\trelpath\tbytes\tmd5\n")
    for s in sel: f.write("\t".join(map(str, s)) + "\n")
for k, (n, b) in sorted(groups.items()): print(f"{n:6d} {b/1e6:10.1f} MB  {k}")
print(f"TOTAL {len(sel)} files, {sum(s[2] for s in sel)/1e9:.2f} GB")
