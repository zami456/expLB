"""READ-ONLY lookup in the lab census (audit/phase1/labpc/joined_labpc.tsv.gz), streamed line by line.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_fmap_F055-F081_lookup.py <regex> [<regex> ...]
Each regex is matched (re.search) against relpath of rows whose root is C:\\Users\\user4\\expLB (or any root with --any).
Prints: relpath | bytes | mtime | md5_lf | on_drive | in_asix(asix_paths) | in_pcood | asix_same_name_diff_content | drive_same_name_diff_content
"""
import gzip, re, sys

args = sys.argv[1:]
anyroot = "--any" in args
args = [a for a in args if a != "--any"]
rxs = [re.compile(a) for a in args]
with gzip.open("audit/phase1/labpc/joined_labpc.tsv.gz", "rt", encoding="utf-8", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    for line in f:
        c = line.rstrip("\n").split("\t")
        if len(c) < 14:
            c += [""] * (14 - len(c))
        root, rel = c[0], c[1]
        if not anyroot and not root.endswith("expLB"):
            continue
        if any(r.search(rel) for r in rxs):
            asix = c[9] if c[8] == "1" else ""
            print(" | ".join([(root + "/" if anyroot else "") + rel, c[2], c[3], c[5][:12], "drive=" + c[6],
                              "asix=" + c[8] + (":" + asix if asix else ""), "pcood=" + c[10],
                              "asixDIFF=" + c[12], "driveDIFF=" + c[13][:80]]))
