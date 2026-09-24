"""Read-only lookup for the F028-F054 mapping batch.
For each repo-relative path (or regex with re:), print: lab census row(s) (bytes, mtime, md5, md5_lf, on_drive,
in_asix, asix_same_name_diff_content) and the asix HEAD git blob md5 for the same path, with a SAME/DIFFERENT verdict
(git blob md5 vs lab md5_lf). Streams the gz census; never loads it whole.
Usage: python audit/scripts/p1_fmap_F028-F054_lookup.py <path-or-re:regex> [...]
"""
import gzip, re, sys
AUD = "audit/phase1/"
pats = sys.argv[1:]
rx = [re.compile(p[3:]) if p.startswith("re:") else None for p in pats]
def match(rel):
    for p, r in zip(pats, rx):
        if (r and r.search(rel)) or (not r and rel == p):
            return True
    return False
git = {}
for line in open(AUD + "census_asix_git.tsv", encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 5 and f[0] == "HEAD" and match(f[1]):
        git[f[1]] = (f[2], f[4])
seen = set()
with gzip.open(AUD + "labpc/joined_labpc.tsv.gz", "rt", encoding="utf-8") as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if not f[0].endswith("expLB") or not match(f[1].replace("\\", "/")):
            continue
        rel = f[1].replace("\\", "/"); seen.add(rel)
        g = git.get(rel)
        gv = "GIT-ABSENT" if not g else ("GIT=LAB" if g[1] in (f[5], f[4]) else "GIT!=LAB")
        print(f"LAB\t{rel}\t{f[2]}\t{f[3]}\tmd5={f[4]}\tmd5lf={f[5]}\tdrive={f[6]}\tin_asix={f[8]}\tasixDIFF={f[12]}\tdriveDIFF={f[13]}\t{gv}")
for rel, (b, m) in sorted(git.items()):
    if rel not in seen:
        print(f"GIT-ONLY\t{rel}\t{b}\tmd5={m}")
