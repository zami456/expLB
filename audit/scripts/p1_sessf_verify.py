"""Phase 1 / transcript mining: verify that each recovered snippet really is at the
session+line it claims.

    python audit/scripts/p1_sessf_verify.py <flatdir> <csv> [--sample N] [--seed S]

For each sampled row it takes the longest run of non-space characters in
printed_snippet that contains a digit (a distinctive token), and checks that it
appears somewhere in that session's flat file at that lineno.  Reports OK / MISS.
"""
import csv
import os
import random
import re
import sys


def unesc(s):
    return s.replace("\\n", "\n").replace("\\t", "\t")


def main():
    flatdir, path = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]

    def opt(n, d):
        return args[args.index(n) + 1] if n in args else d

    n = int(opt("--sample", "0"))
    random.seed(int(opt("--seed", "0")))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if n and n < len(rows):
        rows = random.sample(rows, n)
    ok = miss = skip = 0
    for r in rows:
        sess, line = r["session"].strip(), r["line"].strip()
        snip = r["printed_snippet"]
        if r["found"].strip() == "n" or not sess or not line.isdigit():
            skip += 1
            continue
        snip = snip.replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\")
        toks = [t for t in re.split(r"[\s|]+", snip) if any(c.isdigit() for c in t) and len(t) >= 6]
        if not toks:
            toks = [t for t in re.split(r"[\s|]+", snip) if len(t) >= 8]
        if not toks:
            skip += 1
            continue
        tok = max(toks, key=len).strip('"\',;')
        fp = os.path.join(flatdir, "flat_%s.tsv" % sess)
        found = False
        if os.path.exists(fp):
            with open(fp, encoding="utf-8", errors="replace") as f:
                for row in f:
                    p = row.split("\t")
                    if len(p) >= 6 and p[1] == line and tok in p[5]:
                        found = True
                        break
        if found:
            ok += 1
        else:
            miss += 1
            print("MISS %-6s %s L%-6s token=%r" % (r["f_id"], sess, line, tok))
    print("%s: OK %d  MISS %d  SKIP %d  (of %d sampled)" % (os.path.basename(path), ok, miss, skip, len(rows)))


if __name__ == "__main__":
    main()
