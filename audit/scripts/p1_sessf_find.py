"""Phase 1 / transcript mining: search the flattened session corpus.

    python audit/scripts/p1_sessf_find.py <flatdir> <regex> [--ctx N] [--max M]
                                          [--kind K] [--win W] [--all]

Streams the flat_*.tsv files (never loads them), prints one line per hit:

    <session8> L<lineno> <ts> <kind>#<chunk> :: <window of W chars around the match>

--ctx N additionally prints the N chunks before/after the hit (same session+line)
by re-reading that file, so a printed table can be reconstructed.
"""
import os
import re
import sys


def unesc(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n"); i += 2; continue
            if n == "t":
                out.append("\t"); i += 2; continue
            if n == "\\":
                out.append("\\"); i += 2; continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    flatdir = sys.argv[1]
    pat = sys.argv[2]
    args = sys.argv[3:]

    def opt(name, default):
        if name in args:
            return args[args.index(name) + 1]
        return default

    win = int(opt("--win", 220))
    maxhits = int(opt("--max", 60))
    kindf = opt("--kind", None)
    ctx = int(opt("--ctx", 0))
    sessf = opt("--sess", None)
    rx = re.compile(pat)
    hits = 0
    files = sorted(f for f in os.listdir(flatdir) if f.startswith("flat_"))
    if sessf:
        files = [f for f in files if sessf in f]
    for fn in files:
        path = os.path.join(flatdir, fn)
        prev = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in f:
                p = row.rstrip("\n").split("\t")
                if len(p) < 6:
                    continue
                sess, ln, ts, kind, ci, text = p[0], p[1], p[2], p[3], p[4], p[5]
                prev.append(p)
                if len(prev) > ctx + 1:
                    prev.pop(0)
                if kindf and not kind.startswith(kindf):
                    continue
                m = rx.search(text)
                if not m:
                    continue
                hits += 1
                a = max(0, m.start() - win // 2)
                b = min(len(text), m.end() + win // 2)
                print("%s L%s %s %s#%s :: %s" % (sess, ln, ts[:19], kind, ci, unesc(text[a:b]).replace("\n", " | ")))
                if ctx:
                    for q in prev[:-1]:
                        print("      ^ctx %s#%s :: %s" % (q[3], q[4], unesc(q[5])[-win:].replace("\n", " | ")))
                if hits >= maxhits:
                    print("... (max hits reached)")
                    return


if __name__ == "__main__":
    main()
