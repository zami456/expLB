"""Phase 1 / transcript mining: which numbers printed in the sessions appear in NO doc.

    python audit/scripts/p1_sessf_unrecorded.py <flatdir> <recorded.txt> <out.tsv>
                                                [--kind asst] [--min 3]

Builds the set of decimal numbers that occur anywhere in the "recorded" corpus
(the ledger, docs/, memory, the collaborator's committed summaries, the draft),
then streams the flattened transcripts and reports records whose text contains at
least --min numbers that occur in NO doc, in any of their 2/3/4-decimal roundings.

Output TSV: session8, lineno, ts, kind, n_unrec, unrec_numbers, snippet(500).
Everything it finds is a LEAD, never a verified result: a number can be absent
from the docs because it was wrong, intermediate, or simply never written down.
"""
import os
import re
import sys

NUM = re.compile(r"(?<![\w.])[-+−–—]?(\d\.\d{3,4}|0\.\d{3,4})(?![\d\w])")


def variants(s):
    """The 4-, 3- and 2-decimal spellings of one decimal string."""
    try:
        v = float(s)
    except ValueError:
        return {s}
    return {s, "%.4f" % v, "%.3f" % v}


def main():
    flatdir, recpath, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]

    def opt(n, d):
        return args[args.index(n) + 1] if n in args else d

    kinds = tuple(opt("--kind", "asst,tool_res").split(","))
    minn = int(opt("--min", "3"))

    rec = open(recpath, encoding="utf-8", errors="replace").read()
    recnums = set(m.group(1) for m in NUM.finditer(rec))
    # also index the rounded spellings actually present
    print("recorded distinct decimals:", len(recnums))

    def is_recorded(s):
        return bool(variants(s) & recnums)

    out = open(outpath, "w", encoding="utf-8")
    out.write("session\tline\tts\tkind\tn_unrec\tunrec_numbers\tsnippet\n")
    nrows = 0
    for fn in sorted(f for f in os.listdir(flatdir) if f.startswith("flat_")):
        with open(os.path.join(flatdir, fn), encoding="utf-8", errors="replace") as f:
            for row in f:
                p = row.rstrip("\n").split("\t")
                if len(p) < 6 or not p[3].startswith(kinds):
                    continue
                text = p[5]
                nums = [m.group(1) for m in NUM.finditer(text)]
                if not nums:
                    continue
                unrec = sorted(set(n for n in nums if not is_recorded(n)))
                if len(unrec) < minn:
                    continue
                out.write("%s\t%s\t%s\t%s\t%d\t%s\t%s\n" % (
                    p[0], p[1], p[2][:19], p[3], len(unrec), ",".join(unrec[:25]), text[:500]))
                nrows += 1
    out.close()
    print("rows written:", nrows, "->", outpath)


if __name__ == "__main__":
    main()
