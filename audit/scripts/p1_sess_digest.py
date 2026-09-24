"""Phase 1 (extra task): collapse run_events2_raw.tsv into a readable digest.

Groups the raw events by (session, kind, signature) for scripts/drivers/tasks, and by a
normalised body key for inline python, keeping first/last timestamp, repeat count, the
first non-empty result snippet and the first non-empty follow-up assistant text.

Usage:  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_digest.py [kind ...]
Output: stdout (piped by hand); nothing is written.
"""
import re
import sys
from collections import OrderedDict

SRC = "audit/phase1/sessions/run_events2_raw.tsv"


def norm_inline(det):
    d = re.sub(r"\s+", " ", det)
    d = re.sub(r"python3?\s+-\s*<<\s*'?\w+'?", "PY ", d)
    return d[:120]


def main():
    want = set(sys.argv[1:]) or None
    rows = [l.rstrip("\n").split("\t") for l in open(SRC, encoding="utf-8")][1:]
    g = OrderedDict()
    for r in rows:
        if len(r) < 9:
            continue
        sess, ln, ts, kind, sig, desc, det, res, nxt = r[:9]
        if want and kind not in want:
            continue
        key = (sess, kind, sig if sig != "INLINE" else norm_inline(det))
        e = g.setdefault(key, dict(n=0, first=ts, last=ts, line=ln, desc=desc, det=det, res="", nxt=""))
        e["n"] += 1
        e["last"] = max(e["last"], ts)
        if ts < e["first"]:
            e["first"] = ts
            e["line"] = ln
        if not e["res"] and res:
            e["res"] = res
        if not e["nxt"] and nxt:
            e["nxt"] = nxt
        if not e["desc"] and desc:
            e["desc"] = desc
    out = sorted(g.items(), key=lambda kv: (kv[1]["first"], kv[0][0]))
    for (sess, kind, sig), e in out:
        print("%s\t%s\tL%s\tx%d\t%s\t%s" % (sess, e["first"], e["line"], e["n"], kind, sig))
        if e["desc"]:
            print("   desc: " + e["desc"][:160])
        print("   det : " + e["det"][:300])
        if e["res"]:
            print("   res : " + e["res"][:240])
        if e["nxt"]:
            print("   say : " + e["nxt"][:240])
    print("### groups:", len(out))


if __name__ == "__main__":
    main()
