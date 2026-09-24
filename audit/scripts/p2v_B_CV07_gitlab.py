"""Verify batch B / CV-07: is there really a git-vs-lab version conflict for the kd11 passA
(and banks/kd_s23) summary/significance JSONs, with the record/test counts Phase 1 states?

Counts, for each side:
  summary.json      -> number of records, distinct layers, distinct datasets, distinct seeds
  significance.json -> number of tests, distinct layers

git side  : `git show HEAD:<path>`   (the committed version on branch asix)
lab side  : audit/cache/labpc/<path> (md5-verified tar-over-ssh copies of the lab PC files)
"""
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATHS = [
    "run-crossgen/kd11/passA/summary.json",
    "run-crossgen/kd11/passA/significance.json",
    "run-crossgen/kd11/banks/kd_s23/summary.json",
    "run-crossgen/kd11/banks/kd_s23/significance.json",
]


def describe(obj, kind):
    if kind == "summary":
        recs = obj["records"] if isinstance(obj, dict) and "records" in obj else obj
        if isinstance(recs, dict):
            for k in ("results", "rows", "entries"):
                if k in recs:
                    recs = recs[k]
                    break
        lay = sorted({r.get("layer") for r in recs if isinstance(r, dict)})
        seeds = sorted({r.get("seed") for r in recs if isinstance(r, dict)})
        pool = sorted({r.get("pooling") for r in recs if isinstance(r, dict)})
        ds = set()
        for r in recs:
            if isinstance(r, dict):
                for k in ("datasets", "dataset"):
                    v = r.get(k)
                    if isinstance(v, list):
                        ds |= set(v)
                    elif v:
                        ds.add(v)
        return "%d records | layers %s | seeds %s | pooling %s | datasets %d" % (
            len(recs), lay, seeds, pool, len(ds))
    tests = obj["tests"] if isinstance(obj, dict) and "tests" in obj else obj
    lay = sorted({t.get("layer") for t in tests if isinstance(t, dict)})
    blocks = sorted({t.get("block") for t in tests if isinstance(t, dict)})
    return "%d tests | layers %s | blocks %s" % (len(tests), lay, blocks)


for p in PATHS:
    kind = "summary" if p.endswith("summary.json") else "sig"
    print("=" * 78)
    print(p)
    try:
        raw = subprocess.run(["git", "show", "HEAD:" + p], cwd=ROOT, capture_output=True, check=True).stdout
        print("  git  :", describe(json.loads(raw.decode("utf-8")), kind), "| %d B" % len(raw))
    except subprocess.CalledProcessError:
        print("  git  : NOT IN HEAD")
    lab = os.path.join(ROOT, "audit", "cache", "labpc", *p.split("/"))
    if os.path.exists(lab):
        b = open(lab, "rb").read()
        print("  lab  :", describe(json.loads(b.decode("utf-8")), kind), "| %d B" % len(b))
    else:
        print("  lab  : not in audit/cache")
    loc = os.path.join(ROOT, *p.split("/"))
    if os.path.exists(loc):
        b = open(loc, "rb").read()
        print("  wtree:", describe(json.loads(b.decode("utf-8")), kind), "| %d B" % len(b))
    else:
        print("  wtree: absent from the working tree")
