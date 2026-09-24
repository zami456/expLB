"""Every persona axis, every rung, side by side — what was actually tested and on what.

Answers, from the stores rather than from prose:
  * which persona SET each axis is built from (16 for deception, 11 for sycophancy,
    26 for unified, 8 for each knowledge-denial contrast);
  * what each scores at 3B, 8B and 70B;
  * whether the 36-persona bank merges deception and sycophancy (it does not).

Run on the lab PC, where the matrices live.
"""
import json
import os
from collections import defaultdict

NEW = r"C:/Users/user4/expLB-resplit/run-crossgen"
OLD = r"C:/Users/user4/expLB/run-crossgen"
POOL, PROBE = "mean", "paper"

STORES = [
    ("3B", "paper16", os.path.join(OLD, "scale_3b")),
    ("3B", "kd s23", os.path.join(NEW, "kdscale_3b_s23")),
    ("8B", "paper16", os.path.join(OLD, "scale_8b")),
    ("8B", "kd s23", os.path.join(NEW, "kdscale_8b_s23")),
    ("70B", "paper16", os.path.join(OLD, "scale_70b_paper16fp32")),
    ("70B", "kd s23", os.path.join(NEW, "kdscale_70b_s23")),
]

# which persona set each axis is fitted on (from pc_role_groups.json / kd_role_groups.json
# and 6_axes.py:214-244)
BASIS = {
    "raw": "— (8192-d probe, no personas)",
    "random": "— (random direction control)",
    "datapca": "— (PCA on the data itself, matched control)",
    "length": "— (response length only)",
    "axes": "the contrast direction, no PCA",
    "pc:deception": "16 = 8 deceptive + 7 honest + anchor   [the paper's deception set]",
    "pc:sycophancy": "11 = 5 syc + 5 anti-syc + anchor       [the paper's sycophancy set]",
    "pc:unified": "26 = all four groups + anchor          [paper Appendix C]",
    "pc:unified@2": "26, k=2",
    "pc:knowledge_denial": "8 = 4 feigned + 3 sincere-ignorance + anchor  [OURS]",
    "pc:kd_assertion": "8 = 4 feigned + 3 plain-assertion + anchor    [OURS]",
}


def offdiag(d, feature):
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    blk = idx.get((feature, PROBE))
    if blk is None:
        return None
    n = len(d["datasets"])
    v = [blk[i][j] for i in range(n) for j in range(n) if i != j and blk[i][j] is not None]
    return sum(v) / len(v) if v else None


def scan(path):
    m = os.path.join(path, "matrices")
    if not os.path.isdir(m):
        return None
    acc = defaultdict(list)
    nl = 0
    for fn in sorted(os.listdir(m)):
        if not (fn.startswith(POOL + "_L") and fn.endswith("_s42.json")):
            continue
        d = json.load(open(os.path.join(m, fn), encoding="utf-8"))
        nl += 1
        for f in sorted({r["feature"] for r in d["results"]}):
            v = offdiag(d, f)
            if v is not None:
                acc[f].append(v)
    return {f: sum(v) / len(v) for f, v in acc.items()}, nl


def main():
    data = {}
    feats = set()
    for rung, bank, path in STORES:
        s = scan(path)
        if not s:
            print(f"MISSING: {rung} {bank} at {path}")
            continue
        data[(rung, bank)] = s[0]
        data[(rung, bank, "n")] = s[1]
        feats |= set(s[0])

    order = ["raw", "datapca", "random", "length", "axes", "pc:deception",
             "pc:sycophancy", "pc:unified", "pc:unified@2",
             "pc:knowledge_denial", "pc:kd_assertion"]
    order += sorted(f for f in feats if f not in order)

    print("=" * 108)
    print("EVERY AXIS, EVERY RUNG — off-diagonal mean AUROC, layer-averaged")
    print("5 datasets (instructed-deception, mask, convincing-game, ai_liar, roleplaying),")
    print("k=3, mean pooling, `paper` probe, seed 42.  NOTE: no harm-pressure data in this set.")
    print("=" * 108)
    cols = [(r, b) for r, b, _ in STORES if (r, b) in data]
    head = "".join(f"{r+' '+b:>15s}" for r, b in cols)
    print(f"{'feature':22s}{head}")
    print(f"{'  (layers averaged)':22s}" + "".join(
        f"{data.get((r,b,'n'),0):>15d}" for r, b in cols))
    print("-" * 108)
    for f in order:
        if f not in feats:
            continue
        cells = ""
        for r, b in cols:
            v = data[(r, b)].get(f)
            cells += f"{v:15.4f}" if v is not None else f"{'-':>15s}"
        print(f"{f:22s}{cells}")

    print("\n" + "=" * 108)
    print("WHICH PERSONA SET EACH AXIS IS BUILT FROM")
    print("=" * 108)
    for f in order:
        if f in feats and f in BASIS:
            print(f"  {f:22s} {BASIS[f]}")

    print("\n" + "=" * 108)
    print("MARGIN OVER raw (the only comparison that matters)")
    print("=" * 108)
    print(f"{'feature':22s}{head}")
    for f in order:
        if f in ("raw", "random", "datapca", "length") or f not in feats:
            continue
        cells = ""
        for r, b in cols:
            v, raw = data[(r, b)].get(f), data[(r, b)].get("raw")
            cells += f"{v-raw:+15.4f}" if (v is not None and raw is not None) else f"{'-':>15s}"
        print(f"{f:22s}{cells}")


if __name__ == "__main__":
    main()
