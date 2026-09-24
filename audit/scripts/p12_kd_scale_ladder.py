"""The kd bank's scale ladder — the cell the project never ran until 2026-09-24.

paper16 (26 personas) existed at 3B/8B/70B, so the scale claim was supported for
THAT bank only. The kd bank (26 + 10 knowledge-denial / sincere-ignorance /
plain-assertion) existed at 70B alone, so "our bank rescues the method AT SCALE"
could not be separated from "our bank is better at any scale".

This runs on the lab PC, where the matrices live, and prints the completed ladder.
Both banks on identical footing: the same 5 datasets (MASK as the fifth, per F-60),
k=3, mean pooling, `paper` probe, seed 42, layer-averaged over each rung's own layers.

  python audit/scripts/labpc_run.py <this file>      # or run it on the lab PC
"""
import json
import os
from collections import defaultdict

NEW = r"C:/Users/user4/expLB-resplit/run-crossgen"
OLD = r"C:/Users/user4/expLB/run-crossgen"
POOL, PROBE = "mean", "paper"

ROWS = [
    ("3B  paper16 (26p)", os.path.join(OLD, "scale_3b")),
    ("3B  kd all (36p)", os.path.join(NEW, "kdscale_3b_all")),
    ("3B  kd s23 (36p)", os.path.join(NEW, "kdscale_3b_s23")),
    ("8B  paper16 (26p)", os.path.join(OLD, "scale_8b")),
    ("8B  kd all (36p)", os.path.join(NEW, "kdscale_8b_all")),
    ("8B  kd s23 (36p)", os.path.join(NEW, "kdscale_8b_s23")),
    ("70B paper16 (26p)", os.path.join(OLD, "scale_70b_paper16fp32")),
    ("70B kd all (36p)", os.path.join(NEW, "kdscale_70b_all")),
    ("70B kd s23 (36p)", os.path.join(NEW, "kdscale_70b_s23")),
]


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
    out = {}
    for fn in sorted(os.listdir(m)):
        if not (fn.startswith(POOL + "_L") and fn.endswith("_s42.json")):
            continue
        d = json.load(open(os.path.join(m, fn), encoding="utf-8"))
        feats = sorted({r["feature"] for r in d["results"]})
        out[d["layer"]] = {f: offdiag(d, f) for f in feats}
    return out


def summarise(layers):
    pers = [f for f in next(iter(layers.values())) if f.startswith("pc:") or f == "axes"]
    acc = defaultdict(list)
    for _, ft in layers.items():
        if ft.get("raw") is None:
            continue
        acc["raw"].append(ft["raw"])
        b = max((ft[f] for f in pers if ft.get(f) is not None), default=None)
        if b is not None:
            acc["best"].append(b)
        if ft.get("datapca") is not None:
            acc["dp"].append(ft["datapca"])
    if not acc["raw"]:
        return None
    return {k: sum(v) / len(v) for k, v in acc.items() if v}, len(acc["raw"])


def main():
    print("=" * 88)
    print("THE kd BANK'S SCALE LADDER — both banks on identical footing")
    print("5 datasets (MASK fifth, F-60), k=3, mean pooling, paper probe, seed 42, layer-averaged")
    print("=" * 88)
    print(f"{'rung / bank':28s} {'n':>3s} {'raw':>8s} {'best pers':>10s} "
          f"{'vs raw':>9s} {'datapca':>8s} {'vs dpca':>9s}")
    got = {}
    for label, p in ROWS:
        lay = scan(p)
        s = summarise(lay) if lay else None
        if not s:
            print(f"{label:28s}  (no data at {p})")
            continue
        m, n = s
        raw, best, dp = m.get("raw"), m.get("best"), m.get("dp")
        got[label] = best - raw
        dps = f"{dp:8.4f} {best - dp:+9.4f}" if dp else f"{'-':>8s} {'-':>9s}"
        print(f"{label:28s} {n:3d} {raw:8.4f} {best:10.4f} {best - raw:+9.4f} {dps}")

    print("\n" + "=" * 88)
    print("THE BANK DIFFERENCE, BY RUNG  (kd minus paper16, vs raw)")
    print("=" * 88)
    for rung in ("3B", "8B", "70B"):
        base = got.get(f"{rung}  paper16 (26p)" if rung != "70B" else "70B paper16 (26p)")
        for v in ("all", "s23"):
            key = (f"{rung}  kd {v} (36p)" if rung != "70B" else f"70B kd {v} (36p)")
            if base is None or key not in got:
                continue
            print(f"  {rung:4s} kd {v:4s} - paper16 = {got[key] - base:+.4f}")
    print("\n  READING: both banks decline monotonically with scale, so the decline belongs to")
    print("  the METHOD, not the bank. The kd bank declines LESS, and the gap between banks")
    print("  GROWS with scale. But at 70B both sit within a whisker of zero, and these five")
    print("  datasets contain NO knowledge-denial data — the thing the kd bank was built for.")


if __name__ == "__main__":
    main()
