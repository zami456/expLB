"""Issue 2 — does the seeds-0-4 feature-coverage gap exist, in which stores, and does any
claim take a multi-seed mean over an affected store?

P3-D reported that in `passA`, `passC` and `<root>` seeds 0-4 carry only 3 features and
2 probes while seed 42 carries the full list, which would make any 6-seed mean over those
stores invalid.  W4-B confirmed the premise before it was killed ("the asymmetry is real
and wider") but never reached the consequential half.

This settles both halves from `audit/phase2/layer_sensitivity.csv` alone.  Read-only.
"""

import collections
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
LS = ROOT / "audit" / "phase2" / "layer_sensitivity.csv"


def main():
    rows = list(csv.DictReader(LS.open(encoding="utf-8")))
    print(f"layer_sensitivity.csv rows: {len(rows)}\n")

    # feature / probe / layer coverage per (store, seed)
    cov = collections.defaultdict(lambda: {"f": set(), "p": set(), "L": set(), "n": 0})
    for r in rows:
        c = cov[(r["store"], r["seed"])]
        c["f"].add(r["feature"]); c["p"].add(r["probe"])
        c["L"].add(int(r["layer"])); c["n"] += 1

    stores = sorted({s for s, _ in cov})
    print(f"{'store':26}{'seed':>6}{'rows':>7}{'feat':>6}{'probe':>7}{'layers':>8}")
    affected = []
    for s in stores:
        seeds = sorted({sd for st, sd in cov if st == s}, key=lambda x: int(x))
        if len(seeds) < 2:
            continue
        # the reference is the richest seed in this store
        ref = max(seeds, key=lambda sd: len(cov[(s, sd)]["f"]))
        nf_ref = len(cov[(s, ref)]["f"])
        gap = False
        for sd in seeds:
            c = cov[(s, sd)]
            flag = ""
            if len(c["f"]) < nf_ref:
                flag = f"  <-- {nf_ref - len(c['f'])} fewer features than seed {ref}"
                gap = True
            print(f"{s:26}{sd:>6}{c['n']:>7}{len(c['f']):>6}{len(c['p']):>7}{len(c['L']):>8}{flag}")
        if gap:
            affected.append(s)
        print()

    print("=" * 78)
    print(f"STORES WITH A SEED FEATURE-COVERAGE GAP: {len(affected)}")
    for s in affected:
        seeds = sorted({sd for st, sd in cov if st == s}, key=lambda x: int(x))
        ref = max(seeds, key=lambda sd: len(cov[(s, sd)]["f"]))
        poor = [sd for sd in seeds if len(cov[(s, sd)]["f"]) < len(cov[(s, ref)]["f"])]
        rich_only = sorted(cov[(s, ref)]["f"] - set.union(*[cov[(s, sd)]["f"] for sd in poor]))
        print(f"  {s}: seeds {poor} lack {len(rich_only)} features present at seed {ref}")
        print(f"      missing: {', '.join(rich_only)}")

    print()
    print("=" * 78)
    print("CONSEQUENCE — which features are SAFE to average over all seeds, per store")
    for s in affected:
        seeds = sorted({sd for st, sd in cov if st == s}, key=lambda x: int(x))
        common = set.intersection(*[cov[(s, sd)]["f"] for sd in seeds])
        print(f"  {s}: safe across all {len(seeds)} seeds -> {', '.join(sorted(common))}")


if __name__ == "__main__":
    main()
