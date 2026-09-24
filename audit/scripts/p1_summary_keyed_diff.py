"""Keyed comparison of crossgen summary.json versions (git HEAD vs lab copy in audit/cache/labpc/).
Records are matched on (pooling, layer, seed, feature, probe); for matched records the scalar metrics are compared.
Usage: python audit/scripts/p1_summary_keyed_diff.py <relpath> [...]"""
import json, subprocess, sys
KEY = ("pooling", "layer", "seed", "feature", "probe")
METRICS = ("diag_mean", "offdiag_mean_10", "offdiag_mean_9", "tpr_diag_mean", "tpr_offdiag_mean_10", "generalization_gap_10")
for rel in sys.argv[1:]:
    lab = json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))
    git = json.loads(subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True).stdout.decode("utf-8"))
    L = {tuple(r.get(k) for k in KEY): r for r in lab["records"]}
    G = {tuple(r.get(k) for k in KEY): r for r in git["records"]}
    common = set(L) & set(G)
    worst, ndiff = 0.0, 0
    ex = []
    for k in common:
        for m in METRICS:
            a, b = G[k].get(m), L[k].get(m)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d = abs(a - b)
                if d > 1e-9:
                    ndiff += 1
                    if d > worst: worst = d
                    if len(ex) < 5 and d > 1e-6: ex.append((k, m, a, b))
    lay = lambda D: sorted({k[1] for k in D})
    print(f"== {rel}\n   records git={len(G)} lab={len(L)} common={len(common)} git-only={len(set(G)-set(L))}")
    print(f"   git layers={lay(G)} seeds={sorted({k[2] for k in G})}; lab layers={lay(L)[:25]} seeds={sorted({k[2] for k in L})}")
    print(f"   metric values differing on common records: {ndiff}; max |delta| = {worst:.3g}")
    for k, m, a, b in ex: print(f"     {k} {m}: git={a:.6f} lab={b:.6f}")
