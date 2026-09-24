"""Keyed comparison of significance.json versions (git HEAD vs lab copy in audit/cache/labpc/).
Tests are matched on (block, pooling, layer, probe, comparison, seed?); numeric fields compared.
Usage: python audit/scripts/p1_significance_keyed_diff.py <relpath> [...]"""
import json, subprocess, sys
for rel in sys.argv[1:]:
    lab = json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))
    git = json.loads(subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True).stdout.decode("utf-8"))
    t0 = lab["tests"][0]
    keyf = [k for k in ("block", "pooling", "layer", "probe", "comparison", "seed", "bank", "feature") if k in t0]
    key = lambda t: tuple(str(t.get(k)) for k in keyf)
    L = {}; G = {}
    for t in lab["tests"]: L.setdefault(key(t), []).append(t)
    for t in git["tests"]: G.setdefault(key(t), []).append(t)
    dupL = sum(len(v) > 1 for v in L.values()); dupG = sum(len(v) > 1 for v in G.values())
    common = set(L) & set(G)
    nd = 0; worst = {}; ex = []
    for k in common:
        a, b = G[k][0], L[k][0]
        for f, va in a.items():
            vb = b.get(f)
            if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb) == 2 and all(isinstance(x, (int, float)) for x in va + vb):
                pairs = [(f + "[0]", va[0], vb[0]), (f + "[1]", va[1], vb[1])]
            elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                pairs = [(f, va, vb)]
            else:
                if va != vb and f not in ("row_ci95",): pairs = [(f, str(va)[:30], str(vb)[:30])]
                else: pairs = []
            for name, x, y in pairs:
                if isinstance(x, str):
                    nd += 1; worst[name] = "non-numeric"; continue
                d = abs(x - y)
                if d > 1e-9:
                    nd += 1; worst[name] = max(worst.get(name, 0), d)
                    if len(ex) < 4: ex.append((k, name, x, y))
    print(f"== {rel}  key={keyf}\n   tests git={len(git['tests'])} lab={len(lab['tests'])} keys common={len(common)} git-only={len(set(G)-set(L))} dup-keys git={dupG} lab={dupL}")
    print(f"   differing fields on common tests: {nd}; max |delta| per field: { {k: (round(v, 6) if isinstance(v, float) else v) for k, v in worst.items()} }")
    for k, n, x, y in ex: print(f"     {k} {n}: git={x} lab={y}")
