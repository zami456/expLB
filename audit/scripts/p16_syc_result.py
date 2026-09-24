"""Phase 6 / 6D-1 -- read the pc:sycophancy run written to expLB-resplit.

Runs ON THE LAB PC, read-only. Prints only.

The run: 27_crossgen.py, run-coords/passB, paper16 bank (axes_vectors_s3_fp32.pt), 11 datasets
(harm-pressure-kr-matched passed explicitly), seed 42, probe `paper`, the four default controls
kept (raw / random / datapca / length) plus pc:sycophancy.

`offdiag_mean_10` is ALL 11 datasets and `offdiag_mean_9` is all but sycophancy_mt -- the labels
are historical counts, not dataset counts (DF-01).
"""
import json

S = r"C:\Users\user4\expLB-resplit\phase6\run-crossgen\summary.json"
G = r"C:\Users\user4\expLB-resplit\phase6\run-crossgen\significance.json"

j = json.load(open(S))
rs = [r for r in j["records"] if r["probe"] == "paper"]
print(f"datasets in store : {len(j['datasets'])}   block9 = {len(j['block9'])} "
      f"(all but sycophancy_mt)")
print("offdiag_mean_10 = ALL 11 datasets ; offdiag_mean_9 = the 10 without sycophancy_mt\n")

FE = ["raw", "datapca", "random", "length", "pc:deception", "pc:unified", "axes", "pc:sycophancy"]
for pool in ("mean", "last"):
    print(f"===== pooling {pool} =====")
    print(f"  {'feature':14s} " + "  ".join(f"L{L}: all11 / no-syc" for L in (20, 40, 44)))
    for f in FE:
        cells = []
        for L in (20, 40, 44):
            r = next((x for x in rs if x["feature"] == f and x["layer"] == L
                      and x["pooling"] == pool), None)
            cells.append(f"{r['offdiag_mean_10']:.4f} / {r['offdiag_mean_9']:.4f}" if r else "  --  ")
        mark = "  <<<" if f == "pc:sycophancy" else ""
        print(f"  {f:14s} " + "   ".join(cells) + mark)
    print("  --- pc:sycophancy minus the controls ---")
    for L in (20, 40, 44):
        s = next((x for x in rs if x["feature"] == "pc:sycophancy" and x["layer"] == L
                  and x["pooling"] == pool), None)
        if not s:
            continue
        out = []
        for c in ("raw", "datapca", "random", "length"):
            r = next((x for x in rs if x["feature"] == c and x["layer"] == L
                      and x["pooling"] == pool), None)
            if r:
                out.append(f"{c} {s['offdiag_mean_10'] - r['offdiag_mean_10']:+.4f}"
                           f" / {s['offdiag_mean_9'] - r['offdiag_mean_9']:+.4f}")
        print(f"    L{L}: " + "   ".join(out))
    print()

try:
    g = json.load(open(G))
    tests = g.get("tests", g if isinstance(g, list) else [])
    syc = [t for t in tests if "sycophancy" in json.dumps(t)]
    print(f"===== significance.json : {len(tests)} tests, {len(syc)} mentioning sycophancy =====")
    if syc:
        k = list(syc[0].keys())
        print("  keys:", k)
    for t in syc[:24]:
        lbl = t.get("comparison") or f"{t.get('feature_a')} - {t.get('feature_b')}"
        print(f"  {str(lbl)[:40]:40s} block={t.get('block')} pool={t.get('pooling')} "
              f"L={t.get('layer')} probe={t.get('probe')} diff={t.get('diff')} q={t.get('q')} "
              f"sig={t.get('significant_datasets')}")
except Exception as e:
    print("significance.json:", type(e).__name__, e)
