"""Phase 3, P3-D: the remaining 25 rows of audit/phase1/sessions/findings_unrecorded.csv
(P13 checked 10: S-01..S-04, S-09, S-11..S-14, S-29). Everything here is recomputed from raw
result files in the repo working tree, or from audit/phase2/layer_sensitivity.csv.

Sections:
  S-15  the length controls on every zero-shot sycophancy/deception AUROC  (run-coords/q10_sycophancy_axis.json)
  S-35  the `random` control column in the scale tables                    (audit/phase2/layer_sensitivity.csv)
  S-24  two cap axis "variants" are the same contrast                      (layer_sensitivity.csv, cap stores)
  S-32  the kd bank's largest dataset-level effect / near-miss             (run-crossgen/kd11/banks/kd/significance.json)
  S-08  the NDIF attention-memory arithmetic conflict                      (pure arithmetic)
  S-14b the per-dataset zero-shot nulls, other layers/poolings             (run-coords/zeroshot_nulls.json)
"""
import csv, json, math, os, statistics, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BAR = "=" * 100


def jload(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


# ---------------------------------------------------------------- S-15
print(BAR)
print("S-15  Every zero-shot AUROC in run-coords/q10_sycophancy_axis.json carries two length")
print("      controls. The thesis doc quotes two cells; here is the whole grid's behaviour.")
print("      auroc -> within_length_quintiles / length_regressed_out.")
print(BAR)
q = jload("run-coords/q10_sycophancy_axis.json")
print(f"  n rows = {q['n']}")
print(f"  random control block: {json.dumps(q['random'])[:300]}")
print(f"  length_only block:    {json.dumps(q['length_only'])[:300]}")
drops = collections.defaultdict(list)
best_rows = []
for bank_pool, dirs in q["results"].items():
    for direction, blk in dirs.items():
        for L, cell in blk["key_layers"].items():
            a = cell.get("auroc")
            wq = cell.get("within_length_quintiles")
            lr = cell.get("length_regressed_out")
            if a is None:
                continue
            if wq is not None:
                drops[(bank_pool, direction)].append((int(L), a, wq, lr))
        b = blk.get("best")
        best_rows.append((bank_pool, direction, blk.get("best_layer"), b))
print(f"\n  cells with both controls: {sum(len(v) for v in drops.values())} over {len(drops)} (bank|pooling x direction) series")
print(f"\n  {'bank|pool':>14} {'direction':>12} {'n':>3} {'mean auroc':>11} {'mean wq':>9} {'mean lr':>9} "
      f"{'mean |d| wq':>12} {'mean |d| lr':>12} {'worst drop':>11}")
for k, v in sorted(drops.items()):
    a = [x[1] for x in v]
    wq = [x[2] for x in v]
    lr = [x[3] for x in v if x[3] is not None]
    dwq = [x[2] - x[1] for x in v]
    dlr = [x[3] - x[1] for x in v if x[3] is not None]
    worst = min(dwq + dlr) if (dwq or dlr) else float("nan")
    print(f"  {k[0]:>14} {k[1]:>12} {len(v):>3} {statistics.fmean(a):>11.4f} {statistics.fmean(wq):>9.4f} "
          f"{statistics.fmean(lr) if lr else float('nan'):>9.4f} {statistics.fmean(dwq):>+12.4f} "
          f"{statistics.fmean(dlr) if dlr else float('nan'):>+12.4f} {worst:>+11.4f}")
print("\n  The two cells the thesis doc quotes, plus every other key layer (paper16|mean, assistant):")
for L, a, wq, lr in sorted(drops[("paper16|mean", "assistant")]):
    print(f"      L{L:<3} auroc {a:.4f} -> quintiles {wq:.4f} ({wq-a:+.4f})  regressed {lr:.4f} ({lr-a:+.4f})")
print("\n  How often does a length control move an AUROC by more than 0.05?")
tot = bad = 0
for k, v in drops.items():
    for L, a, wq, lr in v:
        tot += 1
        if abs(wq - a) > 0.05 or (lr is not None and abs(lr - a) > 0.05):
            bad += 1
print(f"      {bad} of {tot} cells")
print("\n  Does any direction survive both controls above 0.65 at its best layer?")
for bp, dr, bl, b in sorted(best_rows):
    if b is None:
        continue
    cell = q["results"][bp][dr]["key_layers"].get(str(bl), {})
    print(f"      {bp:>14} {dr:>12} best L{bl}: auroc {b:.4f} | key-layer cell {cell.get('auroc')} "
          f"quint {cell.get('within_length_quintiles')} regr {cell.get('length_regressed_out')}")
print(f"\n  length_only (a response-length-only zero-shot scorer on the same rows) = {q['length_only']:.4f}")
print(f"  random-direction null (mean pooling): per-layer |dev| p95 = "
      f"{q['random']['mean']['per_layer_abs_dev_p95']:.4f}, best-of-layers p95 = "
      f"{q['random']['mean']['best_of_layers_p95']:.4f}, best-of-layers max = "
      f"{q['random']['mean']['best_of_layers_max']:.4f}")
print("  -> any direction whose |AUROC-0.5| does not exceed the p95 is not distinguishable from random.")
for bp, dr, bl, b in sorted(best_rows):
    if b is None:
        continue
    pool = bp.split("|")[1]
    p95 = q["random"][pool]["best_of_layers_p95"]
    dev = abs(b - 0.5) + 0.5
    print(f"      {bp:>14} {dr:>12} best {b:.4f} (|dev|+0.5 = {dev:.4f}) vs best-of-layers p95 {p95:.4f}"
          f"  -> {'BEATS null' if dev > p95 else 'inside null'}"
          f"   vs length_only {q['length_only']:.4f} -> {'beats length' if dev > q['length_only'] else 'BELOW length-only'}")

# ---------------------------------------------------------------- S-35
print()
print(BAR)
print("S-35  The `random` control column exists in every per-layer scale table and no F-entry")
print("      quotes it. Recomputed from the matrices themselves (layer_sensitivity.csv).")
print("      Question it answers: is the persona feature's transfer better than a RANDOM direction?")
print(BAR)
rows = list(csv.DictReader(open(os.path.join(R, "audit", "phase2", "layer_sensitivity.csv"), encoding="utf-8")))
for r in rows:
    r["layer"] = int(r["layer"])
    r["offdiag_mean"] = float(r["offdiag_mean"])
KEY = ("store", "pooling", "layer", "seed", "feature", "probe", "dataset_set")
IDX = {tuple(r[k] for k in KEY): r for r in rows}
STORE_LAYERS = collections.defaultdict(set)
for r in rows:
    STORE_LAYERS[r["store"]].add(r["layer"])


def od(*a):
    x = IDX.get(a)
    return x["offdiag_mean"] if x else None


for store in ("scale_3b", "scale_8b", "scale_70b", "scale_70b_paper16fp32"):
    lays = sorted(STORE_LAYERS[store])
    seeds = sorted({r["seed"] for r in rows if r["store"] == store})
    feats = sorted({r["feature"] for r in rows if r["store"] == store})
    print(f"\n  {store}: layers {lays[:4]}..{lays[-1] if lays else '-'} ({len(lays)}), seeds {seeds}, features {feats}")
    for pooling in ("mean", "last"):
        for probe in ("paper",):
            for seed in seeds:
                cells = []
                for L in lays:
                    rw = od(store, pooling, L, seed, "raw", probe, "all")
                    rnd = od(store, pooling, L, seed, "random", probe, "all")
                    ax = od(store, pooling, L, seed, "axes", probe, "all")
                    pu = od(store, pooling, L, seed, "pc:unified@2", probe, "all")
                    dp = od(store, pooling, L, seed, "datapca", probe, "all")
                    if rnd is None or rw is None:
                        continue
                    cells.append((L, rw, rnd, ax, pu, dp))
                if not cells:
                    continue
                n = len(cells)
                print(f"    {pooling} {probe} seed {seed}: {n} layers | random mean {statistics.fmean(c[2] for c in cells):.4f} "
                      f"(min {min(c[2] for c in cells):.4f} max {max(c[2] for c in cells):.4f}) | raw mean "
                      f"{statistics.fmean(c[1] for c in cells):.4f}")
                for name, i in (("axes", 3), ("pc:unified@2", 4), ("datapca", 5)):
                    v = [(c[0], c[i], c[2]) for c in cells if c[i] is not None]
                    if not v:
                        continue
                    wins = sum(1 for _, f, rn in v if f > rn)
                    print(f"        {name:<13} beats random at {wins}/{len(v)} layers | "
                          f"mean margin over random {statistics.fmean(f - rn for _, f, rn in v):+.4f} | "
                          f"worst layer L{min(v, key=lambda t: t[1]-t[2])[0]} "
                          f"{min(f - rn for _, f, rn in v):+.4f}")

# ---------------------------------------------------------------- S-24
print()
print(BAR)
print("S-24  Two of the cap bank's three declared axis variants are the same contrast")
print("      (cos = +0.9313 with capitulation, identical norm 4.411, per the transcript).")
print("      Independent test: do pc:cap_only and pc:cap_vs_default produce the SAME transfer matrix?")
print(BAR)
pairs = [("pc:cap_only", "pc:cap_vs_default"), ("pc:cap_only@2", "pc:cap_vs_default@2"),
         ("pc:capitulation", "pc:cap_only"), ("pc:capitulation", "pc:cap_vs_default"),
         ("pc:cap_vs_neutral", "pc:cap_vs_para"), ("pc:capitulation@2", "pc:cap_only@2")]
for store in sorted(s for s in STORE_LAYERS if "cap" in s):
    print(f"\n  store {store}")
    for a, b in pairs:
        d = []
        for r in rows:
            if r["store"] != store or r["feature"] != a:
                continue
            other = IDX.get((r["store"], r["pooling"], r["layer"], r["seed"], b, r["probe"], r["dataset_set"]))
            if other:
                d.append(abs(r["offdiag_mean"] - other["offdiag_mean"]))
        if d:
            print(f"    {a:<22} vs {b:<22} n={len(d):<4} max|diff| {max(d):.6f}  mean {statistics.fmean(d):.6f}  "
                  f"identical cells {sum(1 for x in d if x == 0)}/{len(d)}")

# ---------------------------------------------------------------- S-32
print()
print(BAR)
print("S-32  The kd bank's largest dataset-level effect, and the closest any of its tests came to")
print("      significance. F-31's headline is 'almost nothing clears; 6 of 4,032'.")
print("      Recomputed from the GIT copy of run-crossgen/kd11/banks/kd/significance.json (F-31's version).")
print(BAR)
p = "run-crossgen/kd11/banks/kd/significance.json"
try:
    sig = jload(p)
except Exception as e:
    print("  NOT AVAILABLE locally:", e)
    sig = None
if sig is not None:
    tests = sig if isinstance(sig, list) else sig.get("tests", sig)
    if isinstance(tests, dict):
        print("  top-level keys:", list(tests.keys())[:12])
        for k in tests:
            if isinstance(tests[k], list):
                tests = tests[k]
                break
    print(f"  n tests = {len(tests)}")
    print("  example test keys:", sorted(tests[0].keys()))
    def g(t, *names):
        for n in names:
            if n in t:
                return t[n]
        return None
    scored = [(g(t, "diff", "delta", "margin"), g(t, "dataset_q_bh", "q_bh", "q"), t) for t in tests]
    scored = [(d, q, t) for d, q, t in scored if isinstance(d, (int, float))]
    top = sorted(scored, key=lambda x: -x[0])[:8]
    print(f"\n  Largest positive effects (by diff):")
    for d, qv, t in top:
        print(f"    diff {d:+.4f}  q_bh {qv}  {g(t,'comparison','contrast')} | layer {g(t,'layer')} "
              f"pooling {g(t,'pooling')} probe {g(t,'probe')} | CI {g(t,'dataset_ci95','ci95','ci')} "
              f"| pairs_won {g(t,'pairs_won')}")
    qs = [(qv, d, t) for d, qv, t in scored if isinstance(qv, (int, float))]
    qs.sort()
    print(f"\n  Smallest BH q values (this file's family = {len(tests)} tests):")
    for qv, d, t in qs[:8]:
        print(f"    q {qv:.4f}  diff {d:+.4f}  {g(t,'comparison','contrast')} | layer {g(t,'layer')} "
              f"pooling {g(t,'pooling')} probe {g(t,'probe')} | CI {g(t,'dataset_ci95','ci95','ci')}")
    print(f"\n  tests with q < 0.05: {sum(1 for qv, _, _ in qs if qv < 0.05)}")
    print(f"  tests with q < 0.10: {sum(1 for qv, _, _ in qs if qv < 0.10)}")

# ---------------------------------------------------------------- S-08
print()
print(BAR)
print("S-08  NDIF attention-memory arithmetic: the transcript's 4.997 GiB vs the memory note's 5.36 GiB")
print(BAR)
B, T, b = 6, 1869, 4
for heads, label in ((64, "GQA key heads / 64"), (8, "kv heads 8"), (128, "128")):
    g = B * heads * T * T * b / 1024 ** 3
    print(f"  {B} x {heads} heads x {T}^2 x {b} bytes = {g:.3f} GiB   [{label}]")
print(f"  to reach 5.36 GiB you need heads = {5.36 * 1024**3 / (B * T * T * b):.2f}")
print("  NDIF's own message quoted in the transcript: 'Tried to allocate 5.00 GiB'")

# ---------------------------------------------------------------- S-14b
print()
print(BAR)
print("S-14b  Per-dataset zero-shot random-direction nulls: the full grid, not only L40 mean.")
print(BAR)
try:
    z = jload("run-coords/zeroshot_nulls.json")
except Exception as e:
    print("  NOT AVAILABLE:", e)
    z = None
if z is not None:
    print("  top-level keys:", list(z.keys())[:10])
    for corpus, blk in z.items():
        if not isinstance(blk, dict):
            continue
        per = blk.get("per_dataset") or {k: v for k, v in blk.items() if isinstance(v, dict) and "deception_axis" in v}
        if not per:
            continue
        for axis in ("deception_axis", "sycophancy_axis", "assistant_axis", "unified_axis"):
            hits = []
            for ds, cell in per.items():
                c = cell.get(axis)
                if not isinstance(c, dict):
                    continue
                hits.append((ds, c.get("auroc"), c.get("p_vs_random"), c.get("beats_random_p95")))
            if hits:
                won = sum(1 for _, _, _, b_ in hits if b_)
                print(f"    {corpus:<22} {axis:<16} beats p95 on {won}/{len(hits)} datasets | " +
                      ", ".join(f"{d}:{a:.3f}{'*' if b_ else ''}" for d, a, _, b_ in sorted(hits)))
