"""Phase 3, P3-D supplement 2.
  X1  persona axes vs a RANDOM direction, 10-dataset set vs the paper's five (passA, 80 layers)
  S-06  the Q8 per-token table: TPR at 1% FPR, including the zero-shot axis row nobody quotes
  S-23  the earliest honest self-assessment: the 3-probe x 4-feature table it rested on
  S-14c the per-dataset zero-shot nulls in the other three corpora of zeroshot_nulls.json
"""
import csv, json, os, statistics, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BAR = "=" * 100
rows = list(csv.DictReader(open(os.path.join(R, "audit", "phase2", "layer_sensitivity.csv"), encoding="utf-8")))
for r in rows:
    r["layer"] = int(r["layer"])
    r["offdiag_mean"] = float(r["offdiag_mean"])
IDX = {(r["store"], r["pooling"], r["layer"], r["seed"], r["feature"], r["probe"], r["dataset_set"]): r
       for r in rows}


def od(*a):
    x = IDX.get(a)
    return x["offdiag_mean"] if x else None


print(BAR)
print("X1  Persona axes vs a RANDOM direction of the same dimensionality, on the SAME grid,")
print("    on the 10-dataset set and on the paper's five. passA, 80 layers, seed 42.")
print("    (`random` is the project's own control feature, computed in every matrix.)")
print(BAR)
for ds in ("all", "paper5"):
    for pool in ("mean", "last"):
        for probe in ("paper", "mms", "lr_reg10"):
            out = []
            for feat, ctrl in (("axes", "random"), ("pc:unified", "random"),
                               ("pc:unified@2", "random@2"), ("pc:deception", "random")):
                d = []
                for L in range(80):
                    f_, c_ = od("passA", pool, L, "42", feat, probe, ds), od("passA", pool, L, "42", ctrl, probe, ds)
                    if f_ is not None and c_ is not None:
                        d.append(f_ - c_)
                if d:
                    out.append(f"{feat} {statistics.fmean(d):+.4f} ({sum(1 for x in d if x > 0)}/{len(d)})")
            print(f"  {ds:<7} {pool:<5} {probe:<9} " + " | ".join(out))
print("\n  Same thing stated as the gap that matters: (feature - raw) minus (random - raw)")
for ds in ("all", "paper5"):
    for pool in ("mean",):
        for probe in ("paper",):
            for feat, ctrl in (("axes", "random"), ("pc:unified@2", "random@2")):
                a = [od("passA", pool, L, "42", feat, probe, ds) - od("passA", pool, L, "42", "raw", probe, ds)
                     for L in range(80) if od("passA", pool, L, "42", feat, probe, ds) is not None]
                b = [od("passA", pool, L, "42", ctrl, probe, ds) - od("passA", pool, L, "42", "raw", probe, ds)
                     for L in range(80) if od("passA", pool, L, "42", ctrl, probe, ds) is not None]
                print(f"    {ds:<7} {feat:<13}: mean margin vs raw {statistics.fmean(a):+.4f}; "
                      f"the random control's own margin vs raw {statistics.fmean(b):+.4f}; "
                      f"difference {statistics.fmean(a)-statistics.fmean(b):+.4f}")

print()
print(BAR)
print("S-06  Q8: per-token vs pooled. The leave-one-dataset-out table, incl. TPR at 1% FPR.")
print(BAR)
for fn in ("run-coords/pertoken_q8.json", "run-coords/q8_trained_pertoken.json"):
    p = os.path.join(R, fn)
    if not os.path.exists(p):
        print(f"  {fn}: NOT FOUND locally")
        continue
    d = json.load(open(p, encoding="utf-8"))
    print(f"\n  {fn}: top-level keys {list(d.keys())[:12]}")
    txt = json.dumps(d)
    print(f"    contains 'tpr': {'tpr' in txt.lower()}   length {len(txt)} chars")

    def rec(o, path="", depth=0):
        if depth > 4:
            return
        if isinstance(o, dict):
            if any(k.lower().startswith("tpr") for k in o):
                print(f"    {path}: " + " ".join(f"{k}={o[k]}" for k in sorted(o) if not isinstance(o[k], (list, dict))))
                return
            for k, v in list(o.items())[:60]:
                rec(v, f"{path}/{k}", depth + 1)
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            for i, v in enumerate(o[:60]):
                rec(v, f"{path}[{i}]", depth + 1)
    rec(d)

print()
print(BAR)
print("S-23  The earliest honest self-assessment (2026-09-15): 'the signal is positive but about")
print("      half the size the paper claims'. Its table, from run-coords/probes_5sets.json.")
print(BAR)
for fn in ("run-coords/probes_5sets.json", "run-coords/probes_L20.json", "run-coords/smoke.json",
           "run-coords/smoke2.json"):
    p = os.path.join(R, fn)
    if not os.path.exists(p):
        print(f"  {fn}: NOT FOUND")
        continue
    d = json.load(open(p, encoding="utf-8"))
    print(f"\n  {fn}: keys {list(d.keys())[:14]}")
    s = json.dumps(d)
    print("   ", s[:1400].replace("\\n", " "))

print()
print(BAR)
print("S-14c  zeroshot_nulls.json -- the other three corpora")
print(BAR)
z = json.load(open(os.path.join(R, "run-coords/zeroshot_nulls.json"), encoding="utf-8"))
for corpus, blk in z.items():
    print(f"\n  == {corpus}: keys {list(blk.keys())[:14] if isinstance(blk, dict) else type(blk)}")
    if not isinstance(blk, dict):
        continue
    s = json.dumps(blk)
    print("    ", s[:1600])
