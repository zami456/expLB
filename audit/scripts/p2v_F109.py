"""F-109: mean single-source / pooled-LODO / row-matched LODO over the six datasets, per layer."""
import json, os, statistics, sys
d = sys.argv[1]
for fn in sorted(os.listdir(d), key=lambda s: int(''.join(c for c in s.split('_')[1][1:] if c.isdigit()))):
    p = os.path.join(d, fn)
    j = json.load(open(p, encoding="utf-8"))
    pd = j["per_dataset"]
    print(f"\n== {fn} layer={j['layer']} pooling={j['pooling']} probe={j.get('probe')} seed={j['seed']} ==")
    print("   top-level keys:", sorted(j.keys()))
    print("   datasets:", j["datasets"])
    for key in ("single_offdiag", "pooled_lodo", "pooled_lodo_match", "pooled_full", "pooled_match", "single_own"):
        vals = [v[key] for v in pd.values() if key in v]
        if vals:
            print(f"   mean {key:<20} = {statistics.fmean(vals):.4f}   (n={len(vals)})")
    for k, v in j.items():
        if k not in ("per_dataset", "datasets", "n_train") and not isinstance(v, (int, float, str)):
            print("   other:", k, json.dumps(v)[:300])
