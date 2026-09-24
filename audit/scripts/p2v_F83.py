"""F-83: seed-stability of the pooled-LODO margin at L40, and the Mahalanobis R^2 comparison."""
import json, os, statistics
root = "C:/Users/user4/expLB/run-premise"
print("=== seed sweep, premise_L40_mean_s*.json (root) ===")
for s in ("1", "7", "13", "42", "99"):
    p = os.path.join(root, f"premise_L40_mean_s{s}.json")
    if not os.path.exists(p): print(f" seed {s}: MISSING"); continue
    j = json.load(open(p, encoding="utf-8"))
    sm = j["summary"]
    m = sm["pooled_lodo"] - sm["single"]
    mm = sm.get("pooled_lodo_match", float('nan')) - sm["single"]
    print(f" seed {s:>2}: n_ds={len(j['datasets'])} single={sm['single']:.4f} pooled_lodo={sm['pooled_lodo']:.4f} "
          f"margin={m:+.4f} | lodo_match={sm.get('pooled_lodo_match', float('nan')):.4f} margin_match={mm:+.4f} "
          f"cov={j.get('covariance')} mtime={__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M}")
    for k in ("margin_ci95", "bootstrap", "margin"):
        if k in j: print("    extra", k, json.dumps(j[k])[:200])
print("\n=== Mahalanobis R^2 at L40 ===")
for rel in ("premise_L40_mean_s42.json", "mahal/premise_L40_mean_s42.json", "mahal/premise_L40_mean_s42_total.json"):
    p = os.path.join(root, rel)
    if not os.path.exists(p): print(rel, "MISSING"); continue
    j = json.load(open(p, encoding="utf-8"))
    pr = j.get("predictors", {})
    print(f" {rel}: cov={j.get('covariance')} shrinkage={j.get('shrinkage')} n_datasets={len(j['datasets'])} n_pairs={len(j.get('pairs', []))}")
    for k, v in pr.items():
        print(f"    {k:<12} r={v.get('r'):.4f} r2={v.get('r2'):.4f} n={v.get('n')}")
