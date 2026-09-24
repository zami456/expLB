"""P12-briefing follow-up on the lab: the leave-one-CLUSTER-out gains (O-003) computed from the
per-target cells, plus whatever the files' own `summary` blocks say, and the volume-vs-diversity
test that the briefing also reports as p = 0.38.  Read-only."""
import json
import os
import statistics
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB"
PREMISE = f"{ROOT}/run-premise"
LEAKED = ("harm-pressure-knowledge-report", "harm-pressure-kr-matched")

for L in (20, 32, 40):
    fp = f"{PREMISE}/clustered/clustered_L{L}_mean_s42.json"
    c = json.load(open(fp, encoding="utf-8"))
    print("=" * 78)
    print(f"L{L}   {fp}")
    print(f"  clusters = {json.dumps(c.get('clusters'))}")
    print(f"  leaked   = {json.dumps(c.get('leaked'))}")
    print(f"  probe={c.get('probe')} pooling={c.get('pooling')} seed={c.get('seed')}")
    print(f"  summary (verbatim) = {json.dumps(c.get('summary'), indent=2)}")
    per = c["per_dataset"]
    names = list(per)
    single = np.array([per[t]["single_offdiag"] for t in names])
    lodo = np.array([per[t]["pooled_lodo"] for t in names])
    loco = np.array([per[t]["pooled_loco"] for t in names])
    clean = [i for i, t in enumerate(names) if t not in LEAKED]
    print(f"  n targets = {len(names)}   clean = {len(clean)}")
    print(f"  mean single_offdiag = {single.mean():.4f}")
    print(f"  gain LODO (all 11)  = {(lodo - single).mean():+.4f}"
          f"   positive {int((lodo > single).sum())}/{len(names)}")
    print(f"  gain LOCO (all 11)  = {(loco - single).mean():+.4f}"
          f"   positive {int((loco > single).sum())}/{len(names)}")
    print(f"  gain LOCO (clean 9) = {(loco[clean] - single[clean]).mean():+.4f}"
          f"   positive {int((loco[clean] > single[clean]).sum())}/{len(clean)}")
    print(f"  HP-KR  LODO {per['harm-pressure-knowledge-report']['pooled_lodo']:.4f}"
          f"  LOCO {per['harm-pressure-knowledge-report']['pooled_loco']:.4f}"
          f"  drop {per['harm-pressure-knowledge-report']['pooled_lodo'] - per['harm-pressure-knowledge-report']['pooled_loco']:.4f}")
    neg = [t for t, a, b in zip(names, loco, single) if a <= b]
    print(f"  LOCO targets NOT positive: {neg}")

# the other p = 0.38 in the briefing: pooled full vs pooled row-budget-matched
print("=" * 78)
print("the briefing's OTHER p = 0.38: 'pooled ~6,700 rows and pooled ~666 rows are "
      "statistically indistinguishable (p = 0.38)'")
fp = f"{PREMISE}/premise_L40_mean_s42.json"
d = json.load(open(fp, encoding="utf-8"))
per = d["per_dataset"]
names = list(per)
full = np.array([per[t]["pooled_full"] for t in names])
match = np.array([per[t]["pooled_match"] for t in names])
lodo = np.array([per[t]["pooled_lodo"] for t in names])
lodo_m = np.array([per[t]["pooled_lodo_match"] for t in names])
print(f"  n_train = {json.dumps(d.get('n_train'))}")
print(f"  summary  pooled_full {d['summary']['pooled_full']:.4f}"
      f"   pooled_match {d['summary']['pooled_match']:.4f}"
      f"   pooled_lodo {d['summary']['pooled_lodo']:.4f}"
      f"   pooled_lodo_match {d['summary']['pooled_lodo_match']:.4f}")
rng = np.random.default_rng(42)
for nm, a, b in (("pooled_full - pooled_match", full, match),
                 ("pooled_lodo - pooled_lodo_match", lodo, lodo_m)):
    diff = a - b
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    bs = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"  {nm:<34} {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  two-sided p {p:.3f}")

# any stored significance for it
for k in sorted(d):
    if k not in ("summary", "per_dataset", "n_train"):
        v = d[k]
        print(f"  key {k}: {json.dumps(v)[:400] if not isinstance(v, (int, float, str)) else v}")

print("=" * 78)
print("mtimes")
for p in (f"{PREMISE}/clustered/clustered_L20_mean_s42.json",
          f"{PREMISE}/clustered/clustered_L32_mean_s42.json",
          f"{PREMISE}/clustered/clustered_L40_mean_s42.json",
          f"{PREMISE}/cluster_perm_kd_matched.json",
          f"{PREMISE}/premise_L40_mean_s42.json"):
    st = os.stat(p)
    import datetime
    print(f"  {st.st_size:>8}  "
          f"{datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}  {p}")
