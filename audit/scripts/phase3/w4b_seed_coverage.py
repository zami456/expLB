"""W4b CONFLICT 2: per-(store, seed) feature/probe coverage in the crossgen results.
Source: audit/phase2/layer_sensitivity.csv (one row per store/bank/k/pooling/layer/seed/
        dataset_set/feature/probe with its off-diagonal mean).
Run: python audit/scripts/phase3/w4b_seed_coverage.py
"""
import csv, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
P = ROOT / 'audit/phase2/layer_sensitivity.csv'
rows = list(csv.DictReader(open(P, newline='', encoding='utf-8-sig')))
print("rows:", len(rows))
print("columns:", list(rows[0].keys()))
print("stores:", sorted(set(r['store'] for r in rows)))
print("seeds :", sorted(set(r['seed'] for r in rows)))

feat = collections.defaultdict(set); prob = collections.defaultdict(set)
lay  = collections.defaultdict(set); cnt = collections.Counter()
bnk  = collections.defaultdict(set); pool = collections.defaultdict(set)
for r in rows:
    k = (r['store'], r['seed'])
    feat[k].add(r['feature']); prob[k].add(r['probe']); lay[k].add(r['layer'])
    bnk[k].add(r['bank_name'] or r['bank']); pool[k].add(r['pooling']); cnt[k] += 1

print("\n=== per (store, seed): n_rows | n_features | n_probes | n_layers | n_banks | poolings ===")
for k in sorted(feat, key=lambda x: (x[0], x[1])):
    print(f"{k[0]:<12} seed={k[1]:<4} rows={cnt[k]:<7} feats={len(feat[k]):<3} probes={len(prob[k]):<2} "
          f"layers={len(lay[k]):<3} banks={len(bnk[k]):<2} pool={sorted(pool[k])}")

print("\n=== feature / probe lists per (store,seed) where feature count < max for that store ===")
maxf = collections.defaultdict(int)
for (s, sd), v in feat.items():
    maxf[s] = max(maxf[s], len(v))
for k in sorted(feat, key=lambda x: (x[0], x[1])):
    s = k[0]
    if len(feat[k]) < maxf[s]:
        print(f"{s} seed={k[1]}: features={sorted(feat[k])} probes={sorted(prob[k])}")

print("\n=== stores where ALL seeds have identical feature sets (no asymmetry) ===")
for s in sorted(set(r['store'] for r in rows)):
    sets = {sd: frozenset(v) for (st, sd), v in feat.items() if st == s}
    psets = {sd: frozenset(v) for (st, sd), v in prob.items() if st == s}
    same = len(set(sets.values())) == 1 and len(set(psets.values())) == 1
    print(f"{s:<12} seeds={sorted(sets)}  uniform_features={same}  "
          f"n_feat_by_seed={ {sd: len(v) for sd, v in sorted(sets.items())} }  "
          f"n_probe_by_seed={ {sd: len(v) for sd, v in sorted(psets.items())} }")
