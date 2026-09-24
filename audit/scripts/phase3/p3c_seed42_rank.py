"""P3-C: is seed 42 (the seed every headline uses) systematically the most favourable seed?
For every cell with >=5 seeds, rank seed 42's `feature - raw` gap among the seeds present.
Under no selection the expected mean rank is (n+1)/2 and P(rank 1) = 1/n.
Read-only; input audit/phase2/layer_sensitivity.csv"""
import csv, collections, statistics as st
ROWS = list(csv.DictReader(open('audit/phase2/layer_sensitivity.csv', encoding='utf-8-sig')))
V = collections.defaultdict(dict)
N = {}
for r in ROWS:
    k = (r['store'], r['dataset_set'], r['pooling'], r['probe'], int(r['layer']), r['feature'])
    V[k][r['seed']] = float(r['offdiag_mean'])
    N[k] = (r['n_pairs'], r['diag_mean'])
res = collections.defaultdict(list)
for k, d in V.items():
    store, ds, pool, probe, L, feat = k
    if feat == 'raw':
        continue
    rawk = (store, ds, pool, probe, L, 'raw')
    raw = V.get(rawk, {})
    seeds = sorted(set(d) & set(raw))
    if len(seeds) < 5 or '42' not in seeds:
        continue
    gap = {s: d[s] - raw[s] for s in seeds}
    order = sorted(seeds, key=lambda s: -gap[s])
    res[feat].append((order.index('42') + 1, len(seeds)))
tot = []
for feat, v in sorted(res.items()):
    ranks = [a for a, _ in v]; n = v[0][1]
    print(f"{feat:14s} cells={len(v):4d} n_seeds={n}  mean rank of seed42 = {st.mean(ranks):.2f} "
          f"(chance {(n+1)/2:.1f})  rank1 share = {sum(1 for r in ranks if r==1)/len(ranks):.3f} "
          f"(chance {1/n:.3f})  rank<=2 share = {sum(1 for r in ranks if r<=2)/len(ranks):.3f}")
    tot += ranks
print(f"\nALL persona+control features: cells={len(tot)}  mean rank {st.mean(tot):.2f} "
      f"(chance 3.5)  rank1 share {sum(1 for r in tot if r==1)/len(tot):.3f} (chance 0.167)")
# structural check: does the seed-42 run differ in n_pairs / diag_mean?
print("\n--- structural check: n_pairs and diag_mean by seed, root/all/mean/mms/axes ---")
for L in (20, 40, 44):
    k = ('<root>', 'all', 'mean', 'mms', L, 'axes')
    rows = [r for r in ROWS if (r['store'], r['dataset_set'], r['pooling'], r['probe'],
                                int(r['layer']), r['feature']) == k]
    print(f"  L{L}: " + "  ".join(f"s{r['seed']}:pairs={r['n_pairs']},diag={float(r['diag_mean']):.4f}"
                                  for r in sorted(rows, key=lambda x: int(x['seed']))))
