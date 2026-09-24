"""P3-C: for each store/feature, how does the REPORTED layer rank among all stored layers?
Metric: off-diagonal transfer mean minus the store's own raw baseline, seed 42, mean pooling,
paper probe, 11-dataset set where available. Read-only; input audit/phase2/layer_sensitivity.csv
"""
import csv, collections
ROWS = list(csv.DictReader(open('audit/phase2/layer_sensitivity.csv', encoding='utf-8-sig')))
SEED, POOL, PROBE = '42', 'mean', 'paper'
by = collections.defaultdict(dict)          # (store, dataset_set, feature) -> layer -> val
for r in ROWS:
    if r['seed'] == SEED and r['pooling'] == POOL and r['probe'] == PROBE:
        by[(r['store'], r['dataset_set'], r['feature'])][int(r['layer'])] = float(r['offdiag_mean'])
out = []
for (store, ds, feat), d in sorted(by.items()):
    if feat in ('raw', 'datapca', 'random', 'length') or feat.endswith('@2') is None:
        continue
    raw = by.get((store, ds, 'raw'), {})
    common = sorted(set(d) & set(raw))
    if len(common) < 5:
        continue
    gap = {L: d[L] - raw[L] for L in common}
    order = sorted(common, key=lambda L: -gap[L])
    mean = sum(gap.values()) / len(gap)
    for rep in (20, 40, 44):
        if rep in gap:
            out.append((store, ds, feat, len(common), rep, gap[rep], order.index(rep)+1,
                        mean, gap[order[0]], order[0]))
print(f"{'store':26s} {'ds':7s} {'feature':22s} {'nL':3s} {'L':3s} {'gap@L':>8s} "
      f"{'rank':>5s} {'mean':>8s} {'best':>8s} {'bestL':>5s}")
for o in out:
    if o[4] != 40:   # print L40 rows first below; keep all
        pass
for o in sorted(out, key=lambda x: (x[0], x[1], x[2], x[4])):
    print(f"{o[0]:26s} {o[1]:7s} {o[2]:22s} {o[3]:3d} {o[4]:3d} {o[5]:+8.4f} "
          f"{o[6]:3d}/{o[3]:<2d} {o[7]:+8.4f} {o[8]:+8.4f} {o[9]:5d}")
