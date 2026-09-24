"""W6F2-memory: claims M-085, M-086, M-088, M-089, M-090, M-091, M-092.

Source note: memory/significance-and-random-nulls.md
Raw files (git working tree, laptop copy):
  run-coords/significance.json          (mtime 2026-09-17 01:03)
  run-coords/zeroshot_nulls.json        (mtime 2026-09-17 01:06)
  run-coords/p1_L44_last_withsyc.json   (L44 last, bank axes_vectors_s3_fp32.pt)
  run-coords/FULL_L40.json              (L40 mean, bank axes_vectors_s3.pt  <-- OLD bank, RF-15)
Read-only.
"""
import json, math, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RC = os.path.join(ROOT, "run-coords")

D9 = ['ai_liar', 'convincing-game', 'harm-pressure-choice',
      'harm-pressure-knowledge-report', 'insider-trading', 'instructed-deception',
      'mask', 'roleplaying', 'sandbagging_v2']


def load(p):
    return json.load(open(os.path.join(RC, p)))


sig = load("significance.json")

print("=" * 78)
print("M-085 / M-088 / M-089 / M-092 -- run-coords/significance.json")
print("=" * 78)
print("datasets (n=%d): %s" % (len(sig['datasets']), sig['datasets']))
print("boot_rows=%s boot_datasets=%s margin=%s reproduction_worst_diff=%.3g"
      % (sig['boot_rows'], sig['boot_datasets'], sig['margin'], sig['reproduction_worst_diff']))

print("\n-- M-086: named Q2 comparisons")
want = ["axes vs raw, same layer 40 (mean)", "axes vs raw, same layer 44 (mean)",
        "axes vs raw, same layer 44 (last)",
        "axes@44 vs raw@32, each at its best (mean)",
        "axes@44 vs raw@28, each at its best (last)"]
for c in sig['comparisons']:
    if c['name'] in want:
        print("  %-45s diff=%+.4f rowCI95=[%+.4f,%+.4f] row_p=%s" %
              (c['name'], c['diff'], c['row_ci95'][0], c['row_ci95'][1], c['row_p']))

print("\n-- M-088: does ANY Q2/Q6 difference survive the dataset bootstrap at BH q<0.05?")
n_sig = 0
for c in sig['comparisons']:
    if c.get('question') in ('Q2', 'Q6') and c.get('dataset_q_bh') is not None:
        flag = c.get('significant')
        n_sig += bool(flag)
        print("  %-45s q_bh=%.4f significant=%s" % (c['name'], c['dataset_q_bh'], flag))
print("  -> number significant at BH q<0.05: %d" % n_sig)

print("\n-- M-088: per-method dataset CI95 spans")
spans = []
for r in sig['per_method']:
    s = r['dataset_ci95'][1] - r['dataset_ci95'][0]
    spans.append(s)
    tag = " <== note's example" if (r['pooling'] == 'mean' and r['feature'] == 'axes' and r['layer'] == 44) else ""
    print("  %-5s %-13s L%-3d mean=%.4f dsCI95=[%.4f,%.4f] span=%.3f%s"
          % (r['pooling'], r['feature'], r['layer'], r['mean'],
             r['dataset_ci95'][0], r['dataset_ci95'][1], s, tag))
print("  span min=%.3f max=%.3f mean=%.3f median=%.3f"
      % (min(spans), max(spans), sum(spans) / len(spans), sorted(spans)[len(spans) // 2]))

print("\n-- M-089: the 'closest' Q6 comparison")
for c in sig['comparisons']:
    if c['name'] == 'axes: layer 40 vs 22 (mean)':
        print("  " + json.dumps(c))

print("\n-- M-092: the four Q6 '20 vs 22' equivalence tests (margin=%s)" % sig['margin'])
lo90, hi90 = [], []
for c in sig['comparisons']:
    if 'layer 20 vs 22' in c['name']:
        print("  %-30s diff=%+.4f rowCI95=[%+.4f,%+.4f] rowCI90=[%+.4f,%+.4f] row_equiv=%s"
              % (c['name'], c['diff'], c['row_ci95'][0], c['row_ci95'][1],
                 c['row_ci90'][0], c['row_ci90'][1], c['row_equivalent_within_margin']))
        print("  %-30s dsCI95=[%+.4f,%+.4f] dsCI90=[%+.4f,%+.4f] ds_equiv=%s q_bh=%s"
              % ("", c['dataset_ci95'][0], c['dataset_ci95'][1],
                 c['dataset_ci90'][0], c['dataset_ci90'][1],
                 c['equivalent_within_margin'], c.get('dataset_q_bh')))
        lo90.append(c['dataset_ci90'][0]); hi90.append(c['dataset_ci90'][1])
print("  -> dataset CI90 envelope over the 4 tests: [%+.4f, %+.4f]" % (min(lo90), max(hi90)))

# ---------------------------------------------------------------- M-090 / M-091
print()
print("=" * 78)
print("M-090 / M-091 -- pairwise win rate and per-dataset drivers, axes vs raw")
print("=" * 78)


def matrix(fn, feat, probe):
    d = load(fn)
    for r in d['p1']:
        if r['features'] == feat and r['probe'] == probe:
            return d, r
    raise KeyError((fn, feat, probe))


def offdiag_pairs(m):
    out = {}
    for a in D9:
        for b in D9:
            if a == b:
                continue
            x = m.get(f'{a}->{b}')
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            out[(a, b)] = x
    return out


for label, fn, probe in [("L44 last", "p1_L44_last_withsyc.json", "mms"),
                         ("L40 mean", "FULL_L40.json", "mms")]:
    da, ra = matrix(fn, 'axes', probe)
    db, rb = matrix(fn, 'raw', probe)
    A = offdiag_pairs(ra['matrix']); B = offdiag_pairs(rb['matrix'])
    keys = sorted(set(A) & set(B))
    wins = sum(1 for k in keys if A[k] > B[k])
    print("\n%s  file=%s bank=%s probe=%s  (9 labelled datasets)"
          % (label, fn, da['bank'], probe))
    print("  off-diag mean axes=%.4f raw=%.4f  diff=%+.4f  (sig.json per_method: see above)"
          % (sum(A[k] for k in keys) / len(keys),
             sum(B[k] for k in keys) / len(keys),
             sum(A[k] - B[k] for k in keys) / len(keys)))
    print("  PAIRWISE WIN RATE axes>raw: %d/%d = %.1f%%" % (wins, len(keys), 100 * wins / len(keys)))
    print("  per-dataset AS SOURCE (train set), mean over its 8 targets:")
    for s in D9:
        ks = [k for k in keys if k[0] == s]
        da_ = sum(A[k] - B[k] for k in ks) / len(ks)
        print("    %-33s axes=%.4f raw=%.4f delta=%+.4f  wins=%d/%d"
              % (s, sum(A[k] for k in ks) / len(ks), sum(B[k] for k in ks) / len(ks),
                 da_, sum(1 for k in ks if A[k] > B[k]), len(ks)))
    print("  per-dataset AS TARGET (test set), mean over its 8 sources:")
    for t in D9:
        ks = [k for k in keys if k[1] == t]
        da_ = sum(A[k] - B[k] for k in ks) / len(ks)
        print("    %-33s axes=%.4f raw=%.4f delta=%+.4f  wins=%d/%d"
              % (t, sum(A[k] for k in ks) / len(ks), sum(B[k] for k in ks) / len(ks),
                 da_, sum(1 for k in ks if A[k] > B[k]), len(ks)))

# ---------------------------------------------------------------- M-085 zero-shot
print()
print("=" * 78)
print("M-085 -- zero-shot AUROCs vs the 1000-draw random null (zeroshot_nulls.json)")
print("=" * 78)
zn = load("zeroshot_nulls.json")
for blk in ("corpus_L40_mean", "corpus_L44_last"):
    b = zn[blk]
    p95 = b['pooled']['random_p95_auroc_dev']
    print("  %s pooled: p95=%.4f max=%.4f n=%d" % (blk, p95, b['pooled']['random_max_auroc_dev'],
                                                   b['pooled']['n_random']))
    for k, v in b['pooled'].items():
        if isinstance(v, dict):
            print("     %-16s auroc=%.4f p=%.4f beats_p95=%s" % (k, v['auroc'], v['p_vs_random'],
                                                                 v['beats_random_p95']))
    for ds, vv in b['per_dataset'].items():
        d_ = vv['deception_axis']
        print("     %-32s auroc=%.4f  null p95=%.4f max=%.4f  p=%.4f  beats=%s"
              % (ds, d_['auroc'], vv['random_p95_auroc_dev'], vv['random_max_auroc_dev'],
                 d_['p_vs_random'], d_['beats_random_p95']))
print("  sleeper_L44_last (Q11):")
for ds, vv in zn['sleeper_L44_last'].items():
    d_ = vv['deception_axis']
    print("     %-24s auroc=%.4f  null p95=%.4f  p=%.4f  beats=%s"
          % (ds, d_['auroc'], vv['random_p95_auroc_dev'], d_['p_vs_random'], d_['beats_random_p95']))
print("  sycophancy_mt (L40 mean cell):")
for k, v in zn['sycophancy_mt']['L40_mean'].items():
    if isinstance(v, dict):
        print("     %-28s auroc=%.4f p=%.4f beats=%s" % (k, v['auroc'], v['p_vs_random'],
                                                         v['beats_random_p95']))
print("     random p95=%.4f max=%.4f" % (zn['sycophancy_mt']['L40_mean']['random_p95_auroc_dev'],
                                         zn['sycophancy_mt']['L40_mean']['random_max_auroc_dev']))
