"""W6F2-memory: claims M-034, M-035 (memory/persona-axes-encode-register.md:25,32).

Source of truth: run-crossgen/kd11/matrices/mean_L40_s42.json (git working tree copy)
  bank = axes_vectors_s3_fp32.pt / bank_name = paper16, k=5, seed 42,
  11 datasets, 80% train / own-20% diagonal.
Cross-check against audit/phase2/layer_sensitivity.csv (store kd11, dataset_set 'all').
Read-only.
"""
import json, os, csv

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
M = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", "mean_L40_s42.json")
d = json.load(open(M))
DS = d['datasets']
print("file:", os.path.relpath(M, ROOT))
print("layer=%s pooling=%s seed=%s k=%s bank=%s bank_name=%s frac_train=%s"
      % (d['layer'], d['pooling'], d['seed'], d['k'], d['bank'], d['bank_name'], d['frac_train']))
print("datasets (%d): %s" % (len(DS), DS))

SYC = DS.index('sycophancy_mt')


def offdiag(mat, drop=()):
    keep = [i for i in range(len(DS)) if i not in drop]
    v = [mat[i][j] for i in keep for j in keep if i != j]
    return sum(v) / len(v), len(v)


print("\n=== M-034: off-diagonal mean with vs without sycophancy_mt (L40, mean, probe=paper) ===")
print("%-16s %8s %8s %8s" % ("feature", "11ds", "10ds", "lift"))
lifts = {}
for r in d['results']:
    if r['probe'] != 'paper':
        continue
    a11, n11 = offdiag(r['auroc'])
    a10, n10 = offdiag(r['auroc'], drop=(SYC,))
    lifts[r['feature']] = a10 - a11
    print("%-16s %8.4f %8.4f %+8.4f   (n=%d -> %d)" % (r['feature'], a11, a10, a10 - a11, n11, n10))
print("\nnote's numbers: raw +0.025, axes +0.061, pc:deception +0.069, 'about 2.5x more'")
print("ratio axes/raw       = %.2f" % (lifts['axes'] / lifts['raw']))
print("ratio pc:deception/raw = %.2f" % (lifts['pc:deception'] / lifts['raw']))

print("\n--- same, for the other two probes (robustness) ---")
for probe in ('mms', 'lr_reg10'):
    print(" probe=%s" % probe)
    for r in d['results']:
        if r['probe'] != probe or r['feature'] not in ('raw', 'axes', 'pc:deception', 'pc:unified'):
            continue
        a11, _ = offdiag(r['auroc'])
        a10, _ = offdiag(r['auroc'], drop=(SYC,))
        print("   %-16s %8.4f %8.4f %+8.4f" % (r['feature'], a11, a10, a10 - a11))

print("\n=== M-035: pc:unified vs raw across all 11 datasets, L40 mean, probe=paper ===")
for r in sorted([x for x in d['results'] if x['probe'] == 'paper'], key=lambda x: -offdiag(x['auroc'])[0]):
    a11, _ = offdiag(r['auroc'])
    print("  %-16s off-diag(11) = %.4f" % (r['feature'], a11))

print("\n--- cross-check vs audit/phase2/layer_sensitivity.csv (store kd11, L40, mean, paper, all) ---")
for row in csv.DictReader(open(os.path.join(ROOT, "audit", "phase2", "layer_sensitivity.csv"))):
    if (row['store'] == 'kd11' and row['layer'] == '40' and row['pooling'] == 'mean'
            and row['probe'] == 'paper' and row['dataset_set'] == 'all'):
        print("  %-16s %s (n_pairs=%s)" % (row['feature'], row['offdiag_mean'], row['n_pairs']))
