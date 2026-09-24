# W6E-memory-q : Q10 claims M-046..M-049
# Source note: memory/q10-sycophancy-axis-inverted.md
import json, os
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
p = os.path.join(ROOT, 'run-coords/q10_sycophancy_axis.json')
d = json.load(open(p))
print('FILE', p)
print('n =', d['n'])
print()
print('=== q10_sycophancy_axis.json : key_layers table (paper16 & own100) ===')
for bank in ['paper16|mean', 'paper16|last', 'own100|mean', 'own100|last']:
    for direc in ['sycophancy', 'deception', 'unified', 'assistant', 'syc_PC1']:
        v = d['results'][bank].get(direc)
        if not v: continue
        for L in ['40', '44']:
            kl = v['key_layers'].get(L)
            if not kl: continue
            print(f'{bank:15s} {direc:11s} L{L:>2s}  auroc={kl["auroc"]:.4f}  ci95=[{kl["ci95"][0]:.4f},{kl["ci95"][1]:.4f}]  within_len_quint={kl["within_length_quintiles"]:.4f}  len_regressed={kl["length_regressed_out"]:.4f}')
print()
print('random (q10 file):', json.dumps(d['random'], indent=1))
print('length_only:', d['length_only'])
print()
p2 = os.path.join(ROOT, 'run-coords/zeroshot_nulls.json')
z = json.load(open(p2))
print('FILE', p2)
print('=== zeroshot_nulls.json : sycophancy_mt ===')
for k, v in z['sycophancy_mt'].items():
    print(f'-- {k}: random_p95_auroc_dev={v["random_p95_auroc_dev"]:.4f} (|AUROC-0.5| = {abs(v["random_p95_auroc_dev"]-0.5):.4f}), '
          f'random_max_dev={v["random_max_auroc_dev"]:.4f} (|dev|={abs(v["random_max_auroc_dev"]-0.5):.4f}), n_random={v["n_random"]}, '
          f'cos(syc,assistant)={v["cos_sycophancy_assistant"]:.4f}')
    for direc in ['sycophancy_axis', 'assistant_axis', 'sycophancy_minus_assistant', 'deception_axis']:
        a = v[direc]
        print(f'     {direc:28s} auroc={a["auroc"]:.4f} p_vs_random={a["p_vs_random"]:.4f} beats_p95={a["beats_random_p95"]}')

print()
p3 = os.path.join(ROOT, 'audit/cache/labpc/run-coords/confounds_L44_last.json')
c = json.load(open(p3))
print('FILE', p3, '(lab-PC copy cached by the audit)')
print('layer', c['layer'], 'pooling', c['pooling'], 'k', c['k'], 'bank', c['bank'])
print('=== length_only_auroc + median response length per dataset ===')
for r in c['confounds']['length_only_auroc']:
    print(f'  {r["dataset"]:32s} auroc_from_length={r["auroc_from_length"]:.4f}  median_pos={r["median_pos"]}  median_neg={r["median_neg"]}')
