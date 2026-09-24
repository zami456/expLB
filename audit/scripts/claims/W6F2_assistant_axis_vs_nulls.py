"""W6F2-memory, side question flagged in audit/phase3/CONTROLS.md:
"the assistant axis's 0.710 on sycophancy_mt may fall below the null after a length control,
 and the best-of-layers p95 is 0.694, not 0.634."

This script settles the BEST-OF-LAYERS half from stored data and states exactly what is
missing for the LENGTH half.

Raw files (git == lab, md5-checked 2026-09-23):
  run-coords/q10_sycophancy_axis.json   md5 6bc5190b714c8fe75ca3fa384dbe726d
  run-coords/zeroshot_nulls.json        md5 0c241a653587a570ed37a9bffe578dcd
  audit/cache/labpc/run-coords/confounds_L44_last.json (lab-only)
Read-only.
"""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RC = os.path.join(ROOT, "run-coords")
q = json.load(open(os.path.join(RC, "q10_sycophancy_axis.json")))
zn = json.load(open(os.path.join(RC, "zeroshot_nulls.json")))

print("rows: %s" % q['n'])
print("length_only AUROC on sycophancy_mt (q10 file)         = %.4f" % q['length_only'])
c = json.load(open(os.path.join(ROOT, "audit", "cache", "labpc", "run-coords",
                                "confounds_L44_last.json")))
lo = [r for r in c['confounds']['length_only_auroc'] if r['dataset'] == 'sycophancy_mt'][0]
print("length_only AUROC on sycophancy_mt (confounds file)   = %.4f (median %s vs %s tokens)"
      % (lo['auroc_from_length'], lo['median_pos'], lo['median_neg']))

print("\n--- the three nulls, side by side ---")
for cell in ('L40_mean', 'L40_last', 'L44_mean', 'L44_last'):
    b = zn['sycophancy_mt'][cell]
    print("  per-cell %s: random p95 = %.4f, max of 1000 = %.4f"
          % (cell, b['random_p95_auroc_dev'], b['random_max_auroc_dev']))
for pool in ('mean', 'last'):
    r = q['random'][pool]
    print("  best-of-layers (%s pooling): per-layer p95 = %.4f, BEST-OF-LAYERS p95 = %.4f, max = %.4f"
          % (pool, r['per_layer_abs_dev_p95'], r['best_of_layers_p95'], r['best_of_layers_max']))

print("\n--- the assistant axis against each null (oriented = max(a, 1-a)) ---")
for pool in ('mean', 'last'):
    r = q['random'][pool]
    for bank in ('paper16', 'own100'):
        bl = q['results'][f'{bank}|{pool}']['assistant']['by_layer']
        ori = {int(k): max(v, 1 - v) for k, v in bl.items()}
        bestL, bestV = max(ori.items(), key=lambda kv: kv[1])
        print("  %-8s %-4s : L40=%.4f  L44=%.4f  best-of-layers=%.4f at L%d"
              % (bank, pool, ori.get(40, float('nan')), ori.get(44, float('nan')), bestV, bestL))
        print("           vs best-of-layers p95 %.4f -> %s ; vs best-of-layers max %.4f -> %s"
              % (r['best_of_layers_p95'], "CLEARS" if bestV > r['best_of_layers_p95'] else "no",
                 r['best_of_layers_max'], "CLEARS" if bestV > r['best_of_layers_max'] else "no"))

print("\n--- the L40/mean cell CONTROLS.md quotes, against all three nulls ---")
a40 = q['results']['paper16|mean']['assistant']['by_layer']['40']
print("  assistant axis L40 mean = %.4f" % a40)
print("  vs per-cell p95   0.6304 -> %s" % ("CLEARS" if a40 > 0.6304 else "no"))
print("  vs per-layer p95  %.4f -> %s" % (q['random']['mean']['per_layer_abs_dev_p95'],
                                          "CLEARS" if a40 > q['random']['mean']['per_layer_abs_dev_p95'] else "no"))
print("  vs best-of-layers p95 %.4f -> %s" % (q['random']['mean']['best_of_layers_p95'],
                                              "CLEARS" if a40 > q['random']['mean']['best_of_layers_p95'] else "no"))
print("  vs length-only    %.4f -> %s (margin %+.4f)"
      % (q['length_only'], "above" if a40 > q['length_only'] else "below", a40 - q['length_only']))

print("\n--- the sycophancy axis's inversion, for contrast ---")
for cell in ('L40_mean', 'L40_last', 'L44_mean', 'L44_last'):
    b = zn['sycophancy_mt'][cell]['sycophancy_axis']
    print("  %s auroc=%.4f (oriented %.4f) p=%.4f" % (cell, b['auroc'], max(b['auroc'], 1 - b['auroc']),
                                                      b['p_vs_random']))

print("""
WHAT IS NOT SETTLED: a LENGTH-CONTROLLED assistant-axis reading. That needs the per-row
assistant-axis projections on sycophancy_mt together with each row's token count (to
residualise length out, or to score within length strata). No stored file in run-coords,
audit/cache or run-crossgen holds per-row projections -- q10_sycophancy_axis.json and
zeroshot_nulls.json store only AUROCs, and dotproducts_mean.json stores per-layer AUROCs.
Computing it requires the pooled sycophancy_mt activations (lab, run-coords/passB or passC).
""")
