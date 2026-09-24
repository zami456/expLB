"""W6F2-memory: claims M-080, M-081 (length-only AUROCs), M-083, M-084
(results-chapter-audit: Q11 0.853 / sycophancy_mt 0.15 / gender-secret 0.997 /
Q5 pc:deception 0.691) and M-093 (Pass A vs Pass B layer sweeps).

Raw files:
  audit/cache/labpc/run-coords/confounds_L44_last.json  (lab-only, md5 11bc6e65..., MANIFEST MATCH)
  run-coords/q11_sleeper_L44_last.json                  (git == lab, md5 dd99eba1...)
  run-coords/q5_semantics_L44_last.json                 (git == lab, md5 5b7c206a...)
  run-coords/layer_sweep_mms.json      = Pass A, mean pooling, mms probe (git == lab)
  run-coords/layer_sweep_passB_mms.json = Pass B, mean pooling, mms probe (git == lab)
Read-only.
"""
import json, os, statistics as st

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RC = os.path.join(ROOT, "run-coords")

print("=" * 78)
print("M-080 / M-081 -- length-only AUROC per dataset (L44, last token)")
print("=" * 78)
c = json.load(open(os.path.join(ROOT, "audit", "cache", "labpc", "run-coords",
                                "confounds_L44_last.json")))
print("file layer=%s pooling=%s bank=%s ; %d datasets carry rows, length control ran on %d"
      % (c['layer'], c['pooling'], c['bank'], len(c['datasets']),
         len(c['confounds']['length_only_auroc'])))
rows = c['confounds']['length_only_auroc']
print("datasets WITH rows but WITHOUT a length row: %s"
      % sorted(set(c['datasets']) - {r['dataset'] for r in rows}))
print("\n%-34s %8s %10s %10s" % ("dataset", "AUROC", "median+", "median-"))
for r in sorted(rows, key=lambda r: -r['auroc_from_length']):
    tag = "  <== M-080" if r['dataset'] == 'harm-pressure-knowledge-report' else (
        "  <== M-081" if r['dataset'] == 'instructed-deception' else "")
    print("%-34s %8.4f %10.1f %10.1f%s"
          % (r['dataset'], r['auroc_from_length'], r['median_pos'], r['median_neg'], tag))
above = [r for r in rows if r['auroc_from_length'] > 0.5]
print("\nabove 0.5 on %d of %d ; inversions: %s"
      % (len(above), len(rows), [r['dataset'] for r in rows if r['auroc_from_length'] <= 0.5]))
print("label-shuffle null: mean %.4f sd %.4f n=%d" % (c['null']['mean'], c['null']['sd'], c['null']['n']))

print()
print("=" * 78)
print("M-083 -- Q11 'trained on the corpus, applied to sleepers' = 0.853, and the 0.15 source")
print("=" * 78)
q = json.load(open(os.path.join(RC, "q11_sleeper_L44_last.json")))
print("layer=%s pooling=%s probe=%s bank=%s" % (q['layer'], q['pooling'], q['probe'], q['bank']))
cs = q['tests']['corpus_to_sleeper']
srcs = sorted({k.split('|')[1] for k in cs})
ST = ['soft-trigger-gender', 'soft-trigger-greeting', 'soft-trigger-time']
for f in ('raw', 'axes'):
    allv = [cs[f'{f}|{s}'][t] for s in srcs for t in ST]
    nov = [cs[f'{f}|{s}'][t] for s in srcs if s != 'sycophancy_mt' for t in ST]
    print("%-5s mean over %d sources x 3 soft-triggers = %.4f ; excluding sycophancy_mt (%d sources) = %.4f"
          % (f, len(srcs), st.mean(allv), len(srcs) - 1, st.mean(nov)))
    sv = [cs[f'{f}|sycophancy_mt'][t] for t in ST]
    print("      sycophancy_mt as source: %s  mean %.4f" % ([round(x, 4) for x in sv], st.mean(sv)))

print()
print("=" * 78)
print("M-084 -- gender-secret within-organism, and Q5 mask <-> sycophancy_mt")
print("=" * 78)
print("within_organism block of q11_sleeper_L44_last.json:")
for k, v in q['tests']['within_organism'].items():
    print("   %-28s %.4f" % (k, v))
q5 = json.load(open(os.path.join(RC, "q5_semantics_L44_last.json")))
print("\nq5_semantics_L44_last.json source = %s" % q5['source'])
print("mask <-> sycophancy_mt (mean of the two directed cells), probe=mms:")
for r in q5['results']:
    if r['probe'] != 'mms':
        continue
    cp = r['cross_provenance_same_construct']
    a, b = cp['mask->sycophancy_mt'], cp['sycophancy_mt->mask']
    print("   %-14s mask->syc=%.4f syc->mask=%.4f  mean=%.4f" % (r['features'], a, b, (a + b) / 2))

print()
print("=" * 78)
print("M-093 -- Pass A vs Pass B layer sweeps (mean pooling, mms probe)")
print("=" * 78)
for tag, fn in (("Pass A", "layer_sweep_mms.json"), ("Pass B", "layer_sweep_passB_mms.json")):
    d = json.load(open(os.path.join(RC, fn)))
    s = d['sweep']
    print("\n%s  file=%s pooling=%s probe=%s bank=%s  (%d layers)"
          % (tag, fn, d['pooling'], d['probe'], d['bank'], len(s)))
    for feat in ('raw', 'axes', 'pc:deception'):
        b = max(s, key=lambda r: r.get(feat, -1))
        print("   %-13s peak %.4f at L%d" % (feat, b[feat], b['layer']))
    w = [r['layer'] for r in s if r.get('axes', 0) > r.get('raw', 1)]
    print("   layers where axes > raw: %s (%d of %d)" % (w, len(w), len(s)))
