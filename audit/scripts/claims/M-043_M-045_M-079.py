"""W6F2-memory: claims M-043, M-044, M-045 (persona-set-decides-detection.md:24-29)
and M-079 (regrouping-cannot-replace-personas.md:3).

Raw files (git working tree):
  run-coords/bank_run_axes.pt.json                        (own100 bank, L20, mean, 5 datasets)
  run-coords/bank_run-paper16_axes_vectors_s3.pt.json     (paper16 s3 bank, same rows)
  run-coords/regroup_hf.json                              (HF role library, our vs disclosure grouping)
  run-paper16/axes_vectors_s3.pt                          (for the paper16 cosine column)
Read-only.
"""
import json, os, math, torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RC = os.path.join(ROOT, "run-coords")

own = json.load(open(os.path.join(RC, "bank_run_axes.pt.json")))
p16 = json.load(open(os.path.join(RC, "bank_run-paper16_axes_vectors_s3.pt.json")))

print("=" * 78)
print("M-043 / M-044 / M-045 -- own100 vs paper16 detection, layer 20")
print("=" * 78)
for tag, d in (("own100", own), ("paper16", p16)):
    print("%-8s layer=%s pooling=%s k=%s bank=%s datasets=%s"
          % (tag, d['layer'], d['pooling'], d['k'], d['bank'], list(d['datasets'])))

LAB = list(own['datasets'])
print("\nlabel_semantics: %s" % {k: v['label_semantics'] for k, v in own['splits'].items()})


def key(r):
    return (r['features'], r['probe'])


A = {key(r): r for r in own['p1']}
B = {key(r): r for r in p16['p1']}
print("\n%-14s %-9s %8s %8s %9s   (note)" % ("features", "probe", "own100", "paper16", "gain"))
NOTE = {('pc:deception', 'paper'): (0.706, 0.815), ('pc:deception', 'lr_reg10'): (0.698, 0.787),
        ('axes', 'paper'): (0.708, 0.781), ('axes', 'lr_reg10'): (0.683, 0.785),
        ('axes_resid', 'paper'): (0.553, 0.770), ('axes_resid', 'lr_reg10'): (0.554, 0.776)}
gains = []
for k in NOTE:
    if k not in A or k not in B:
        print("%-14s %-9s MISSING" % k)
        continue
    a, b = A[k]['off_diagonal'], B[k]['off_diagonal']
    gains.append(b - a)
    print("%-14s %-9s %8.4f %8.4f %+9.4f   (note %.3f / %.3f)"
          % (k[0], k[1], a, b, b - a, NOTE[k][0], NOTE[k][1]))
print("\nn_pairs: own100 %s, paper16 %s (off-diagonal cells used)"
      % (A[('axes', 'paper')]['n_pairs'], B[('axes', 'paper')]['n_pairs']))
print("M-044: gain range over the 4 non-resid rows = %+.4f .. %+.4f  (note says +0.08 to +0.10)"
      % (min(gains[:4]), max(gains[:4])))
print("M-045: paper16 axes_resid vs axes (paper probe) = %.4f vs %.4f, cost %+.4f"
      % (B[('axes_resid', 'paper')]['off_diagonal'], B[('axes', 'paper')]['off_diagonal'],
         B[('axes_resid', 'paper')]['off_diagonal'] - B[('axes', 'paper')]['off_diagonal']))
print("       own100  axes_resid vs axes (paper probe) = %.4f vs %.4f, cost %+.4f"
      % (A[('axes_resid', 'paper')]['off_diagonal'], A[('axes', 'paper')]['off_diagonal'],
         A[('axes_resid', 'paper')]['off_diagonal'] - A[('axes', 'paper')]['off_diagonal']))

print("\nAll rows present in both files (for completeness):")
for k in sorted(set(A) & set(B)):
    print("   %-16s %-9s own100=%.4f paper16=%.4f  diff=%+.4f"
          % (k[0], k[1], A[k]['off_diagonal'], B[k]['off_diagonal'],
             B[k]['off_diagonal'] - A[k]['off_diagonal']))

# --------------------------------------------------------------------- M-079
print()
print("=" * 78)
print("M-079 -- regrouping the HF library by disclosure stance")
print("=" * 78)
rg = json.load(open(os.path.join(RC, "regroup_hf.json")))
print("conceal (n=%d): %s" % (len(rg['conceal']), rg['conceal']))
print("disclose (n=%d): %s" % (len(rg['disclose']), rg['disclose']))
print("ours_dec (n=%d) ours_hon (n=%d)" % (len(rg['ours_dec']), len(rg['ours_hon'])))
p16b = torch.load(os.path.join(ROOT, "run-paper16", "axes_vectors_s3.pt"),
                  map_location='cpu', weights_only=False)


def cos(a, c):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)))


print("\n%-6s %10s %12s %10s" % ("layer", "ours", "regrouped", "paper16 s3"))
ours, reg, pap = [], [], []
for r in rg['results']:
    L = r['layer']
    i = p16b['layers'].index(L)
    c = cos(p16b['axes']['deception'][i], p16b['axes']['assistant'][i])
    ours.append(r['ours']); reg.append(r['disclosure']); pap.append(c)
    print("%-6d %10.4f %12.4f %10.4f" % (L, r['ours'], r['disclosure'], c))
mo = sum(abs(x) for x in ours) / len(ours)
mr = sum(abs(x) for x in reg) / len(reg)
mp = sum(abs(x) for x in pap) / len(pap)
print("%-6s %10.4f %12.4f %10.4f   <- mean |cos|" % ("mean", mo, mr, mp))
print("\nreduction from regrouping        = (%.4f - %.4f)/%.4f = %.1f%%   (note: 'buys 11%%')"
      % (mo, mr, mo, 100 * (mo - mr) / mo))
print("reduction from paper16 personas  = (%.4f - %.4f)/%.4f = %.1f%%" % (mo, mp, mo, 100 * (mo - mp) / mo))
print("share of the ours->paper16 gap recovered by regrouping = %.1f%%"
      % (100 * (mo - mr) / (mo - mp)))
print("per-layer regrouping reduction: %s"
      % ["%.1f%%" % (100 * (abs(o) - abs(r)) / abs(o)) for o, r in zip(ours, reg)])
