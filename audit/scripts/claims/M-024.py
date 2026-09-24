# M-024: "16 scenarios could not move any conclusion (AUROC CI ~+-0.25)", ai_audit exclusion
import torch, collections, math

P = 'run-coords/ai_audit/items.pt'
d = torch.load(P, map_location='cpu', weights_only=False)
ids = [x[0] for x in d]
scen = collections.Counter(i.rsplit(':', 1)[0] for i in ids)
lens = sorted(len(x[1]) for x in d)
print(f'{P}: {len(d)} rows, {len(scen)} distinct scenarios, '
      f'{sorted(set(scen.values()))} samples each')
print(f'  token lengths: min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]}')
print('  (the note quotes 4,211-7,543 tokens, median 6,028, 80 rows = 16 scenarios, 5 samples each)')
H = 64
print('\n  eager attention at batch size 1 (rows x heads x T^2 x 4 bytes):')
for T in (lens[0], lens[len(lens)//2], lens[-1]):
    print(f'    T={T}: {H*T*T*4/2**30:.2f} GiB')
print('\n  AUROC 95% CI half-width (Hanley-McNeil), 8 deceptive vs 8 honest scenarios:')


def hm(A, n1, n2):
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n2 - 1) * (Q2 - A * A)) / (n1 * n2))


for A in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f'    AUROC {A:.1f}: clustered n1=n2=8 -> +-{1.96*hm(A,8,8):.3f}   '
          f'(rows as independent, n1=n2=40 -> +-{1.96*hm(A,40,40):.3f})')
