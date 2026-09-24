"""M-014 / M-015: memory note judge-choice-barely-matters.md (lines 3, 8-18)

Claims:
  - "Re-judged all 1,400 paper16 rollouts" with Llama-3.1-405B into run-paper16/scores_405b/
  - exact-score agreement 91.2% (1,231 of 1,350 comparable)
  - 70B distribution 0:12, 2:249, 3:1089 ; 405B 0:19, 1:2, 2:250, 3:1079
  - score-3 rate 80.7% vs 79.9%, a 0.8 pp difference
  - disagreement concentrates on handler 74%, plausible_denier 76%, fact_checker 78%
    (read as per-persona AGREEMENT rates)

Recomputed from the two score stores. Read-only.
"""
import json, os, sys, glob

root = sys.argv[1] if len(sys.argv) > 1 else 'run-paper16'
a_dir, b_dir = os.path.join(root, 'scores'), os.path.join(root, 'scores_405b')

A, B = {}, {}
for d, D in ((a_dir, A), (b_dir, B)):
    for p in sorted(glob.glob(os.path.join(d, '*.json'))):
        role = os.path.basename(p)[:-5]
        for k, v in json.load(open(p)).items():
            D[(role, k)] = v

print(f'70B store  {a_dir}: {len(glob.glob(os.path.join(a_dir,"*.json")))} files, {len(A)} scored items')
print(f'405B store {b_dir}: {len(glob.glob(os.path.join(b_dir,"*.json")))} files, {len(B)} scored items')
common = sorted(set(A) & set(B))
print(f'comparable (same role+item in both): {len(common)}')
print(f'only in 70B: {len(set(A)-set(B))}   only in 405B: {len(set(B)-set(A))}')

agree = sum(1 for k in common if A[k] == B[k])
print(f'exact-score agreement: {agree}/{len(common)} = {100*agree/len(common):.4f}%')


def dist(D, keys):
    out = {}
    for k in keys:
        out[D[k]] = out.get(D[k], 0) + 1
    return dict(sorted(out.items()))


print('70B  distribution over ALL its items :', dist(A, A))
print('405B distribution over ALL its items :', dist(B, B))
print('70B  distribution over COMMON items  :', dist(A, common))
print('405B distribution over COMMON items  :', dist(B, common))

for name, D, keys in (('70B all', A, list(A)), ('405B all', B, list(B)),
                      ('70B common', A, common), ('405B common', B, common)):
    n3 = sum(1 for k in keys if D[k] == 3)
    print(f'score-3 rate {name:12s}: {n3}/{len(keys)} = {100*n3/len(keys):.2f}%')

print('\nper-role agreement (lowest 6):')
roles = {}
for k in common:
    r = k[0]
    roles.setdefault(r, [0, 0])
    roles[r][1] += 1
    if A[k] == B[k]:
        roles[r][0] += 1
for r, (ok, n) in sorted(roles.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]:
    print(f'  {r:28s} {ok}/{n} = {100*ok/n:.1f}%')

# how many rollouts exist in total (the "1,400" claim)
resp = os.path.join(root, 'responses')
if os.path.isdir(resp):
    tot = 0
    nf = 0
    for p in sorted(glob.glob(os.path.join(resp, '*'))):
        nf += 1
        try:
            if p.endswith('.jsonl'):
                n = sum(1 for _ in open(p, encoding='utf-8'))
            else:
                o = json.load(open(p, encoding='utf-8'))
                n = len(o)
            tot += n
        except Exception as e:
            print('  skip', p, e)
    print(f'\nresponses/: {nf} files, {tot} rollout records')
