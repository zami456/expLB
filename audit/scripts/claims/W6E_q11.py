# W6E-memory-q : Q11 claims M-050..M-053  (memory/q11-sleeper-detection-is-falsity.md)
import json, os, statistics
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
def J(p): return json.load(open(os.path.join(ROOT, p)))
for f in ('run-coords/q11_sleeper_L44_last.json', 'run-coords/q11_sleeper_L20_last.json'):
    d = J(f)
    print('#' * 20, f)
    print('layer', d['layer'], 'pooling', d['pooling'], 'probe', d['probe'], 'bank', d['bank'])
    t = d['tests']
    print('-- within_organism:', json.dumps({k: round(v, 4) for k, v in t['within_organism'].items()}))
    print('-- cross_trigger  :', json.dumps({k: round(v, 4) for k, v in t['cross_trigger'].items()}))
    st = [k for k in t['cross_trigger'] if k.startswith('raw|soft-trigger')]
    print('   raw cross-trigger mean over 3 organisms =',
          round(statistics.mean(t['cross_trigger'][k] for k in st), 4))
    sa = [k for k in t['cross_trigger'] if k.startswith('axes|soft-trigger')]
    print('   axes cross-trigger mean over 3 organisms =',
          round(statistics.mean(t['cross_trigger'][k] for k in sa), 4))
    wr = [k for k in t['within_organism'] if k.startswith('raw|soft-trigger')]
    print('   raw within-organism mean over 3 organisms =',
          round(statistics.mean(t['within_organism'][k] for k in wr), 4))
    wa = [k for k in t['within_organism'] if k.startswith('axes|soft-trigger')]
    print('   axes within-organism mean over 3 organisms =',
          round(statistics.mean(t['within_organism'][k] for k in wa), 4))
    print('-- zeroshot:', json.dumps({k: round(v, 4) for k, v in t['zeroshot'].items()}))
    print('-- token_stratified:')
    for k, v in t['token_stratified'].items():
        if isinstance(v, dict):
            print(f'   {k:45s} probe_from_ID={v["probe_from_instructed-deception"]:.4f} zeroshot={v["zeroshot"]:.4f} n={v["n"]} n_pos={v["n_pos"]}')
        else:
            print(f'   {k:45s} = {v}')
    # pooled answer-token identity over the three soft-trigger organisms
    ts = t['token_stratified']
    decT = decF = honT = honF = 0
    for org in ('soft-trigger-gender', 'soft-trigger-greeting', 'soft-trigger-time'):
        kT, kF = f'raw|{org}|True', f'raw|{org}|False'
        if kT in ts and isinstance(ts[kT], dict):
            decT += ts[kT]['n_pos']; honT += ts[kT]['n'] - ts[kT]['n_pos']
            decF += ts[kF]['n_pos']; honF += ts[kF]['n'] - ts[kF]['n_pos']
    if decT + decF:
        pT = decT / (decT + decF); qT = honT / (honT + honF)
        print(f'   POOLED over 3 organisms: deceptive say "True" {decT}/{decT+decF} = {pT:.4f}; '
              f'honest say "True" {honT}/{honT+honF} = {qT:.4f}; token-only AUROC = {((1-pT)+qT)/2:.4f}')
        rawwt = [v['probe_from_instructed-deception'] for k, v in ts.items() if k.startswith('raw|soft-trigger') and isinstance(v, dict)]
        zswt = [v['zeroshot'] for k, v in ts.items() if k.startswith('raw|soft-trigger') and isinstance(v, dict)]
        axwt = [v['probe_from_instructed-deception'] for k, v in ts.items() if k.startswith('axes|soft-trigger') and isinstance(v, dict)]
        print(f'   within-token raw probe range  = [{min(rawwt):.4f}, {max(rawwt):.4f}]')
        print(f'   within-token zero-shot range  = [{min(zswt):.4f}, {max(zswt):.4f}]')
        print(f'   within-token AXES probe range = [{min(axwt):.4f}, {max(axwt):.4f}]')
    # M-051 : answer-token identity
    print('-- token identity by organism (from token_stratified n / n_pos):')
    for org in ('soft-trigger-gender', 'soft-trigger-greeting', 'soft-trigger-time'):
        kT = f'raw|{org}|True'; kF = f'raw|{org}|False'
        if kT in t['token_stratified']:
            nT, pT = t['token_stratified'][kT]['n'], t['token_stratified'][kT]['n_pos']
            nF, pF = t['token_stratified'][kF]['n'], t['token_stratified'][kF]['n_pos']
            # n_pos = deceptive rows within that answer token
            dec = pT + pF; hon = (nT - pT) + (nF - pF)
            print(f'   {org}: says "True" -> deceptive {pT}/{dec} = {pT/dec:.3f} ; honest {nT-pT}/{hon} = {(nT-pT)/hon:.3f} '
                  f'(rows n={nT+nF})')
    if 'corpus_to_sleeper' in t:
        print('-- corpus_to_sleeper (selected):')
        for k in ('raw|sycophancy_mt', 'axes|mask'):
            if k in t['corpus_to_sleeper']:
                print(f'   {k}: ' + json.dumps({kk: round(vv, 4) for kk, vv in t['corpus_to_sleeper'][k].items()}))
        # "trained on on-policy corpus (no triggers anywhere)" = mean over the 9/10 corpus probes?
        for feat in ('raw', 'axes'):
            vals = {}
            for k, v in t['corpus_to_sleeper'].items():
                if not k.startswith(feat + '|'): continue
                vals[k] = statistics.mean(v[o] for o in ('soft-trigger-gender', 'soft-trigger-greeting', 'soft-trigger-time'))
            print(f'   {feat}: per-source mean over 3 organisms = ' + json.dumps({k: round(v, 4) for k, v in vals.items()}))
            print(f'   {feat}: MEAN over all sources = {statistics.mean(vals.values()):.4f}; '
                  f'mean excluding sycophancy_mt/mask = '
                  f'{statistics.mean(v for k, v in vals.items() if "sycophancy_mt" not in k and "mask" not in k):.4f}')
            gs = {k: v["gender-secret"] for k, v in t['corpus_to_sleeper'].items() if k.startswith(feat + '|')}
            print(f'   {feat}: gender-secret mean over sources = {statistics.mean(gs.values()):.4f}, '
                  f'range [{min(gs.values()):.4f},{max(gs.values()):.4f}]')
    print()
print('=== zeroshot_nulls.json : sleeper_L44_last ===')
z = J('run-coords/zeroshot_nulls.json')['sleeper_L44_last']
for k, v in z.items():
    a = v['deception_axis']
    print(f'{k:24s} random_p95={v["random_p95_auroc_dev"]:.4f} random_max={v["random_max_auroc_dev"]:.4f} '
          f'n_random={v["n_random"]} deception_axis auroc={a["auroc"]:.4f} p={a["p_vs_random"]:.4f} beats_p95={a["beats_random_p95"]}')
