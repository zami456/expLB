"""P4-scale: recompute the 3B / 8B / 70B scale-rung headline numbers.

Reads only audit/phase2/layer_sensitivity.csv (W1's per-layer off-diagonal means,
themselves computed from the stored crossgen matrices) and prints the tables that
F-58, F-63, F-100, F-101, F-102 and F-110 quote.

Run:  PYTHONIOENCODING=utf-8 python audit/scripts/claims/P4_scale.py [section]
"""
import csv, sys, collections, statistics

LS = 'audit/phase2/layer_sensitivity.csv'


def load(**filt):
    out = []
    with open(LS, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            ok = True
            for k, v in filt.items():
                if isinstance(v, (list, tuple, set)):
                    if r[k] not in v:
                        ok = False
                        break
                elif r[k] != v:
                    ok = False
                    break
            if ok:
                r['offdiag_mean'] = float(r['offdiag_mean'])
                r['layer'] = int(r['layer'])
                out.append(r)
    return out


def bylayer(rows):
    """{layer: {feature: offdiag_mean}}"""
    d = collections.defaultdict(dict)
    for r in rows:
        d[r['layer']][r['feature']] = r['offdiag_mean']
    return d


def summarise(store, pooling, dataset_set='paper5', k='3', probe='paper', seed='42',
              layers=None, best_feats=('axes', 'pc:deception')):
    rows = load(store=store, pooling=pooling, dataset_set=dataset_set, k=k,
                probe=probe, seed=seed)
    d = bylayer(rows)
    ls = sorted(d) if layers is None else [l for l in sorted(d) if l in layers]
    rec = []
    for l in ls:
        f = d[l]
        if 'raw' not in f or 'datapca' not in f:
            continue
        best = max(f[b] for b in best_feats if b in f)
        bestname = max(((f[b], b) for b in best_feats if b in f))[1]
        rec.append(dict(layer=l, raw=f['raw'], datapca=f['datapca'], best=best,
                        bestname=bestname, axes=f.get('axes'),
                        pcdec=f.get('pc:deception'),
                        pcuni2=f.get('pc:unified@2'), pcuni=f.get('pc:unified'),
                        random=f.get('random'), length=f.get('length'),
                        best_raw=best - f['raw'], best_dpca=best - f['datapca'],
                        axes_dpca=f['axes'] - f['datapca'] if 'axes' in f else None))
    return rec


def agg(rec, key):
    v = [r[key] for r in rec if r[key] is not None]
    return (statistics.mean(v) if v else float('nan')), len(v), sum(1 for x in v if x > 0)


def table(title, rec):
    print(f'\n## {title}  ({len(rec)} layers: {[r["layer"] for r in rec]})')
    print(f'{"L":>4} {"raw":>8} {"axes":>8} {"pc:dec":>8} {"dpca":>8} {"best":>8} '
          f'{"best-raw":>9} {"best-dpca":>10} {"bestfeat"}')
    for r in rec:
        print(f'{r["layer"]:>4} {r["raw"]:>8.4f} {r["axes"]:>8.4f} {r["pcdec"]:>8.4f} '
              f'{r["datapca"]:>8.4f} {r["best"]:>8.4f} {r["best_raw"]:>9.4f} '
              f'{r["best_dpca"]:>10.4f} {r["bestname"]}')
    for key in ('raw', 'best_raw', 'best_dpca', 'axes_dpca'):
        m, n, pos = agg(rec, key)
        print(f'  layer-mean {key:>10} = {m:+.4f}   (positive at {pos}/{n})')


F100_LAYERS = [l for l in [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48,
                           52, 56, 60, 64, 68]]  # 19 layers, excludes 72/76

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if what in ('all', 'f63'):
        print('=' * 78)
        print('F-63 / F-58 single-layer cells (mean pooling, k=3, probe=paper, seed 42)')
        for store, lay in (('scale_3b', 16), ('scale_8b', 16), ('scale_70b', 40)):
            rec = summarise(store, 'mean', layers=[lay])
            for r in rec:
                print(f'{store:>26} L{r["layer"]:<3} raw={r["raw"]:.4f} '
                      f'axes={r["axes"]:.4f} pc:dec={r["pcdec"]:.4f} '
                      f'pc:uni@2={r["pcuni2"]} dpca={r["datapca"]:.4f} '
                      f'random={r["random"]} | best={r["bestname"]} '
                      f'best-raw={r["best_raw"]:+.4f} best-dpca={r["best_dpca"]:+.4f}')
        rec = summarise('scale_70b_paper16fp32', 'mean', layers=[40])
        for r in rec:
            print(f'{"scale_70b_paper16fp32":>26} L{r["layer"]:<3} raw={r["raw"]:.4f} '
                  f'axes={r["axes"]:.4f} pc:dec={r["pcdec"]:.4f} dpca={r["datapca"]:.4f} '
                  f'| best={r["bestname"]} best-raw={r["best_raw"]:+.4f} '
                  f'best-dpca={r["best_dpca"]:+.4f}')

    if what in ('all', 'f100'):
        print('\n' + '=' * 78)
        print('F-100 / F-110: three-rung layer-means, mean pooling, k=3, probe=paper')
        for store, lays in (('scale_3b', None), ('scale_8b', None),
                            ('scale_70b', F100_LAYERS), ('scale_70b', None),
                            ('scale_70b_paper16fp32', F100_LAYERS),
                            ('scale_70b_paper16fp32', None)):
            rec = summarise(store, 'mean', layers=lays)
            tag = '19L' if lays else 'allL'
            mr, nr, _ = agg(rec, 'raw')
            mbr, nbr, pbr = agg(rec, 'best_raw')
            mbd, nbd, pbd = agg(rec, 'best_dpca')
            mad, nad, pad = agg(rec, 'axes_dpca')
            print(f'{store:>26} {tag:>5} n={nr:>2}  raw={mr:.4f}  '
                  f'best-raw={mbr:+.4f} ({pbr}/{nbr})  best-dpca={mbd:+.4f} ({pbd}/{nbd})  '
                  f'axes-dpca={mad:+.4f}')

    if what in ('all', 'f101'):
        print('\n' + '=' * 78)
        print('F-101 / F-102: both poolings, all stored layers, k=3, probe=paper')
        for store in ('scale_3b', 'scale_8b', 'scale_70b', 'scale_70b_paper16fp32'):
            for pool in ('mean', 'last'):
                rec = summarise(store, pool)
                mbr, nbr, pbr = agg(rec, 'best_raw')
                mbd, nbd, pbd = agg(rec, 'best_dpca')
                mad, _, _ = agg(rec, 'axes_dpca')
                mraw, _, _ = agg(rec, 'raw')
                mbest = statistics.mean(r['best'] for r in rec)
                # L40 rank on best-dpca (descending)
                rank = None
                if any(r['layer'] == 40 for r in rec):
                    order = sorted(rec, key=lambda r: -r['best_dpca'])
                    rank = [r['layer'] for r in order].index(40) + 1
                print(f'{store:>26} {pool:>5} n={nbr:>2}  raw={mraw:.4f} best={mbest:.4f}  '
                      f'best-raw={mbr:+.4f} ({pbr}/{nbr})  best-dpca={mbd:+.4f} ({pbd}/{nbd})  '
                      f'axes-dpca={mad:+.4f}  L40rank={rank}')

    if what in ('all', 'detail'):
        print('\n' + '=' * 78)
        for store in ('scale_3b', 'scale_8b', 'scale_70b', 'scale_70b_paper16fp32'):
            for pool in ('mean', 'last'):
                table(f'{store} / {pool}', summarise(store, pool))
