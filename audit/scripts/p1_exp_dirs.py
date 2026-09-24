"""Phase 1 Task A helper: aggregate the lab census (joined_labpc.tsv.gz) by folder.
Streams the gz file (low RAM). For every root + directory prefix of depth 1..3 prints:
files, MB, min/max mtime, #on_drive, #in_asix, #in_pcood, #asix_same_name_diff, top extensions.
Output: audit/phase1/exp_work/lab_dirs.tsv
Usage: python audit/scripts/p1_exp_dirs.py
"""
import gzip, collections, os
BS = chr(92)
BASE = os.path.join(os.path.dirname(__file__), '..', 'phase1')
src = os.path.join(BASE, 'labpc', 'joined_labpc.tsv.gz')
out = os.path.join(BASE, 'exp_work', 'lab_dirs.tsv')
agg = {}
with gzip.open(src, 'rt', encoding='utf-8', errors='replace') as f:
    hdr = f.readline().rstrip('\n').split('\t')
    ix = {h: i for i, h in enumerate(hdr)}
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < len(hdr):
            p += [''] * (len(hdr) - len(p))
        root = p[ix['root']].replace(BS, '/')
        rel = p[ix['relpath']].replace(BS, '/')
        parts = rel.split('/')
        dirs = parts[:-1]
        ext = os.path.splitext(parts[-1])[1].lower() or '(none)'
        try:
            b = int(p[ix['bytes']] or 0)
        except ValueError:
            b = 0
        mt = p[ix['mtime']]
        for d in range(0, min(3, len(dirs)) + 1):
            key = (root, '/'.join(dirs[:d]) if d else '<root>', d)
            a = agg.get(key)
            if a is None:
                a = agg[key] = [0, 0, '9999', '0000', 0, 0, 0, 0, collections.Counter()]
            a[0] += 1; a[1] += b
            if mt and mt < a[2]: a[2] = mt
            if mt and mt > a[3]: a[3] = mt
            a[4] += p[ix['on_drive']] == '1'
            a[5] += p[ix['in_asix']] == '1'
            a[6] += p[ix['in_pcood']] == '1'
            a[7] += bool(p[ix['asix_same_name_diff_content']].strip())
            a[8][ext] += 1
with open(out, 'w', encoding='utf-8', newline='\n') as o:
    o.write('root\tdir\tdepth\tfiles\tMB\tmin_mtime\tmax_mtime\ton_drive\tin_asix\tin_pcood\tasix_diff\ttop_ext\n')
    for (root, d, dep), a in sorted(agg.items()):
        te = ','.join(f'{k}:{v}' for k, v in a[8].most_common(4))
        o.write(f'{root}\t{d}\t{dep}\t{a[0]}\t{a[1]/1e6:.1f}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\t{a[6]}\t{a[7]}\t{te}\n')
print('rows', len(agg), '->', out)
