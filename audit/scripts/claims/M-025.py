# M-025: "F-94 | F-89's '78% / three-to-one' | ~2:1 over 16 layers; only *presence* clears"
# Level 1: recompute the ratio and the clear-counts from F-94's own 16-layer table
# (kd/docs/FINDINGS.md lines 4114-4130), parsed from the file.
import re, statistics

txt = open('kd/docs/FINDINGS.md', encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(txt) if l.startswith('### F-94 '))
tbl = [l for l in txt[start:start + 40] if re.match(r'^\|\s*\**\d', l)]
print(f'F-94 table rows parsed from kd/docs/FINDINGS.md: {len(tbl)}')
content, presence, clears_c, clears_p, retention = [], [], 0, 0, []
for l in tbl:
    c = [x.strip().replace('**', '') for x in l.strip('|').split('|')]
    L = int(c[0])
    hn, sn, ret = c[6], c[7], c[8]
    cv = float(hn.replace('\\*', '').replace('*', '').replace('+', ''))
    pv = float(sn.replace('\\*', '').replace('*', '').replace('+', ''))
    content.append(cv); presence.append(pv)
    if '*' in hn: clears_c += 1
    if '*' in sn: clears_p += 1
    retention.append(float(ret.replace('%', '').replace('−', '-')))
    print(f'  L{L:>2}  content {cv:+.3f}{"*" if "*" in hn else " "}   presence {pv:+.3f}{"*" if "*" in sn else " "}   retains {ret}')
mc, mp = statistics.fmean(content), statistics.fmean(presence)
print(f'\nmean content  = {mc:+.4f}   (the entry says +0.034)')
print(f'mean presence = {mp:+.4f}   (the entry says +0.070)')
print(f'presence : content = {mp/mc:.2f} : 1   -> "about 2:1"')
print(f'content clears at {clears_c} of {len(tbl)} layers; presence clears at {clears_p} of {len(tbl)}')
print(f'retention median {statistics.median(retention):.0f}%  min {min(retention):.0f}%  max {max(retention):.0f}%')
print('\n(F-89 layer-40 row, the superseded headline: retention 78%, presence/content = '
      f'{0.120/0.025:.1f}:1 -> "three-to-one")')
