"""M-029: memory note paper16-assistant-direction-not-lu-axis.md (description line 3 + table 19-24)

  "paper16's 'assistant' axis is default minus mean(25 paper personas), only cos 0.30-0.37 with
   Lu et al.'s published axis; against the published axis paper16's deception contamination is
   -0.14..-0.23, not -0.34..-0.37"

  table: layer | cos(p16 assistant, published AA) | p16_s3 deception vs published AA |
         HF 22v22 grouping vs published AA | p16_s3 deception vs its own assistant
          20 | 0.366 | -0.168 | -0.686 | -0.364
          22 | 0.306 | -0.141 | -0.662 | -0.338
          40 | 0.372 | -0.226 | -0.610 | -0.370
          60 | 0.300 | -0.180 | -0.516 | -0.374

Inputs: the cached published axis (lu-christina/assistant-axis-vectors, llama-3.3-70b/
assistant_axis.pt) and run-paper16/axes_vectors_s3.pt (the OLD bank, which is what the
note's date 2026-09-17 and vectors_dir imply).  Read-only.

The HF 22v22 column is reproduced with the CONCEAL/DISCLOSE role groups hard-coded in
src/assistant_axis_ndif/16_regroup_hf.py, using the cached role_vectors.
"""
import glob, os, sys
import numpy as np
import torch

CACHE = os.path.expanduser('~/.cache/huggingface/hub/'
                           'datasets--lu-christina--assistant-axis-vectors/snapshots')
snap = sorted(glob.glob(os.path.join(CACHE, '*')))[0]
AA_PATH = os.path.join(snap, 'llama-3.3-70b', 'assistant_axis.pt')
BANK = sys.argv[1] if len(sys.argv) > 1 else 'run-paper16/axes_vectors_s3.pt'
LAYERS = [20, 22, 40, 60]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


obj = torch.load(AA_PATH, map_location='cpu', weights_only=False)
print(f'### published axis: {AA_PATH}')
if isinstance(obj, dict):
    print('    keys:', list(obj)[:10])
    aa = obj.get('vector', obj.get('axis', None))
    if aa is None:
        aa = list(obj.values())[0]
else:
    aa = obj
aa = torch.as_tensor(aa).float().numpy()
print(f'    published axis shape {aa.shape} dtype {getattr(obj, "dtype", "")}')

bank = torch.load(BANK, map_location='cpu', weights_only=False)
print(f'\n### paper16 bank: {BANK}  vectors_dir={bank.get("vectors_dir")}')
print(f'    roles stored: {len(bank["roles"])}; role_groups: '
      f'{ {k: len(v) for k, v in bank["role_groups"].items()} }')


def axis(name, L):
    a = bank['axes'][name]
    if isinstance(a, dict):
        return torch.as_tensor(a[L]).float().numpy()
    return torch.as_tensor(a).float().numpy()[bank['layers'].index(L)]


# HF 22v22: concealment vs disclosure groups from 16_regroup_hf.py
CONCEAL = None
src = 'src/assistant_axis_ndif/16_regroup_hf.py'
if os.path.exists(src):
    ns = {}
    txt = open(src, encoding='utf-8').read()
    for tag in ['CONCEAL', 'DISCLOSE']:
        i = txt.index(f'{tag} = ')
        j = txt.index(']', i)
        exec(txt[i:j + 1], ns)
    CONCEAL, DISCLOSE = ns['CONCEAL'], ns['DISCLOSE']
    print(f'    CONCEAL {len(CONCEAL)} roles, DISCLOSE {len(DISCLOSE)} roles')

rv = {}
for p in glob.glob(os.path.join(snap, 'llama-3.3-70b', 'role_vectors', '*.pt')):
    o = torch.load(p, map_location='cpu', weights_only=False)
    v = o['vector'] if isinstance(o, dict) and 'vector' in o else o
    rv[os.path.basename(p)[:-3]] = torch.as_tensor(v).float().numpy()
print(f'    cached HF role vectors: {len(rv)}')

print(f'\n{"layer":>6} {"cos(p16 asst, AA)":>18} {"p16 dec vs AA":>15} '
      f'{"HF 22v22 vs AA":>15} {"p16 dec vs own asst":>20}')
for L in LAYERS:
    asst = axis('assistant', L)
    dec = axis('deception', L)
    c1 = cos(asst, aa[L])
    c2 = cos(dec, aa[L])
    c4 = cos(dec, asst)
    c3 = float('nan')
    if CONCEAL:
        p = [rv[n][L] for n in CONCEAL if n in rv]
        q = [rv[n][L] for n in DISCLOSE if n in rv]
        if p and q:
            c3 = cos(np.mean(p, axis=0) - np.mean(q, axis=0), aa[L])
    print(f'{L:>6} {c1:>18.3f} {c2:>15.3f} {c3:>15.3f} {c4:>20.3f}   '
          f'(HF groups {len(p)}v{len(q)})' if CONCEAL else '')
