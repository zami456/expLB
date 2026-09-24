"""Create the harm-pressure-kr-matched VIEW over passB's existing HP-KR activations.

9_persona_probes.load_dataset keys rows by row_id and SKIPS any chunk row absent from
rows.jsonl. So a directory holding a 258-line rows.jsonl plus the same chunk files yields
exactly the matched slice -- no re-extraction, no code change to the loader.

Chunks are HARDLINKED, not copied: same volume, zero extra bytes, and unlike a junction a
hardlinked file is safe to delete individually.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'src/assistant_axis_ndif')
import eval_datasets as E

STORE = Path('run-coords/passB')
SRC = STORE / 'harm-pressure-knowledge-report'
DST = STORE / 'harm-pressure-kr-matched'

keep = E.hpkr_slice('text_matched')
print(f"matched row_ids: {len(keep)}")

rows = [json.loads(l) for l in (SRC / 'rows.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
sel = [r for r in rows if r['row_id'] in keep]
print(f"source rows.jsonl {len(rows)} -> selected {len(sel)}")
from collections import Counter
print("label balance:", dict(Counter(r.get('label') for r in sel)))
print("distinct item_ids:", len({r.get('item_id') for r in sel}))

DST.mkdir(parents=True, exist_ok=True)
(DST / 'rows.jsonl').write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel), encoding='utf-8')

n_link = 0
for c in sorted(SRC.glob('chunk_*.pt')):
    t = DST / c.name
    if t.exists():
        continue
    try:
        os.link(c, t); n_link += 1
    except OSError as e:
        print("hardlink failed, falling back to copy:", e)
        import shutil; shutil.copy2(c, t); n_link += 1
print(f"chunks linked: {n_link} (source has {len(list(SRC.glob('chunk_*.pt')))})")

# prove the view loads to exactly the matched rows
import importlib.util as ilu
spec = ilu.spec_from_file_location('P', 'src/assistant_axis_ndif/9_persona_probes.py')
P = ilu.module_from_spec(spec); spec.loader.exec_module(P)
X, meta = P.load_dataset(STORE, 'harm-pressure-kr-matched', 40, 'mean')
print(f"\nVIEW LOADS: X{X.shape}  meta {len(meta)}")
print("labels:", dict(Counter(m['label'] for m in meta)))
Xf, mf = P.load_dataset(STORE, 'harm-pressure-knowledge-report', 40, 'mean')
print(f"full HP-KR for comparison: X{Xf.shape}")
