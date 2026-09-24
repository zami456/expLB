"""Create the harm-pressure-kr-matched VIEW in passA and passC, as already done for passB.

Same mechanism: a directory holding a filtered rows.jsonl plus HARDLINKS to the same chunks.
load_dataset keys rows by row_id and skips any chunk row absent from rows.jsonl, so the view
yields exactly the matched slice with no re-extraction.

passA and passC are 2,000-row SUBSAMPLES, so this view will be far smaller there than the 258
rows it has in passB. The counts are printed rather than assumed, because that number decides
whether the column is reportable at all.
"""
import json, os, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'src/assistant_axis_ndif')
import eval_datasets as E

keep = E.hpkr_slice('text_matched')
print(f"matched row_ids in the full corpus: {len(keep)}\n")
for store in ("passA", "passC"):
    SRC = Path(f'run-coords/{store}/harm-pressure-knowledge-report')
    DST = Path(f'run-coords/{store}/harm-pressure-kr-matched')
    if not SRC.is_dir():
        print(f"{store}: source absent"); continue
    rows = [json.loads(l) for l in (SRC/'rows.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    sel = [r for r in rows if r['row_id'] in keep]
    c = Counter(r.get('label') for r in sel)
    print(f"{store}: HP-KR rows {len(rows)} -> matched {len(sel)}   labels {dict(c)}")
    if len(sel) < 40:
        print(f"  ! only {len(sel)} rows -- too thin to train on; the column will be noise")
    DST.mkdir(parents=True, exist_ok=True)
    (DST/'rows.jsonl').write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in sel),
                                  encoding='utf-8')
    n = 0
    for ch in sorted(SRC.glob('chunk_*.pt')):
        t = DST/ch.name
        if t.exists(): continue
        try: os.link(ch, t); n += 1
        except OSError:
            import shutil; shutil.copy2(ch, t); n += 1
    print(f"  linked {n} chunks")
