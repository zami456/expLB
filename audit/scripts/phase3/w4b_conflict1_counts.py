"""W4b CONFLICT 1: recount print-only ledger entries from raw audit files.

Sources:
  audit/phase1/fmap_all.csv                          (109 F-entries; free-text col script_writes_file)
  audit/phase1/sessions/findings_targets.csv         (recovery targets)
  audit/phase1/sessions/findings_printed_numbers.csv (recovered console numbers)
Run: python audit/scripts/phase3/w4b_conflict1_counts.py
"""
import csv, collections, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[3]

def load(p):
    with open(ROOT / p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

fmap = load('audit/phase1/fmap_all.csv')
targ = load('audit/phase1/sessions/findings_targets.csv')
prn  = load('audit/phase1/sessions/findings_printed_numbers.csv')

def norm(v):
    """Classify the free-text script_writes_file cell.
    PURE_Y  : starts with y and never mentions a print-only part
    PURE_N  : starts with n (a print-only entry with no persisted artifact)
    MIXED   : names both a written artifact and a print-only part
    OTHER   : n/a or unparseable
    """
    s = (v or '').strip().lower()
    if not s or s.startswith('n/a'):
        return 'OTHER'
    has_n = bool(re.search(r'(^|[^a-z])n($|[^a-z])', s)) or 'print-only' in s or 'console' in s or 'by hand' in s
    starts_y = s.startswith('y')
    starts_n = s.startswith('n')
    if starts_y and has_n:
        return 'MIXED'
    if starts_y:
        return 'PURE_Y'
    if starts_n:
        return 'PURE_N'
    return 'OTHER'

cls = {r['f_id'].strip(): norm(r['script_writes_file']) for r in fmap}
cc = collections.Counter(cls.values())
print("=== A. fmap_all.csv (109 F-entries) ===")
print("rows:", len(fmap), "| distinct f_id:", len(set(cls)))
for k in ('PURE_Y','MIXED','PURE_N','OTHER'):
    print(f"  {k:7s} {cc[k]}")
print("  PURE_N + MIXED (entries with ANY print-only number) =", cc['PURE_N']+cc['MIXED'])
print("  PURE_Y alone (fully persisted)                      =", cc['PURE_Y'])
print("  PURE_Y + MIXED (entries that write SOME file)       =", cc['PURE_Y']+cc['MIXED'])
print("  first-char tally: y =", sum(1 for r in fmap if (r['script_writes_file'] or '').strip().lower().startswith('y')),
      " n =", sum(1 for r in fmap if (r['script_writes_file'] or '').strip().lower().startswith('n') and not (r['script_writes_file'] or '').strip().lower().startswith('n/a')),
      " other =", sum(1 for r in fmap if not (r['script_writes_file'] or '').strip().lower()[:1] in ('y','n') or (r['script_writes_file'] or '').strip().lower().startswith('n/a')))
print("  PURE_N f_ids:", sorted(k for k,v in cls.items() if v=='PURE_N'))

print()
print("=== B. findings_targets.csv ===")
print("rows:", len(targ))
tids = [r['f_id'].strip() for r in targ]
print("distinct f_id:", len(set(tids)))
print("duplicate f_id:", {k:v for k,v in collections.Counter(tids).items() if v>1})
ct = collections.Counter(norm(r['script_writes_file']) for r in targ)
print("normalised script_writes_file:", dict(ct))

ts = set(tids); fs = set(cls)
pure_n = {k for k,v in cls.items() if v=='PURE_N'}
mixed  = {k for k,v in cls.items() if v=='MIXED'}
anyn   = pure_n | mixed
print()
print("=== C. overlap: does 72 == the fmap print-only set? ===")
print("targets not in fmap:", sorted(ts - fs))
print("|targets| =", len(ts), " |fmap PURE_N| =", len(pure_n), " |fmap PURE_N+MIXED| =", len(anyn))
print("targets & (PURE_N+MIXED):", len(ts & anyn))
print("(PURE_N+MIXED) missing from targets:", sorted(anyn - ts))
print("targets that are neither PURE_N nor MIXED:", sorted(ts - anyn), "->",
      {k: cls.get(k,'NOT-IN-FMAP') for k in sorted(ts - anyn)})

print()
print("=== D. findings_printed_numbers.csv ===")
print("rows:", len(prn), "| distinct f_id:", len(set(r['f_id'].strip() for r in prn)))
print("found:", dict(collections.Counter((r['found'] or '').strip() for r in prn)))
print("matches_ledger:", dict(collections.Counter((r['matches_ledger'] or '').strip() for r in prn)))
byid = collections.defaultdict(list)
for r in prn:
    byid[r['f_id'].strip()].append((r['found'] or '').strip())
rec = {k for k,v in byid.items() if any(x.startswith('y') for x in v)}
print("f_ids with >=1 found=y:", len(rec))
print("targets with NO found=y row:", sorted(ts - rec))
print("printed_numbers f_ids not in targets:", sorted(set(byid) - ts))
