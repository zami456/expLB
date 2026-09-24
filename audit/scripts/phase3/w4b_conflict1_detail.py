"""W4b CONFLICT 1 detail: the 3 f_ids that differ between fmap's print-only set and
findings_targets.csv, and the true recovery tally in findings_printed_numbers.csv."""
import csv, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
def load(p):
    with open(ROOT / p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
fmap = {r['f_id'].strip(): r for r in load('audit/phase1/fmap_all.csv')}
targ = {r['f_id'].strip(): r for r in load('audit/phase1/sessions/findings_targets.csv')}
prn  = load('audit/phase1/sessions/findings_printed_numbers.csv')

print("=== the 3 boundary f_ids ===")
for fid in ('F-35','F-58','F-88','F-44'):
    r = fmap.get(fid, {})
    t = targ.get(fid)
    print(f"\n{fid}  in_targets={t is not None}")
    print("  fmap.script_writes_file :", repr(r.get('script_writes_file')))
    print("  fmap.mapping_status     :", repr(r.get('mapping_status')))
    print("  fmap.title_short        :", repr(r.get('title_short'))[:120])
    if t:
        print("  targ.script_writes_file :", repr(t.get('script_writes_file')))
        print("  targ.notes              :", repr(t.get('notes'))[:200])

print("\n=== recovery tally in findings_printed_numbers.csv ===")
byid = collections.defaultdict(list)
for r in prn:
    byid[r['f_id'].strip()].append(((r['found'] or '').strip(), (r['matches_ledger'] or '').strip()[:1]))
def best(v):
    fs = [a for a,_ in v]
    if any(a=='y' for a in fs): return 'y'
    if any(a=='partial' for a in fs): return 'partial'
    return 'n'
tal = collections.Counter(best(v) for v in byid.values())
print("per-f_id best 'found':", dict(tal), " total f_ids:", len(byid))
for fid,v in sorted(byid.items()):
    if best(v) != 'y':
        print(f"  {fid}: {v}")
# matches_ledger normalised by first token
m = collections.Counter()
for r in prn:
    s = (r['matches_ledger'] or '').strip().lower()
    if s.startswith('y'): m['y'] += 1
    elif s.startswith('partial'): m['partial'] += 1
    elif s.startswith('n/a'): m['n/a'] += 1
    elif s.startswith('n'): m['n'] += 1
    else: m['?'] += 1
print("matches_ledger normalised (144 rows):", dict(m))
