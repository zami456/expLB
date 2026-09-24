"""P3-C: the dataset-level significance ceiling. With the target dataset as bootstrap unit
(n = 5..11), how wide is the interval, and what is the smallest margin the design can resolve?
Reads the lab significance.json copies in audit/cache/. Read-only."""
import json, glob, os, statistics as st, collections

files = sorted(glob.glob('audit/cache/labpc/run-crossgen/**/significance.json', recursive=True))
agg = collections.defaultdict(list)
for f in files:
    d = json.load(open(f, encoding='utf-8'))
    tests = d['tests'] if isinstance(d, dict) and 'tests' in d else d
    for t in tests:
        ci = t.get('dataset_ci95') or t.get('ci95')
        if not ci or len(ci) != 2:
            continue
        hw = (ci[1] - ci[0]) / 2
        label = f.split('run-crossgen')[1].replace(os.sep, '/')
        agg[label].append((abs(t.get('diff', 0)), hw, t.get('dataset_q_bh'), t.get('dataset_p')))

tot = []
print(f"{'file':44s} {'tests':>6s} {'medHW':>7s} {'p90HW':>7s} {'|diff|>HW':>12s} {'q<0.05':>7s}")
for f, v in sorted(agg.items()):
    hw = [x[1] for x in v]
    res = sum(1 for d, h, q, p in v if d > h)
    sig = sum(1 for d, h, q, p in v if q is not None and q < 0.05)
    print(f"{f:44s} {len(v):6d} {st.median(hw):7.4f} {sorted(hw)[int(.9*len(hw))]:7.4f} "
          f"{res:6d} ({100*res/len(v):4.1f}%) {sig:6d}")
    tot += v

hw = [x[1] for x in tot]
print(f"\nALL {len(tot)} dataset-level tests: median half-width {st.median(hw):.4f}, "
      f"mean {st.mean(hw):.4f}, p10 {sorted(hw)[int(.1*len(hw))]:.4f}, "
      f"p90 {sorted(hw)[int(.9*len(hw))]:.4f}")
res = sum(1 for d, h, q, p in tot if d > h)
print(f"  tests whose |effect| exceeds its own half-width: {res} ({100*res/len(tot):.1f}%)")
q = [x[2] for x in tot if x[2] is not None]
if q:
    print(f"  tests clearing BH q<0.05 as the files stand today: "
          f"{sum(1 for x in q if x < 0.05)} of {len(q)} "
          f"({100*sum(1 for x in q if x < 0.05)/len(q):.2f}%)")
