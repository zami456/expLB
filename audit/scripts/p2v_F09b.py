"""F-09 deciles under several quantile conventions + margin peakedness."""
import json, statistics, sys
p = sys.argv[1]
v = []
m = []
with open(p, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if isinstance(r.get("logprob_gold"), (int, float)): v.append(r["logprob_gold"])
        if isinstance(r.get("margin"), (int, float)): m.append(r["margin"])
v.sort(); n = len(v)
print("n =", n)
# convention A: statistics.quantiles (inclusive) -> 9 cut points
qa = statistics.quantiles(v, n=10, method="inclusive")
print("deciles cuts (inclusive, 9 pts):", [round(x, 1) for x in qa])
# convention B: value at index floor(i*n/10) for i=0..9 (lower edge of each decile bin)
qb = [v[int(i * n / 10)] for i in range(10)]
print("bin lower edges floor(i*n/10):  ", [round(x, 1) for x in qb])
# convention C: numpy-style linear percentile at 0,10,...,90
def pct(q):
    k = q / 100 * (n - 1); lo = int(k); hi = min(lo + 1, n - 1); fr = k - lo
    return v[lo] * (1 - fr) + v[hi] * fr
qc = [pct(i * 10) for i in range(10)]
print("percentiles 0,10,..,90 (linear):", [round(x, 1) for x in qc])
print("exact zeros:", sum(1 for x in v if x == 0.0), f"({sum(1 for x in v if x==0.0)/n:.1%})")
print(">= -0.0001 :", sum(1 for x in v if x >= -1e-4), f"({sum(1 for x in v if x>=-1e-4)/n:.1%})")
print(">= -0.01   :", sum(1 for x in v if x >= -1e-2), f"({sum(1 for x in v if x>=-1e-2)/n:.1%})")
print("margin: n=", len(m), "mean", round(statistics.fmean(m), 4), "| >=0.99:", sum(1 for x in m if x >= 0.99), f"({sum(1 for x in m if x>=0.99)/len(m):.1%})")
