"""F-59 power check: at n=20 (and 12), are the three denial rates distinguishable?
Pure arithmetic on the counts recomputed in P6_lab_a.py; no data files needed."""
from math import comb


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    p0 = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI by bisection on the binomial tail (no scipy)."""
    def tail_ge(p):   # P(X >= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    def tail_le(p):   # P(X <= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_ge(m) < alpha / 2: a = m
            else: b = m
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if tail_le(m) > alpha / 2: a = m
            else: b = m
        hi = (a + b) / 2
    return lo, hi


print("denial rates as measured (from kd/run/denial/*.jsonl):")
for name, k, n in (("primary  (wrong, harmful)", 5, 20),
                   ("fallback1(wrong, control)", 3, 20),
                   ("fallback2(margin, harmful)", 5, 12)):
    lo, hi = clopper_pearson(k, n)
    print(f"  {name:28} {k}/{n} = {k/n:.3f}   exact 95% CI [{lo:.3f}, {hi:.3f}]")
print()
print("the gate was 'denial rate >= 0.70':")
for name, k, n in (("primary", 5, 20), ("fallback1", 3, 20), ("fallback2", 5, 12)):
    lo, hi = clopper_pearson(k, n)
    print(f"  {name:10} upper 95% bound {hi:.3f} -> gate {'EXCLUDED' if hi < 0.70 else 'not excluded'}")
print()
print("are the routes distinguishable from each other? (Fisher exact, two-sided)")
print(f"  primary 5/20 vs fallback1 3/20 : p = {fisher_two_sided(5, 15, 3, 17):.4f}")
print(f"  primary 5/20 vs fallback2 5/12 : p = {fisher_two_sided(5, 15, 5, 7):.4f}")
print(f"  fallback1 3/20 vs fallback2 5/12: p = {fisher_two_sided(3, 17, 5, 7):.4f}")
