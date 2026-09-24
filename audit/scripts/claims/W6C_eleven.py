"""W6C-briefing: the parts of the "our eleven" table (section 5a) that are reachable
without the lab PC.

The table's five rows at L20 / L40 are
    our raw single-source baseline   0.6289 / 0.6623
    axis1 (8v7)                      0.6572 / 0.7325     <- collab bank, LAB ONLY
    ext_axis1 (12v13)                0.7386 / 0.7996     <- collab bank, LAB ONLY
    pooled leave-one-dataset-out     0.8027 / 0.8545
    share of the margin from the kd bank   74% / 49%

This script recomputes the raw baseline from the local kd11 matrices and the pooled LODO
from the cached run-premise files, then reconstructs the share from those two plus the two
lab-only arms as recorded by batch P12 (audit/phase2/out/P12/P12_lab.txt), so the arithmetic
of the share can be checked even while the collab matrices are unreachable.
"""
import hashlib
import io
import json
import sys

p = sys.stdout.write


def load(path):
    with io.open(path, "rb") as f:
        b = f.read()
    return json.loads(b.decode("utf-8")), hashlib.md5(b).hexdigest()


def raw_offdiag(m, feature="raw", probe="paper"):
    for r in m["results"]:
        if r["feature"] == feature and r["probe"] == probe:
            a = r["auroc"]
            n = len(a)
            v = [a[i][j] for i in range(n) for j in range(n) if i != j and a[i][j] is not None]
            return sum(v) / len(v), len(v)
    return None, 0


p("== raw single-source baseline, our eleven, from the LOCAL kd11 matrices ==\n")
raw = {}
for pool in ("mean", "last"):
    for L in (20, 40):
        f = "run-crossgen/kd11/matrices/%s_L%d_s42.json" % (pool, L)
        m, md5 = load(f)
        v, n = raw_offdiag(m)
        raw[(pool, L)] = v
        p("  %-5s L%-3d raw off-diagonal mean = %.4f  (n=%d pairs, %d datasets)  md5 %s\n"
          % (pool, L, v, n, len(m["datasets"]), md5))
p("  briefing section 5a: 0.6289 (L20) and 0.6623 (L40), mean pooling\n\n")

p("== pooled leave-one-dataset-out, our eleven, from the CACHED run-premise files ==\n")
pooled = {}
for L in (20, 40):
    d, md5 = load("audit/cache/labpc/run-premise/premise_L%d_mean_s42.json" % L)
    pooled[L] = d["summary"]["pooled_lodo"]
    p("  L%-3d pooled_lodo = %.4f   pooled_lodo_match = %.4f   single = %.4f   md5 %s\n"
      % (L, d["summary"]["pooled_lodo"], d["summary"]["pooled_lodo_match"],
         d["summary"]["single"], md5))
p("  briefing section 5a: 0.8027 (L20) and 0.8545 (L40)\n\n")

p("== the share, reconstructed ==\n")
p("  share = (ext_axis1 - axis1) / (ext_axis1 - raw)\n")
P12 = {20: (0.6572, 0.7386), 40: (0.7325, 0.7996)}  # axis1, ext_axis1 -- from P12_lab.txt
for L in (20, 40):
    a1, ea = P12[L]
    r = raw[("mean", L)]
    p("  L%-3d axis1 %.4f (P12/lab)  ext %.4f (P12/lab)  raw %.4f (recomputed here)"
      "  -> share %.1f%%\n" % (L, a1, ea, r, 100.0 * (ea - a1) / (ea - r)))
    p("        gap to pooled: axis1 %+.4f   ext_axis1 %+.4f\n"
      % (pooled[L] - a1, pooled[L] - ea))
p("  briefing section 5a / section 7: 74%% (L20) and 49%% (L40); 'kd bank is 49-74%% of the margin'\n")
p("  briefing section 5a: 'the extended arm by 0.064 at L20 and 0.055 at L40,"
  " and axis1 by 0.146 and 0.122'\n")
