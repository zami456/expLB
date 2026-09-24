"""Phase-2 batch P2-pooled: F-96 (F-63's three-rung margin across depth).

Uses audit/phase2/layer_sensitivity.csv (W1), which holds the off-diagonal mean per
(store, bank, k, pooling, layer, seed, dataset_set, feature, probe).

F-96 says: k=3, probe=paper, mean pooling, seed 42, on
instructed-deception / mask / convincing-game / ai_liar / roleplaying  (= dataset_set 'paper5').
"""
from __future__ import annotations

import collections
import csv
import statistics

SRC = "audit/phase2/layer_sensitivity.csv"
STORES = {"scale_3b": "3B", "scale_8b": "8B", "scale_70b": "70B",
          "scale_70b_paper16fp32": "70B paper16fp32 (RF-16 repair)"}

rows = []
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    if r["store"] in STORES:
        rows.append(r)
print(f"rows for the scale stores: {len(rows)}")
combos = collections.Counter((r["store"], r["pooling"], r["probe"], r["dataset_set"], r["seed"],
                              r["k"]) for r in rows)
print("available (store, pooling, probe, dataset_set, seed, k):")
for kk, v in sorted(combos.items()):
    print("   ", kk, v)


def pick(store, pooling="mean", probe="paper", ds="paper5", seed="42"):
    out = {}
    for r in rows:
        if (r["store"] == store and r["pooling"] == pooling and r["probe"] == probe
                and r["dataset_set"] == ds and r["seed"] == seed):
            out.setdefault(int(r["layer"]), {})[r["feature"]] = float(r["offdiag_mean"])
    return out


print("\n" + "=" * 96)
print("F-96  matched-control margin  best(axes, pc:deception) - datapca,  and  axes - raw")
print("=" * 96)
STATED = {"scale_3b": dict(at=("+0.2023", 16), mean="+0.1197", rng="[+0.014, +0.202]",
                           rank="1st of 7", nlay=7, axes_raw_at="+0.0195",
                           axes_raw_mean="-0.0063", axes_raw_pos="4 / 7", best_raw="+0.033",
                           best_raw_pos="7/7"),
          "scale_8b": dict(at=("+0.1147", 16), mean="+0.1027", rng="[+0.030, +0.166]",
                           rank="4th of 8", nlay=8, axes_raw_at="-0.0249",
                           axes_raw_mean="-0.0190", axes_raw_pos="2 / 8", best_raw="+0.009",
                           best_raw_pos="6/8"),
          "scale_70b": dict(at=("+0.0794", 40), mean="-", rng="-", rank="only layer available",
                            nlay=1, axes_raw_at="+0.0131", axes_raw_mean="-",
                            axes_raw_pos="1 / 1", best_raw="-", best_raw_pos="-")}

for store in ("scale_3b", "scale_8b", "scale_70b", "scale_70b_paper16fp32"):
    d = pick(store)
    if not d:
        print(f"\n{store}: no (mean, paper, paper5, seed 42) rows")
        continue
    lays = sorted(d)
    st = STATED.get(store)
    print(f"\n--- {STORES[store]}  ({len(lays)} layers: {lays}) "
          f"{'ledger says ' + str(st['nlay']) if st else ''}")
    print(f"  {'L':>4}{'axes':>9}{'pc:dec':>9}{'best':>9}{'datapca':>9}{'raw':>9}"
          f"{'best-dpca':>11}{'axes-raw':>10}{'best-raw':>10}")
    m_ctrl, m_axraw, m_bestraw = [], [], []
    for L in lays:
        f = d[L]
        ax, pc = f.get("axes"), f.get("pc:deception")
        best = max(v for v in (ax, pc) if v is not None)
        dp, rw = f.get("datapca"), f.get("raw")
        m_ctrl.append((L, best - dp))
        m_axraw.append((L, ax - rw))
        m_bestraw.append((L, best - rw))
        print(f"  {L:>4}{ax:>9.4f}{pc:>9.4f}{best:>9.4f}{dp:>9.4f}{rw:>9.4f}"
              f"{best - dp:>+11.4f}{ax - rw:>+10.4f}{best - rw:>+10.4f}")
    vals = [v for _, v in m_ctrl]
    order = sorted(m_ctrl, key=lambda t: -t[1])
    print(f"  best-datapca : layer-mean {statistics.fmean(vals):+.4f}   "
          f"range [{min(vals):+.4f}, {max(vals):+.4f}]   min over layers {min(vals):+.4f}")
    if st:
        print(f"      ledger: at L{st['at'][1]} {st['at'][0]}, layer-mean {st['mean']}, "
              f"range {st['rng']}, rank {st['rank']}")
        want = st["at"][1]
        rank = [L for L, _ in order].index(want) + 1 if want in dict(m_ctrl) else None
        print(f"      measured at L{want}: "
              f"{dict(m_ctrl).get(want, float('nan')):+.4f}, rank {rank} of {len(vals)}")
    va = [v for _, v in m_axraw]
    print(f"  axes-raw     : layer-mean {statistics.fmean(va):+.4f}   positive at "
          f"{sum(1 for v in va if v > 0)}/{len(va)}")
    if st:
        print(f"      ledger: at L{st['at'][1]} {st['axes_raw_at']}, layer-mean "
              f"{st['axes_raw_mean']}, positive {st['axes_raw_pos']}")
        print(f"      measured at L{st['at'][1]}: "
              f"{dict(m_axraw).get(st['at'][1], float('nan')):+.4f}")
    vb = [v for _, v in m_bestraw]
    print(f"  best-raw     : layer-mean {statistics.fmean(vb):+.4f}   positive at "
          f"{sum(1 for v in vb if v > 0)}/{len(vb)}")
    if st:
        print(f"      ledger: layer-mean {st['best_raw']}, positive {st['best_raw_pos']}")

print("\n" + "=" * 96)
print("F-96's other quoted row: F-91's kd_s23 / axes  (peak L44 0.6435, L40 0.6304, +0.0131)")
print("=" * 96)
kd = {}
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    if (r["store"] == "kd11/banks/kd_s23" and r["pooling"] == "mean" and r["probe"] == "paper"
            and r["seed"] == "42" and r["feature"] in ("axes", "raw")):
        kd.setdefault(r["dataset_set"], {}).setdefault(int(r["layer"]), {})[r["feature"]] = \
            float(r["offdiag_mean"])
for ds, d in sorted(kd.items()):
    lays = sorted(d)
    ax = {L: d[L].get("axes") for L in lays if d[L].get("axes") is not None}
    if not ax:
        continue
    pk = max(ax, key=ax.get)
    print(f"  dataset_set={ds}: {len(ax)} layers; peak L{pk} {ax[pk]:.4f}; "
          f"L40 {ax.get(40, float('nan')):.4f}; peak-L40 "
          f"{ax[pk] - ax.get(40, float('nan')):+.4f}")
