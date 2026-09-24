#!/usr/bin/env python
"""M1 claim check F-108 -- the HP-KR / kr-matched row-level LEAK, and the LODO figures.

P12 (O-003) already re-derived the LOCO gains, so this script checks the parts it did not:
  (a) the row_id overlap between every pair of the eleven datasets in run-coords/passB,
  (b) the cross-dataset identical-response-text count,
  (c) the LODO gains and the HP-KR LODO->LOCO collapse in the clustered JSONs,
  (d) how much of the leak actually reaches the TRAINING pool under the script's own
      80/20 split (the entry says "every one of its 258 rows is in the training pool";
      the split makes that ~80%, so the exact figure is measured here).

READ-ONLY.
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

R = Path("C:/Users/user4/expLB")
PB = R / "run-coords" / "passB"
DATASETS = ["harm-pressure-choice", "harm-pressure-knowledge-report",
            "harm-pressure-kr-matched", "instructed-deception", "convincing-game",
            "insider-trading", "ai_liar", "roleplaying", "sandbagging_v2", "mask",
            "sycophancy_mt"]

ids, texts, order = {}, {}, {}
for k in DATASETS:
    p = PB / k / "rows.jsonl"
    if not p.exists():
        print(f"MISSING {p}")
        continue
    rid, txt, seq = set(), {}, []
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        rid.add(r["row_id"])
        seq.append(r["row_id"])
        t = (r.get("response_str") or "").strip()
        if t:
            txt.setdefault(t, 0)
            txt[t] += 1
    ids[k] = rid
    texts[k] = txt
    order[k] = seq
    print(f"{k:34} rows {len(seq):6}  distinct row_ids {len(rid):6}  distinct responses {len(txt):6}")

print("\n== (a) pairwise row_id overlap ==")
any_other = 0
for a, b in combinations([k for k in DATASETS if k in ids], 2):
    n = len(ids[a] & ids[b])
    if n:
        print(f"   {a:34} & {b:34}  shared row_ids {n}")
        if not {a, b} <= {"harm-pressure-knowledge-report", "harm-pressure-kr-matched"}:
            any_other += n
print(f"   pairs other than (HP-KR, kr-matched) sharing any row_id: {any_other}")
A, B = "harm-pressure-knowledge-report", "harm-pressure-kr-matched"
if A in ids and B in ids:
    sh = ids[A] & ids[B]
    print(f"   kr-matched rows {len(ids[B])}; of which also HP-KR row_ids {len(sh)} "
          f"= {100 * len(sh) / len(ids[B]):.1f}%")
    print(f"   HP-KR rows {len(ids[A])}; of which also kr-matched {len(sh)} "
          f"= {100 * len(sh) / len(ids[A]):.1f}%")

print("\n== (b) identical response text across datasets ==")
pairs = 0
for a, b in combinations([k for k in DATASETS if k in ids], 2):
    if {a, b} <= {A, B}:
        continue
    common = set(texts[a]) & set(texts[b])
    if common:
        pairs += len(common)
        print(f"   {a} & {b}: {len(common)} identical response strings")
        for t in sorted(common)[:5]:
            print(f"        {t[:110]!r}  (x{texts[a][t]} / x{texts[b][t]})")
print(f"   total cross-dataset identical response strings (excluding the HP-KR pair): {pairs}")

print("\n== (c) clustered_L*.json: LODO / LOCO gains and the HP-KR collapse ==")
for L in (20, 32, 40):
    f = R / "run-premise" / "clustered" / f"clustered_L{L}_mean_s42.json"
    if not f.exists():
        print(f"   MISSING {f}")
        continue
    j = json.loads(f.read_text(encoding="utf-8"))
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pd = j["per_dataset"]
    g_lodo = {k: v["pooled_lodo"] - v["single_offdiag"] for k, v in pd.items()}
    g_loco = {k: v["pooled_loco"] - v["single_offdiag"] for k, v in pd.items()}
    clean = [k for k in pd if k not in set(j["leaked"])]
    print(f"\n   L{L}  md5 {md5}  n_targets {len(pd)}  leaked {j['leaked']}")
    print(f"      stored summary : {json.dumps(j['summary'])}")
    print(f"      recomputed gain_lodo all  {np.mean(list(g_lodo.values())):+.4f}   "
          f"clean {np.mean([g_lodo[k] for k in clean]):+.4f}")
    print(f"      recomputed gain_loco all  {np.mean(list(g_loco.values())):+.4f}   "
          f"clean {np.mean([g_loco[k] for k in clean]):+.4f}")
    print(f"      lodo positive all {sum(v > 0 for v in g_lodo.values())}/{len(g_lodo)}   "
          f"loco positive all {sum(v > 0 for v in g_loco.values())}/{len(g_loco)}   "
          f"clean {sum(g_loco[k] > 0 for k in clean)}/{len(clean)}")
    for k in ("harm-pressure-knowledge-report", "harm-pressure-kr-matched"):
        if k in pd:
            v = pd[k]
            print(f"      {k:34} single {v['single_offdiag']:.4f}  lodo {v['pooled_lodo']:.4f}"
                  f"  loco {v['pooled_loco']:.4f}  ({v['n_lodo_datasets']}->{v['n_loco_datasets']} ds)")
    print("      per-target LOCO gains: " + ", ".join(
        f"{k.split('-')[-1][:12]} {g_loco[k]:+.3f}" for k in sorted(pd)))

print("\n== (d) how much of the leak reaches the TRAINING pool under the 80/20 split ==")
# replay 30g_clustered_holdout's split exactly: np.random.default_rng(42).permutation(n) per
# dataset, in the DATASETS order the script uses (CLUSTERS flattened), first 80% = train.
CL_ORDER = ["harm-pressure-choice", "harm-pressure-knowledge-report", "harm-pressure-kr-matched",
            "instructed-deception", "convincing-game", "insider-trading",
            "ai_liar", "roleplaying", "sandbagging_v2", "mask", "sycophancy_mt"]
present = [k for k in CL_ORDER if k in ids]
rng = np.random.default_rng(42)
splits = {}
for k in present:
    n = len(order[k])
    perm = rng.permutation(n)
    splits[k] = (perm[:int(0.8 * n)], perm[int(0.8 * n):])
    print(f"   {k:34} n {n:6}  train {len(splits[k][0]):6}  test {len(splits[k][1]):6}")
hp_train = {order[A][i] for i in splits[A][0]}
kr_test = [order[B][i] for i in splits[B][1]]
kr_train = [order[B][i] for i in splits[B][0]]
hit = sum(1 for r in kr_test if r in hp_train)
print(f"\n   kr-matched TEST rows {len(kr_test)}; of which in HP-KR's TRAIN split: {hit} "
      f"({100 * hit / max(1, len(kr_test)):.1f}%)")
kr_train_set = set(kr_train)
hp_test = [order[A][i] for i in splits[A][1]]
hit2 = sum(1 for r in hp_test if r in kr_train_set)
print(f"   HP-KR   TEST rows {len(hp_test)}; of which in kr-matched's TRAIN split: {hit2} "
      f"({100 * hit2 / max(1, len(hp_test)):.1f}%)")
print("   NOTE the permutation order depends on the dataset iteration order, so this replay is"
      " exact only if 9_persona_probes.build() returns the datasets in CLUSTERS order (it does:"
      " 30g passes [d for c in CLUSTERS.values() for d in c]).")
