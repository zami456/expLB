"""Verify batch B / F-61: bank composition counts, recomputed from the rollout files themselves.

Reads the audit cache copies of the lab rollouts (md5-verified transfers, audit/cache/MANIFEST.csv):
  audit/cache/labpc/run-paper16/responses/*.jsonl   (paper16 bank)
  audit/cache/labpc/run-kd/responses/*.jsonl        (kd bank)

Prints, per bank: number of role files, rows per role (histogram), distinct questions,
distinct prompt_index (variants), and the grand total of rollouts.
"""
import glob, json, os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), "..", "cache", "labpc")


def bank(name, path):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    print("=" * 72)
    print("%s : %d role files in %s" % (name, len(files), path))
    rows_per = {}
    q_all, v_all = set(), set()
    per_role_q, per_role_v = {}, {}
    total = 0
    for f in files:
        role = os.path.basename(f)[:-6]
        qs, vs, n = set(), set(), 0
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            qs.add(d.get("question"))
            vs.add(str(d.get("prompt_index")))
        rows_per[role] = n
        per_role_q[role] = len(qs)
        per_role_v[role] = len(vs)
        q_all |= qs
        v_all |= vs
        total += n
    print("  rows-per-role histogram:", sorted(Counter(rows_per.values()).items()))
    print("  distinct questions per role:", sorted(Counter(per_role_q.values()).items()))
    print("  distinct prompt_index per role:", sorted(Counter(per_role_v.values()).items()))
    print("  union of questions across roles: %d ; union of prompt_index: %d (%s)"
          % (len(q_all), len(v_all), sorted(v_all)[:8]))
    print("  TOTAL rollouts: %d" % total)
    odd = {k: v for k, v in rows_per.items() if v != Counter(rows_per.values()).most_common(1)[0][0]}
    if odd:
        print("  roles NOT at the modal row count:", odd)
    first = files[0]
    print("  FIRST role file (alphabetical) = %s -> %d rows, %d questions, %d variants"
          % (os.path.basename(first), rows_per[os.path.basename(first)[:-6]],
             per_role_q[os.path.basename(first)[:-6]], per_role_v[os.path.basename(first)[:-6]]))
    return total, len(files), q_all


t16, n16, q16 = bank("paper16", os.path.join(BASE, "run-paper16", "responses"))
tkd, nkd, qkd = bank("kd", os.path.join(BASE, "run-kd", "responses"))
print("=" * 72)
print("question-set overlap paper16 vs kd: %d shared / %d paper16 / %d kd"
      % (len(q16 & qkd), len(q16), len(qkd)))
print("ledger says: paper16 26 roles x 10 q x 5 var = 50/role, 1400 total;"
      " kd 36 roles x 30 q x 5 var = 150/role, 5400 total")
