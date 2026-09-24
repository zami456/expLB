"""READ-ONLY on the lab PC. Verifies F-102's AMENDED truncation check: response_len vs the
max_length cap in each rung's coords store, over the paper's five datasets. Writes nothing."""
import json, os, statistics

R = r"C:\Users\user4\expLB"
FIVE = ["instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"]
STORES = [("3B", os.path.join(R, "run-scale", "3b", "coords"), 1024),
          ("8B", os.path.join(R, "run-scale", "8b", "coords"), 1024),
          ("70B", os.path.join(R, "run-coords", "passB"), 8192)]

for lab, base, cap in STORES:
    if not os.path.isdir(base):
        print(f"{lab}: MISSING {base}")
        continue
    lens, plens, rows, atcap, near = [], [], 0, 0, 0
    missing = []
    for ds in FIVE:
        p = os.path.join(base, ds, "rows.jsonl")
        if not os.path.exists(p):
            missing.append(ds)
            continue
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            rows += 1
            rl = d.get("response_len")
            pl = d.get("prompt_len")
            if rl is not None:
                lens.append(rl)
            if pl is not None:
                plens.append(pl)
            tot = (pl or 0) + (rl or 0)
            if tot >= cap:
                atcap += 1
            elif tot >= cap - 8:
                near += 1
    lens.sort()
    def q(f):
        return lens[min(len(lens) - 1, int(f * len(lens)))] if lens else None
    print(f"{lab}: base={base} cap={cap} rows={rows} missing_datasets={missing}")
    print(f"   at/over cap = {atcap} ({100*atcap/max(1,rows):.2f}%)   within 8 of cap = {near}")
    print(f"   response_len: n={len(lens)} nonzero={sum(1 for x in lens if x)} "
          f"median={q(0.5)} p90={q(0.9)} p99={q(0.99)} max={lens[-1] if lens else None}")
    print(f"   prompt_len:   n={len(plens)} nonzero={sum(1 for x in plens if x)} "
          f"median={statistics.median(plens) if plens else None} max={max(plens) if plens else None}")
    print(f"   fingerprint(sorted response_len) = {hash(tuple(lens))}")
