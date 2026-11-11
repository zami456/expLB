"""Does `user_stay` make Qwen play its weakest roles often enough to use them? (build_confirm.py)

Per weak role: fully-in-role (judge 3) count over its 20 items (5 screen + 15 confirm), the rate with a Wilson 90%
interval, and the projected count of 150 (a role enters the s3 bank at >= 20 of 150, i.e. >= 13.3%). Reference:
Llama-3.3's ORIGINAL rate for the role over all its 150 replies (ref_llama33_kd), i.e. what the thesis bank had.

    .venv-analysis/Scripts/python.exe -I xmodel/qwen_roleplay/analyse_confirm.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.645):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    items = [json.loads(l) for l in (HERE / "bank" / "confirm_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    roles = list(dict.fromkeys(i["role"] for i in items))
    group = {i["role"]: i["group"] for i in items}
    out = []
    print(f"{'role':26}{'group':18}{'screen':>8}{'confirm':>9}{'all 20':>8}{'rate':>7}{'90% CI':>14}{'proj/150':>10}"
          f"{'Llama-3.3 orig':>16}")
    for role in roles:
        s = json.loads((HERE / "runs_screen" / "user_stay" / "scores" / f"{role}.json").read_text(encoding="utf-8"))
        cf = HERE / "runs_confirm" / "user_stay" / "scores" / f"{role}.json"
        c = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {}
        ks, ns = sum(v == 3 for v in s.values()), len(s)
        kc, nc = sum(v == 3 for v in c.values()), len(c)
        k, n = ks + kc, ns + nc
        lo, hi = wilson(k, n)
        ll = json.loads((HERE / "ref_llama33_kd" / "scores" / f"{role}.json").read_text(encoding="utf-8"))
        lk, ln = sum(v == 3 for v in ll.values()), len(ll)
        row = dict(role=role, group=group[role], screen=f"{ks}/{ns}", confirm=f"{kc}/{nc}", k=k, n=n, rate=k / n,
                   ci90=[lo, hi], projected_of_150=round(150 * k / n), llama33_orig=f"{lk}/{ln}")
        out.append(row)
        print(f"{role:26}{group[role]:18}{row['screen']:>8}{row['confirm']:>9}{f'{k}/{n}':>8}{k / n:7.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>14}{row['projected_of_150']:10d}{f'{lk}/{ln} ({lk / ln:.2f})':>16}")
    below = [r["role"] for r in out if r["projected_of_150"] < 20]
    print(f"\nroles projected under 20 of 150 (would drop out of the s3 bank): {below or 'none'}")
    (HERE / "runs_confirm" / "analysis_confirm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
