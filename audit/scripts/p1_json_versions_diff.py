"""Compare the lab (current) and git (asix HEAD) versions of a JSON result file: which numeric leaves differ, and by how much.
Usage: python audit/scripts/p1_json_versions_diff.py <relpath under repo root> [...]  (lab copy read from audit/cache/labpc/)"""
import json, subprocess, sys
def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, o
for rel in sys.argv[1:]:
    lab = json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))
    git = json.loads(subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True).stdout.decode("utf-8"))
    L, G = dict(leaves(lab)), dict(leaves(git))
    only_l, only_g = set(L) - set(G), set(G) - set(L)
    diff = [(k, G[k], L[k]) for k in set(L) & set(G) if L[k] != G[k]]
    num = [(k, g, l) for k, g, l in diff if isinstance(g, (int, float)) and isinstance(l, (int, float)) and not isinstance(g, bool)]
    mx = max((abs(l - g) for k, g, l in num), default=0)
    print(f"== {rel}: leaves lab={len(L)} git={len(G)} | only-lab={len(only_l)} only-git={len(only_g)} | differing={len(diff)} (numeric {len(num)}, max |delta|={mx:.6g})")
    for k, g, l in sorted(diff, key=lambda x: -abs(x[2] - x[1]) if x in num else 0)[:6]:
        print(f"    {k[:90]}: git={str(g)[:40]} lab={str(l)[:40]}")
    for k in sorted(only_l)[:3]: print("    only-lab:", k[:100])
    for k in sorted(only_g)[:3]: print("    only-git:", k[:100])
