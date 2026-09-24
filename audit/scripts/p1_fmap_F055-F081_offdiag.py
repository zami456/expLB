"""PRELIMINARY (Phase 1 mapping only) read of a crossgen matrix JSON on the lab PC, via the read-only peek tool.
Prints, for each requested feature (probe 'paper' unless --probe), the off-diagonal mean AUROC over all cells, and
optionally the column mean for one target dataset (--target). Also prints the zero-shot rows if --zeroshot.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_fmap_F055-F081_offdiag.py <lab_path> feat1 feat2 ... [--probe paper] [--target sycophancy_mt] [--source harm-pressure-kr-matched] [--zeroshot]
Local files (a path that exists on the laptop) are read directly instead.
Nothing is written anywhere. Values printed are PRELIMINARY; Phase 2 recomputes everything.
"""
import json, os, re, subprocess, sys

args = sys.argv[1:]
def opt(name, default=None, flag=False):
    if name in args:
        i = args.index(name)
        if flag:
            del args[i]; return True
        v = args[i + 1]; del args[i:i + 2]; return v
    return default
probe = opt("--probe", "paper"); target = opt("--target"); source = opt("--source"); zs = opt("--zeroshot", flag=True)
path, feats = args[0], args[1:]

def peek(mode, arg=""):
    out = subprocess.run([sys.executable, "audit/scripts/labpc_run.py", "audit/scripts/p1_labpc_runpy.ps1", "FILE=p1_peek.py",
                          f"ARGS={path} {mode} {arg}".strip()], capture_output=True, timeout=300)
    return out.stdout.decode("utf-8", "replace").replace("\x00", "")

if os.path.exists(path):
    d = json.load(open(path, encoding="utf-8"))
    res = d["results"]; ds = d["datasets"]; zsr = d.get("zeroshot", [])
    get = lambda i: res[i]
else:
    fl = [l.split('"feature": ')[1].strip().strip(',').strip('"') for l in peek("grep", '"feature"').splitlines() if '"feature": ' in l]
    pl = [l.split('"probe": ')[1].strip().strip(',').strip('"') for l in peek("grep", '"probe"').splitlines() if '"probe": ' in l]
    res = [{"feature": f, "probe": p} for f, p in zip(fl, pl)]
    ds = json.loads(re.sub(r"python exit.*", "", peek("get", "datasets"), flags=re.S))
    def get(i):
        return json.loads(re.sub(r"python exit.*", "", peek("get", f"results.{i}"), flags=re.S))
    zsr = []
    if zs:
        zsr = json.loads(re.sub(r"python exit.*", "", peek("get", "zeroshot"), flags=re.S))
print("datasets:", ds)
n = len(ds)
for f in feats:
    idx = [i for i, r in enumerate(res) if r["feature"] == f and r["probe"] == probe]
    if not idx:
        print(f"{f}|{probe}: NOT IN FILE"); continue
    a = get(idx[0])["auroc"]
    off = [a[i][j] for i in range(n) for j in range(n) if i != j and a[i][j] is not None]
    line = f"{f}|{probe}: offdiag_mean={sum(off)/len(off):.4f} (cells={len(off)}) results[{idx[0]}]"
    if target and target in ds:
        t = ds.index(target)
        col = [a[i][t] for i in range(n) if i != t and a[i][t] is not None]
        line += f"  col[{target}] mean={sum(col)/len(col):.4f} (n={len(col)})"
    if source and source in ds:
        s_ = ds.index(source)
        row = [a[s_][j] for j in range(n) if j != s_ and a[s_][j] is not None]
        line += f"  row[{source}] mean={sum(row)/len(row):.4f} (n={len(row)})"
    print(line)
for z in zsr:
    if target and target in ds:
        print(f"zeroshot {z['direction']}: {target}={z['auroc'][ds.index(target)]:.4f}")
    else:
        print(f"zeroshot {z['direction']}: {[round(x,4) if x is not None else None for x in z['auroc']]}")
