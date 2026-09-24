"""READ-ONLY peek at a file on the lab PC (runs there with the lab's python). Never writes anything.
Usage (from the laptop):
  python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_peek.py "ARGS=<path> <mode> [arg]"
Paths use forward slashes, e.g. C:/Users/user4/expLB/run-premise/six/summary.json (no spaces allowed in the path).
Modes:
  info            size, mtime, md5
  head [N]        first N characters (default 3000)
  cat             whole file (refuses > 400 KB; use head/grep/keys)
  keys [depth]    JSON structure: keys and value types down to depth (default 3); lists show length + first item
  get <a.b.0.c>   JSON value at a dotted path (list indices as numbers), pretty-printed (truncated at 20 KB)
  grep <regex>    matching lines with line numbers (max 200)
  ls              directory listing (name, size, mtime)
  pt              torch .pt: keys, tensor shapes/dtypes, small scalars (weights_only load)
"""
import datetime, hashlib, json, os, re, sys

p = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "info"; arg = sys.argv[3] if len(sys.argv) > 3 else None
if mode == "ls":
    for e in sorted(os.scandir(p), key=lambda e: e.name):
        st = e.stat()
        print(("D " if e.is_dir() else "F ") + f"{st.st_size:>12} {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M} {e.name}")
    sys.exit()
st = os.stat(p)
if mode == "info":
    print(f"{p}\nbytes {st.st_size}\nmtime {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}\nmd5 {hashlib.md5(open(p, 'rb').read()).hexdigest()}")
elif mode == "head":
    print(open(p, encoding="utf-8", errors="replace").read(int(arg or 3000)))
elif mode == "cat":
    if st.st_size > 400_000: sys.exit("file > 400 KB: use head/grep/keys/get")
    print(open(p, encoding="utf-8", errors="replace").read())
elif mode in ("keys", "get"):
    d = json.load(open(p, encoding="utf-8"))
    if mode == "get":
        for k in (arg or "").split(".") if arg else []:
            d = d[int(k)] if isinstance(d, list) else d[k]
        print(json.dumps(d, indent=1, ensure_ascii=False)[:20000])
    else:
        depth = int(arg or 3)
        def show(o, ind, dep):
            if isinstance(o, dict):
                for k, v in list(o.items())[:60]:
                    t = type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict)) else f"={v!r}"[:60])
                    print("  " * ind + f"{k}: {t}")
                    if dep > 1 and isinstance(v, (dict, list)): show(v, ind + 1, dep - 1)
                if len(o) > 60: print("  " * ind + f"... {len(o) - 60} more keys")
            elif isinstance(o, list) and o:
                print("  " * ind + f"[0 of {len(o)}]: {type(o[0]).__name__}")
                if dep > 1: show(o[0], ind + 1, dep - 1)
        show(d, 0, depth)
elif mode == "grep":
    rx = re.compile(arg); n = 0
    for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
        if rx.search(line):
            print(f"{i}: {line.rstrip()[:400]}"); n += 1
            if n >= 200: break
elif mode == "pt":
    import torch
    o = torch.load(p, map_location="cpu", weights_only=True)
    def show(o, ind=0):
        if isinstance(o, dict):
            for k, v in o.items():
                if hasattr(v, "shape"): print("  " * ind + f"{k}: tensor{tuple(v.shape)} {v.dtype}")
                elif isinstance(v, (dict, list)) and len(str(v)) > 200: print("  " * ind + f"{k}: {type(v).__name__}[{len(v)}]"); show(v, ind + 1) if isinstance(v, dict) else None
                else: print("  " * ind + f"{k}: {v!r}"[:300])
        elif hasattr(o, "shape"): print(f"tensor{tuple(o.shape)} {o.dtype}")
        else: print(repr(o)[:2000])
    show(o)
