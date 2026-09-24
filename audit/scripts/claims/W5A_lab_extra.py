"""Audit Phase 2 / W5A-ledger - follow-ups (read-only)."""
import glob, json, os, re, datetime, hashlib
E = "C:/Users/user4/expLB"

d = json.load(open(E + "/run-premise/audit/audit_L40_s42.json", encoding="utf-8"))
print("== F-90 per-target: mean_of_directions - pooled_lodo_full ==")
t = d["targets"]; a = d["per_target"]["mean_of_directions"]; b = d["per_target"]["pooled_lodo_full"]
w = 0
for i, name in enumerate(t):
    dd = a[i] - b[i]
    w += dd > 0
    print("   %-35s %+0.4f   (mod %.4f  lodo %.4f)" % (name, dd, a[i], b[i]))
print("   wins: %d of %d" % (w, len(t)))
print()
print("== F-90 pairs below 0.5 / 0.4 and the minimum ==")
pr = d["pairs"]
lo = sorted(pr, key=lambda x: x["auroc"])
print("   n_pairs=%d  below 0.5: %d   below 0.4: %d" %
      (len(pr), sum(1 for x in pr if x["auroc"] < 0.5), sum(1 for x in pr if x["auroc"] < 0.4)))
for x in lo[:3]:
    print("   min: %s -> %s = %.4f" % (x["source"], x["target"], x["auroc"]))
print()
print("== F-90: how 30f_result2_audit.py gets its persona bank ==")
s = E + "/src/assistant_axis_ndif/30f_result2_audit.py"
for i, line in enumerate(open(s, encoding="utf-8", errors="replace"), 1):
    if re.search(r"axes|bank|feature|persona|import|argparse|add_argument", line, re.I):
        print("   %4d| %s" % (i, line.rstrip()[:180]))
print()
print("== run/axes.pt contents ==")
try:
    import torch
    o = torch.load(E + "/run/axes.pt", map_location="cpu", weights_only=False)
    def desc(x, pre="", depth=0):
        if depth > 2: return
        if isinstance(x, dict):
            for k, v in list(x.items())[:40]:
                if hasattr(v, "shape"):
                    print("   %s%s: tensor%s %s" % ("  " * depth, pre + str(k), tuple(v.shape), v.dtype))
                elif isinstance(x, dict) and isinstance(v, (str, int, float)):
                    print("   %s%s = %s" % ("  " * depth, pre + str(k), str(v)[:160]))
                elif isinstance(v, (list, tuple)):
                    print("   %s%s: list[%d] %s" % ("  " * depth, pre + str(k), len(v), str(v[:12])[:220]))
                else:
                    print("   %s%s:" % ("  " * depth, pre + str(k)))
                    desc(v, depth=depth + 1)
    desc(o)
except Exception as exc:
    print("   torch load failed:", exc)
print()
print("== F-69 extra: passA HP-KR activation chunks (the '55 chunks / 5.22 GB' claim) ==")
for store in ["passA", "passC"]:
    p = "%s/run-coords/%s/harm-pressure-knowledge-report" % (E, store)
    if not os.path.isdir(p):
        print("  ", p, "MISSING"); continue
    ch = glob.glob(p + "/chunk_*")
    tot = sum(os.path.getsize(c) for c in ch)
    mt = [os.path.getmtime(c) for c in ch]
    print("   %s: %d chunks, %.2f GB, %s .. %s" %
          (store, len(ch), tot / 1e9,
           datetime.datetime.fromtimestamp(min(mt)) if mt else "-",
           datetime.datetime.fromtimestamp(max(mt)) if mt else "-"))
    print("     all files:", sorted(os.listdir(p))[:12])
