"""Audit Phase 2 / W5A-ledger - follow-ups 2 (read-only)."""
import glob, json, os, re, datetime
E = "C:/Users/user4/expLB"

print("== F-69: chunk bytes written on 2026-09-21 only (the '5.22 GB' claim) ==")
for store in ["passA", "passC"]:
    p = "%s/run-coords/%s/harm-pressure-knowledge-report" % (E, store)
    ch = glob.glob(p + "/chunk_*")
    d21 = [c for c in ch if datetime.datetime.fromtimestamp(os.path.getmtime(c)).date()
           == datetime.date(2026, 9, 21)]
    print("   %s: %d of %d chunks dated 09-21, %.2f GB (all %d chunks = %.2f GB)" %
          (store, len(d21), len(ch), sum(os.path.getsize(c) for c in d21) / 1e9,
           len(ch), sum(os.path.getsize(c) for c in ch) / 1e9))
print("   -- logs/hpkr_topup_passA.log tail --")
try:
    L = open(E + "/logs/hpkr_topup_passA.log", encoding="utf-8", errors="replace").read().splitlines()
except Exception:
    L = open(E + "/logs/hpkr_topup_passA.log", encoding="utf-16", errors="replace").read().splitlines()
for line in L[-8:]:
    print("    ", line[:220])
print()
print("== F-21: retries / session_batches settings in the 3B chain ==")
for p in glob.glob(E + "/scale/scripts/*.ps1"):
    for i, line in enumerate(open(p, encoding="utf-8", errors="replace").read().splitlines(), 1):
        if re.search(r"retries|session_batches|gen_batch", line):
            print("   %s:%d  %s" % (os.path.basename(p), i, line.strip()[:200]))
print()
print("== F-57: the kd_s23 significance.json on the lab NOW ==")
p = E + "/run-crossgen/kd11/banks/kd_s23/significance.json"
st = os.stat(p)
print("   %s  %d B (%.2f MB)  mtime %s" % (p, st.st_size, st.st_size / 1e6,
                                           datetime.datetime.fromtimestamp(st.st_mtime)))
s = json.load(open(p, encoding="utf-8"))
tests = s if isinstance(s, list) else s.get("tests")
if isinstance(s, dict) and tests is None:
    print("   keys:", list(s.keys())[:12])
    for k in list(s.keys())[:6]:
        v = s[k]
        print("    ", k, type(v).__name__, len(v) if hasattr(v, "__len__") else v)
if isinstance(tests, list):
    print("   n tests:", len(tests), " layers:", sorted(set(t.get("layer") for t in tests)))
    poolings = sorted(set(t.get("pooling") for t in tests))
    print("   poolings:", poolings, " probes:", sorted(set(t.get("probe") for t in tests)))
    sel = [t for t in tests if t.get("pooling") == "mean" and t.get("probe") == "paper"
           and t.get("comparison") == "pc:unified - raw"]
    print("   pc:unified - raw, mean/paper rows: %d" % len(sel))
    for t in sorted(sel, key=lambda x: x.get("layer")):
        print("     L%-3s diff=%+.4f ci=[%+.4f,%+.4f] q=%s pairs=%s" %
              (t.get("layer"), t.get("diff"), t.get("dataset_ci95", [float('nan')] * 2)[0],
               t.get("dataset_ci95", [float('nan')] * 2)[1], t.get("dataset_q_bh"), t.get("pairs")))
