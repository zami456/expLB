"""P12-briefing, local half: the claims that can be settled from git + the local tree.

O-006 / O-007  -- the collaborator's committed CSVs on origin/feat/persona-pc-ood, read with
                  `git show` (NO checkout).  Full-precision off_diagonal_mean, the identity with
                  the zero_shot column mean, the 8v7 / 12v13 arm definitions, the base-bank path,
                  and whether the ten extra roles really match kd/personas/kd_role_groups.json.
O-001          -- the three numbers in the briefing's Claim 1, from run-crossgen/kd11/matrices
                  (git copy; the lab copy is checked for identity by P12_lab.py).

Read-only.  Run from the repo root:  python audit/scripts/claims/P12_local.py
"""
import csv
import io
import json
import re
import statistics
import subprocess
import sys

COMMITS = ["f386a72", "origin/feat/persona-pc-ood"]
DD = "src/probes/deception-detection"


def show(commit, path):
    p = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True)
    if p.returncode:
        return None
    return p.stdout.decode("utf-8", "replace")


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def hr(t):
    print("=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- O-006 / O-007
hr("O-006 / O-007  -- the collaborator's own files")
for commit in COMMITS:
    sha = subprocess.run(["git", "rev-parse", commit], capture_output=True,
                         text=True).stdout.strip()
    print(f"\ncommit {commit} -> {sha}")
    for L in (20, 40):
        base = f"{DD}/results/persona_ood_ext_L{L}"
        s = show(commit, f"{base}/summary.csv")
        z = show(commit, f"{base}/zero_shot.csv")
        if s is None or z is None:
            print(f"  L{L}: MISSING")
            continue
        sm = {r["features"]: r for r in rows(s)}
        zr = [r for r in rows(z) if r.get("auroc_ext_axis")]  # drops the Alpaca control row
        ds = sorted({r["dataset"] for r in zr})
        # full precision, exactly as the file stores it
        raw_txt = sm["ext_axis1"]["off_diagonal_mean"]
        odm = float(raw_txt)
        zs_ext = statistics.fmean([float(r["auroc_ext_axis"]) for r in zr])
        zs_axis = statistics.fmean([float(r["auroc_axis"]) for r in zr])
        base_tr = float(sm["baseline"]["off_diagonal_mean"])
        print(f"  L{L}: n datasets = {len(ds)}  {ds}")
        print(f"       summary.csv ext_axis1 off_diagonal_mean (verbatim) = {raw_txt!r}")
        print(f"       mean(zero_shot.csv auroc_ext_axis)                 = {zs_ext!r}")
        print(f"       |difference|                                       = {abs(odm - zs_ext):.3e}")
        print(f"       summary.csv axis1 off_diagonal_mean = {sm['axis1']['off_diagonal_mean']}")
        print(f"       mean(zero_shot.csv auroc_axis)      = {zs_axis:.10f}"
              f"   |diff| = {abs(float(sm['axis1']['off_diagonal_mean']) - zs_axis):.3e}")
        print(f"       trained single-source baseline (row 'baseline')    = {base_tr:.4f}")
        print(f"       margin  paper 8v7  vs baseline = {zs_axis - base_tr:+.4f}")
        print(f"       margin  extended   vs baseline = {zs_ext - base_tr:+.4f}")
        print(f"       kd personas worth              = {zs_ext - zs_axis:+.4f}"
              f"   share = {(zs_ext - zs_axis) / (zs_ext - base_tr):.1%}")
        # drop repe_honesty, against the 5-dataset baseline (the reading the briefing uses)
        keep = [r for r in zr if not r["dataset"].startswith("repe_honesty")]
        a5 = statistics.fmean([float(r["auroc_axis"]) for r in keep])
        e5 = statistics.fmean([float(r["auroc_ext_axis"]) for r in keep])
        mb = show(commit, f"{base}/matrix_baseline.csv")
        if mb:
            rd = list(csv.reader(io.StringIO(mb)))
            hdr = rd[0][1:]
            off5 = []
            offall = []
            for r in rd[1:]:
                tr = r[0]
                for te, v in zip(hdr, r[1:]):
                    if te == tr or v == "":
                        continue
                    offall.append(float(v))
                    if not tr.startswith("repe_honesty") and not te.startswith("repe_honesty"):
                        off5.append(float(v))
            b5 = statistics.fmean(off5)
            print(f"       matrix_baseline off-diag over all 6 = {statistics.fmean(offall):.4f}"
                  f" (n={len(offall)})")
            print(f"       [drop repe_honesty] baseline over the 5 = {b5:.4f} (n={len(off5)})"
                  f"   axis1 {a5:.4f} margin {a5 - b5:+.4f}   kd worth {e5 - a5:+.4f}")

# arm definitions + role names -------------------------------------------------
hr("O-006 attribution: extended_arms(), the base bank, and the role names")
for cand in (f"{DD}/deception_detection/scripts/persona_ood.py", f"{DD}/scripts/persona_ood.py"):
    src = show("origin/feat/persona-pc-ood", cand)
    if src:
        print(f"  persona_ood.py found at: {cand}"
              "   (the briefing writes `scripts/persona_ood.py`)")
        break
if src:
    m = re.search(r"def extended_arms.*?(?=\ndef |\Z)", src, re.S)
    if m:
        print("\n  --- extended_arms() ---")
        for line in m.group(0).splitlines()[:24]:
            print("   ", line)

pcs = show("origin/feat/persona-pc-ood", f"{DD}/deception_detection/persona_pcs.py")
his = {}
if pcs:
    lines = pcs.splitlines()
    print("\n  deception_detection/persona_pcs.py lines 29-32 (the briefing cites :31)")
    for i in range(28, min(32, len(lines))):
        print(f"    {i+1:>3}: {lines[i]}")
    print("\n  persona group tuples in HIS persona_pcs.py:")
    for name in ("DECEPTION_PERSONAS", "HONESTY_PERSONAS", "SYCOPHANCY_PERSONAS",
                 "ANTI_SYCOPHANCY_PERSONAS", "KNOWLEDGE_DENIAL_PERSONAS",
                 "KNOWLEDGE_CANDOUR_PERSONAS"):
        m = re.search(rf"^{name}\s*=\s*\((.*?)\)", pcs, re.S | re.M)
        if m:
            names = re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
            his[name] = names
            print(f"    {name:<28} n={len(names):<3} {sorted(names)}")
    i = pcs.find('axes["extended"]')
    print("\n  where the extended axis is actually built (persona_pcs.py):")
    print("   ", pcs[i:i + 160].replace("\n", "\n    "))

try:
    gj = json.load(open("kd/personas/kd_role_groups.json", encoding="utf-8"))
    groups = {k: (v["roles"] if isinstance(v, dict) else v) for k, v in gj["groups"].items()}
    print("\n  kd/personas/kd_role_groups.json groups:")
    for g, v in groups.items():
        print(f"    {g:<22} n={len(v):<3} {sorted(v)}")
    ten = set(his.get("KNOWLEDGE_DENIAL_PERSONAS", [])) | set(
        his.get("KNOWLEDGE_CANDOUR_PERSONAS", []))
    ours = set(groups.get("knowledge_denial", [])) | set(
        groups.get("sincere_ignorance", [])) | set(groups.get("plain_assertion", []))
    print(f"\n  his 10 extra roles == our kd bank's 10 new roles ? {ten == ours}"
          f"  (his {len(ten)}, ours {len(ours)}, symmetric difference {sorted(ten ^ ours)})")
    print(f"  his KNOWLEDGE_DENIAL_PERSONAS == our `knowledge_denial` ? "
          f"{set(his.get('KNOWLEDGE_DENIAL_PERSONAS', [])) == set(groups.get('knowledge_denial', []))}")
    print("  his KNOWLEDGE_CANDOUR_PERSONAS == our `sincere_ignorance` + `plain_assertion` ? "
          f"{set(his.get('KNOWLEDGE_CANDOUR_PERSONAS', [])) == set(groups.get('sincere_ignorance', [])) | set(groups.get('plain_assertion', []))}"
          "   <- two owner groups MERGED into one arm")
except Exception as e:  # noqa: BLE001
    print("  kd_role_groups.json:", e)

# ------------------------------------------------------------------------ O-001
hr("O-001  -- Claim 1's three numbers, from the git copy of kd11/matrices")
P = "run-crossgen/kd11/matrices/mean_L40_s42.json"
d = json.load(open(P, encoding="utf-8"))
ds = d["datasets"]
print(f"  file {P}")
print(f"  bank={d['bank']}  bank_name={d['bank_name']}  k={d['k']}  control={d['control']}"
      f"  pooling={d['pooling']}  layer={d['layer']}  seed={d['seed']}")
print(f"  zero-shot directions stored in this file: {[z['direction'] for z in d['zeroshot']]}")
SY, KR = ds.index("sycophancy_mt"), ds.index("harm-pressure-knowledge-report")
for z in d["zeroshot"]:
    print(f"    zero-shot {z['direction']:<10} sycophancy_mt={z['auroc'][SY]:.4f}"
          f"   harm-pressure-knowledge-report={z['auroc'][KR]:.4f}")


def col(feat, tgt, probe="paper"):
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][i][tgt] for i in range(len(ds))
                 if i != tgt and r["auroc"][i][tgt] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


def row(feat, src_i, probe="paper"):
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][src_i][t] for t in range(len(ds))
                 if t != src_i and r["auroc"][src_i][t] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


print("\n  TRAINED (paper probe), sycophancy_mt as TARGET = column mean over 10 sources:")
for f in ("axes", "pc:deception", "pc:unified", "raw", "datapca", "length", "random"):
    m, n = col(f, SY)
    print(f"    {f:<14} {m:.4f}  (n={n})")
print("  TRAINED, sycophancy_mt as SOURCE = row mean:")
for f in ("axes", "raw", "random"):
    m, n = row(f, SY)
    print(f"    {f:<14} {m:.4f}  (n={n})")
print("  TRAINED, harm-pressure-knowledge-report:")
for f in ("axes", "raw", "random"):
    print(f"    {f:<14} as TARGET {col(f, KR)[0]:.4f}   as SOURCE {row(f, KR)[0]:.4f}")

print("\n  27_crossgen.py: which axis the bare `contrast`/`PC1` rows use")
try:
    cg = open("src/assistant_axis_ndif/27_crossgen.py", encoding="utf-8").read()
    i = cg.index("_cands = [n for n in")
    print("   ", cg[i:i + 320].replace("\n", "\n    "))
except Exception as e:  # noqa: BLE001
    print("   ", e)

sys.stdout.flush()
