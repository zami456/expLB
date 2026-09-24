"""Phase 6 / 6A LEVEL 2 — re-derive the draft's tables from the collaborator's raw result files.

Covers batches B41 (Table 2), B42 (Table 3), B43 (Table 4), B45 (Table 6),
B46 (Table 7), B40 (Table 11).  B23 (Abstract) and B30 (App F) are handled
alongside because their numbers are drawn from the same files.

Source: `origin/paper`, src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}/
        read with `git show` -- no checkout, no branch change.

  zero_shot.csv            per-dataset zero-shot AUROC, one column per arm
  summary.csv              off_diagonal_mean / off_diagonal_excl_hpkr per feature
  zero_shot_operating.csv  BA@1%FPR, recall, AUROC at a calibrated threshold
  liars_bench_comparison.csv   the LB-protocol comparison row block
  matrix_*.csv             full train-one-test-all matrices

Everything printed is rounded to 3 dp for comparison against the draft, with the
full-precision value beside it.
"""
import csv, io, subprocess, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = "origin/paper"
BASE = "src/probes/deception-detection/results"
LAYERS = [20, 40, 44]


def show(path):
    r = subprocess.run(["git", "show", f"{REF}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout


def table(path):
    t = show(path)
    if t is None:
        return None
    return list(csv.DictReader(io.StringIO(t)))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def r3(x):
    return "     -" if x is None else f"{x:.3f}"


# dataset key -> short label used in the draft's tables
DS = {
    "convince_me__plain": "CG",
    "harm_refusal_choice__plain": "HP-C",
    "harm_refusal_knowledge_report__plain": "HP-KR",
    "insider_trading_cadenza__plain": "IT",
    "instruct_dishonesty__plain": "ID",
    "repe_honesty__plain": "RepE",
}
# the draft's three zero-shot arms, in the column order its tables use
ARMS = [("8v7", "auroc_axis"), ("4v6", "auroc_denial_axis"), ("12v13", "auroc_ext_axis")]

ZS, SUM, OP, LB = {}, {}, {}, {}
for L in LAYERS:
    d = f"{BASE}/persona_ood_ext_L{L}"
    ZS[L] = table(f"{d}/zero_shot.csv")
    SUM[L] = table(f"{d}/summary.csv")
    OP[L] = table(f"{d}/zero_shot_operating.csv")
    LB[L] = table(f"{d}/liars_bench_comparison.csv")
    if ZS[L] is None:
        sys.exit(f"missing zero_shot.csv at L{L}")

print("=" * 78)
print("SOURCE: origin/paper  " + BASE + "/persona_ood_ext_L{20,40,44}/")
print("=" * 78)

# ---------------------------------------------------------------- Table 6 / B45
print("\n\n### TABLE 6 (B45, D-182..D-188) -- zero-shot AUROC per dataset")
print("    arms in draft column order: 8v7 | 4v6 | 12v13\n")
hdr = f"{'dataset':7s}"
for L in LAYERS:
    hdr += f" | {'L'+str(L):^22s}"
print(hdr)
print(f"{'':7s}" + (" | " + "  8v7    4v6   12v13 ") * 3)
for key, short in DS.items():
    line = f"{short:7s}"
    for L in LAYERS:
        row = next((x for x in ZS[L] if x["dataset"] == key), None)
        vals = [f(row[c]) if row else None for _, c in ARMS]
        line += " | " + " ".join(r3(v) for v in vals)
    print(line)

# Table 6's "Mean" row -- the draft prints a mean; check BOTH candidate means
print("\n  'Mean' row candidates:")
for L in LAYERS:
    s = SUM[L]
    off = {x["features"]: f(x["off_diagonal_mean"]) for x in s}
    six = {}
    for name, col in ARMS:
        vs = [f(next(x for x in ZS[L] if x["dataset"] == k)[col]) for k in DS]
        six[name] = sum(vs) / len(vs)
    feat = {"8v7": "axis1", "4v6": "denial_axis1", "12v13": "ext_axis1"}
    print(f"   L{L}: mean of the 6 zero-shot cells = "
          + " ".join(f"{n}:{r3(six[n])}" for n, _ in ARMS))
    print(f"        summary.csv off_diagonal_mean = "
          + " ".join(f"{n}:{r3(off.get(feat[n]))}" for n, _ in ARMS))

# ---------------------------------------------------------------- Table 11 / B40
print("\n\n### TABLE 11 (B40, D-226..D-232) -- sycophancy contrast, zero-shot")
for key, short in DS.items():
    line = f"{short:7s}"
    for L in LAYERS:
        row = next((x for x in ZS[L] if x["dataset"] == key), None)
        line += f" | syc_axis {r3(f(row['auroc_syc_axis']) if row else None)}"
    print(line)
print("  off-diagonal mean (syc_axis1, from summary.csv):")
for L in LAYERS:
    v = next((f(x["off_diagonal_mean"]) for x in SUM[L] if x["features"] == "syc_axis1"), None)
    ex = next((f(x["off_diagonal_excl_hpkr"]) for x in SUM[L] if x["features"] == "syc_axis1"), None)
    print(f"   L{L}: all-six {r3(v)}   excl-HP-KR {r3(ex)}")

# ---------------------------------------------------------------- Table 3 / B42
print("\n\n### TABLE 3 (B42, D-109..D-116) -- mean off-diagonal")
FEATMAP = [
    ("D-109", "raw activations (8192-d)", "baseline"),
    ("D-110", "published contrast 8v7", "axis1"),
    ("D-111", "knowledge-denial 4v6", "denial_axis1"),
    ("D-112", "extended contrast 12v13", "ext_axis1"),
    ("D-113", "extended PCs k=3", "ext_pc3"),
    ("D-114", "extended PC1", "ext_pc1"),
    ("D-115", "published PCs k=3", "pc3"),
    ("D-116", "random directions k=3", "random3"),
]
print(f"{'claim':7s} {'feature':26s} {'dim':>4s} |     all six (20/40/44)     |   excl HP-KR (20/40/44)")
for cid, label, feat in FEATMAP:
    allsix, excl, dim = [], [], ""
    for L in LAYERS:
        row = next((x for x in SUM[L] if x["features"] == feat), None)
        allsix.append(f(row["off_diagonal_mean"]) if row else None)
        excl.append(f(row["off_diagonal_excl_hpkr"]) if row else None)
        if row:
            dim = row.get("dim", "")
    print(f"{cid:7s} {label:26s} {dim:>4s} | " + " ".join(r3(v) for v in allsix)
          + "      | " + " ".join(r3(v) for v in excl))

# ---------------------------------------------------------------- Table 4 / B43
print("\n\n### TABLE 4 (B43, D-140..D-151) -- LB protocol, Average (Liars' Bench)")
ARMS4 = [
    ("base probe (RepE-trained)", None),
    ("published contrast (8v7)", "deception axis (8v7)"),
    ("knowledge-denial (4v6)", None),
    ("extended contrast (12v13)", None),
]
for L in LAYERS:
    print(f"\n  --- L{L} ---")
    if OP[L] is None:
        print("    zero_shot_operating.csv MISSING")
        continue
    seen = []
    for row in OP[L]:
        if row["dataset"] != "Average (Liars' Bench)":
            continue
        key = (row["arm"], row["threshold_source"])
        if key in seen:
            continue
        seen.append(key)
        print(f"    {row['arm']:34s} [{row['threshold_source']:18s}] "
              f"BA {r3(f(row['balacc']))}  recall {r3(f(row['recall']))}  "
              f"AUROC {r3(f(row['auroc']))}")

# ---------------------------------------------------------------- Table 7 / B46
print("\n\n### TABLE 7 (B46, D-189..D-194) -- raw vs 12v13, per TEST dataset")
print("  raw = matrix_baseline.csv column mean over the 5 OFF-diagonal training rows")
print("  12v13 = zero-shot ext_axis (a 1-d arm: every training row is identical)\n")
for L in LAYERS:
    d = f"{BASE}/persona_ood_ext_L{L}"
    mb = table(f"{d}/matrix_baseline.csv")
    me = table(f"{d}/matrix_ext_axis1.csv")
    if mb is None:
        print(f"  L{L}: matrix_baseline.csv missing")
        continue
    first = list(mb[0].keys())[0]
    print(f"  --- L{L} ---   (row key column = {first!r})")
    for key, short in DS.items():
        col = next((c for c in mb[0] if c == key), None)
        if col is None:
            continue
        offd = [f(r[col]) for r in mb if r[first] != key and f(r[col]) is not None]
        rawm = sum(offd) / len(offd) if offd else None
        ecol = [f(r[col]) for r in me if r[first] != key and f(r[col]) is not None] if me else []
        extm = sum(ecol) / len(ecol) if ecol else None
        zs = next((f(x["auroc_ext_axis"]) for x in ZS[L] if x["dataset"] == key), None)
        print(f"    {short:7s} raw(off-diag col mean) {r3(rawm)}   "
              f"12v13 trained {r3(extm)}   12v13 zero-shot {r3(zs)}")

# ---------------------------------------------------------------- B23 / abstract
print("\n\n### ABSTRACT (B23) -- liars_bench_comparison.csv, the 0.50/0.26/0.95 row block")
for L in LAYERS:
    if LB[L] is None:
        continue
    print(f"  --- L{L} ---")
    for row in LB[L]:
        print(f"    {row['method']:34s} src={row['source']:18s} "
              f"priv={row['privileged']:5s} BA@1%FPR {r3(f(row['balacc_1pct_fpr']))} "
              f"AUROC {r3(f(row['auroc']))} n={row['n_datasets']}")

print("\n" + "=" * 78)
print("done")
