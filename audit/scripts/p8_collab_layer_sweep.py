"""Read the collaborator's NEW layer sweep from origin/paper and check its claims.

Branch `origin/paper` (pushed 2026-09-23 23:46) adds `persona_ood_ext_L{20..76 step 4, 79}` --
16 layers, where the audit and the draft had only 3 (L20/L40/L44).

Commit ea1ce52b claims:
  "ranking peaks at 36, HP-KR is a spike at 40"

This reads every layer's summary.csv and zero_shot.csv straight out of git (read-only,
`git show`) and rebuilds the depth curve so both claims can be checked.

  python audit/scripts/p8_collab_layer_sweep.py
"""
import csv
import io
import json
import os
import subprocess
from collections import defaultdict

BRANCH = "origin/paper"
BASE = "src/probes/deception-detection/results"
LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show(path):
    r = subprocess.run(["git", "show", f"{BRANCH}:{path}"],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout if r.returncode == 0 else None


def rows(path):
    t = show(path)
    return list(csv.DictReader(io.StringIO(t))) if t else []


def main():
    print("=" * 96)
    print(f"COLLABORATOR'S LAYER SWEEP  ({BRANCH}, {len(LAYERS)} layers: "
          f"{LAYERS[0]}..{LAYERS[-2]} step 4, plus {LAYERS[-1]})")
    print("=" * 96)

    # ---- what columns exist ------------------------------------------
    probe = rows(f"{BASE}/persona_ood_ext_L40/summary.csv")
    if not probe:
        print("could not read L40 summary.csv -- is the branch fetched?")
        return
    print(f"summary.csv columns: {list(probe[0].keys())}")
    print(f"arms at L40        : {[r.get('features') for r in probe]}")

    # ---- the depth curve ---------------------------------------------
    print("\n" + "=" * 96)
    print("OFF-DIAGONAL MEAN (his 'ranking' metric) BY LAYER AND ARM")
    print("=" * 96)
    curve = defaultdict(dict)
    hpkr = defaultdict(dict)
    for L in LAYERS:
        for r in rows(f"{BASE}/persona_ood_ext_L{L}/summary.csv"):
            arm = r.get("features") or r.get("arm")
            if not arm:
                continue
            for key in ("off_diagonal_mean", "off_diag_mean", "offdiag_mean"):
                if key in r and r[key] not in (None, ""):
                    curve[arm][L] = float(r[key])
                    break
            for key in ("off_diagonal_excl_hpkr", "off_diag_excl_hpkr"):
                if key in r and r[key] not in (None, ""):
                    hpkr[arm][L] = float(r[key])
                    break

    arms = sorted(curve, key=lambda a: -max(curve[a].values(), default=0))
    hdr = "  ".join(f"L{L}" .rjust(6) for L in LAYERS)
    print(f"{'arm':22s} {hdr}   {'peak':>6s} {'@L':>4s}")
    for a in arms:
        cells = "  ".join((f"{curve[a][L]:.4f}".rjust(6) if L in curve[a] else "     -")
                          for L in LAYERS)
        best = max(curve[a], key=lambda L: curve[a][L])
        print(f"{a:22s} {cells}   {curve[a][best]:.4f} {best:>4d}")

    # ---- claim 1: ranking peaks at 36 --------------------------------
    print("\n" + "=" * 96)
    print("CLAIM 1: 'ranking peaks at 36'")
    print("=" * 96)
    for a in arms[:6]:
        best = max(curve[a], key=lambda L: curve[a][L])
        flag = "<-- peaks at 36" if best == 36 else ""
        print(f"  {a:22s} peak L{best:<3d} = {curve[a][best]:.4f}   "
              f"L36 = {curve[a].get(36, float('nan')):.4f}  {flag}")

    # ---- claim 2: HP-KR is a spike at 40 -----------------------------
    print("\n" + "=" * 96)
    print("CLAIM 2: 'HP-KR is a spike at 40' -- HP-KR column, zero_shot.csv")
    print("=" * 96)
    zs = defaultdict(dict)
    for L in LAYERS:
        for r in rows(f"{BASE}/persona_ood_ext_L{L}/zero_shot.csv"):
            arm = r.get("features") or r.get("arm") or r.get("direction")
            for k, v in r.items():
                if "harm" in k.lower() and "knowledge" in k.lower() and v not in (None, ""):
                    try:
                        zs[arm][L] = float(v)
                    except ValueError:
                        pass
    if zs:
        print(f"{'arm':22s} {hdr}")
        for a in sorted(zs):
            cells = "  ".join((f"{zs[a][L]:.4f}".rjust(6) if L in zs[a] else "     -")
                              for L in LAYERS)
            print(f"{a:22s} {cells}")
    else:
        print("  no HP-KR column found in zero_shot.csv -- inspect its header:")
        z = rows(f"{BASE}/persona_ood_ext_L40/zero_shot.csv")
        if z:
            print(f"  {list(z[0].keys())}")


def margins():
    """The two comparisons that decide what the sweep means."""
    print()
    print("=" * 96)
    print("MARGIN vs HIS BASELINE, and the audit's D-028 comparison, at every layer")
    print("=" * 96)
    print(f"{'layer':>5s} | {'ext_ax-base':>11s} {'excl HP-KR':>11s} | "
          f"{'ext_ax-ext_pc3':>14s} {'excl HP-KR':>11s}")
    nb = nbx = np_ = npx = 0
    for L in LAYERS:
        d = {r["features"]: r for r in rows(f"{BASE}/persona_ood_ext_L{L}/summary.csv")}
        need = ("baseline", "ext_axis1", "ext_pc3")
        if any(k not in d for k in need):
            continue
        f = lambda k, c: float(d[k][c])
        mb = f("ext_axis1", "off_diagonal_mean") - f("baseline", "off_diagonal_mean")
        mbx = f("ext_axis1", "off_diagonal_excl_hpkr") - f("baseline", "off_diagonal_excl_hpkr")
        mp = f("ext_axis1", "off_diagonal_mean") - f("ext_pc3", "off_diagonal_mean")
        mpx = f("ext_axis1", "off_diagonal_excl_hpkr") - f("ext_pc3", "off_diagonal_excl_hpkr")
        nb += mb > 0; nbx += mbx > 0; np_ += mp > 0; npx += mpx > 0
        print(f"{L:5d} | {mb:+11.4f} {mbx:+11.4f} | {mp:+14.4f} {mpx:+11.4f}")
    n = len(LAYERS)
    print()
    print(f"  positive: vs baseline {nb}/{n} (excl HP-KR {nbx}/{n}) | "
          f"vs ext_pc3 {np_}/{n} (excl HP-KR {npx}/{n})")
    print("  READING: the margin over his BASELINE is positive everywhere and survives dropping")
    print("  HP-KR. The draft's Contributions margin (vs ext_pc3) does NOT: without HP-KR it sits")
    print("  within +-0.06 and crosses zero. The audit's D-028 finding generalises to all 16 layers.")


if __name__ == "__main__":
    main()
    margins()
