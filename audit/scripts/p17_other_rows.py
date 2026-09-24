"""P17 Agent A, task 4: do any D- rows NOT stamped ea1ce52b (the P5/P8 rows citing 45b16d9 /
"his branch @ tip") rest on a CSV value that changed between PIN_PCOOD (45b16d90) and PIN_PAPER?

The only cells that changed (audit/p17/csv_drift_from_45b16d9.csv) are the raw-probe ('baseline')
matrices and the baseline rows of summary.csv / summary_balacc.csv at L20/L40/L44. This script
re-derives, at both SHAs, every baseline-derived quantity those rows cite, at the precision the row
cites it, and writes audit/p17/d_rows_other_affected.csv.
"""
import csv, io, sys, pathlib
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP


def hu(x, p):
    """half-up on the decimal repr, as make_numbers.py fmt() does (a 0.78746 mean prints 0.787, 0.78750 prints 0.788)"""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p17_common as C

SH = {"45b16d90": C.PIN_PCOOD, "16def2de": C.PIN_PAPER}
L3 = (20, 40, 44)
ORD = ["convince_me__plain", "harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain",
       "insider_trading_cadenza__plain", "instruct_dishonesty__plain", "repe_honesty__plain"]
AB = dict(zip(ORD, ["CG", "HP-C", "HP-KR", "IT", "ID", "RepE"]))


def df(sha, p, **kw):
    return pd.read_csv(io.StringIO(C.show(sha, p)), **kw)


def e(L):
    return f"{C.RES}/persona_ood_ext_L{L}"


def q(sha):
    out = {}
    for L in L3:
        s = df(sha, f"{e(L)}/summary.csv").set_index("features")
        sb = df(sha, f"{e(L)}/summary_balacc.csv").set_index("features")
        mb = df(sha, f"{e(L)}/matrix_baseline.csv", index_col=0)
        me = df(sha, f"{e(L)}/matrix_ext_axis1.csv", index_col=0)
        mba = df(sha, f"{e(L)}/matrix_balacc_baseline.csv", index_col=0)
        off = ~np.eye(6, dtype=bool)
        vb, ve = mb.to_numpy(float), me.to_numpy(float)
        out[f"summary baseline off_diag L{L}"] = round(s.loc["baseline", "off_diagonal_mean"], 3)
        out[f"summary baseline diag L{L}"] = round(s.loc["baseline", "diagonal_mean"], 3)
        out[f"summary baseline drop L{L}"] = round(s.loc["baseline", "diagonal_mean"] - s.loc["baseline", "off_diagonal_mean"], 3)
        vv = vb.copy(); np.fill_diagonal(vv, np.nan)
        pt = np.nanmean(vv, axis=0)
        for j, n in enumerate(mb.columns):
            out[f"per-test baseline {AB[n]} L{L} (3dp)"] = hu(pt[j], 3)
            out[f"per-test baseline {AB[n]} L{L} (4dp)"] = hu(pt[j], 4)
        vx = ve.copy(); np.fill_diagonal(vx, np.nan)
        ptx = np.nanmean(vx, axis=0)
        out[f"per-test ext>base count L{L}"] = int(sum(ptx > pt))
        out[f"cells raw>ext L{L}"] = int(((vb > ve) & off).sum())
        i, j = np.unravel_index(np.nanargmin(np.where(off, vb, np.inf)), vb.shape)
        out[f"min off-diag baseline cell L{L}"] = f"{AB[mb.index[i]]}->{AB[mb.columns[j]]} {vb[i, j]:.4f}"
        out[f"balacc baseline cells_at_floor L{L}"] = int(sb.loc["baseline", "cells_at_floor"])
        out[f"balacc baseline cells exactly 0.5 L{L}"] = int(((mba.to_numpy(float) == 0.5) & off).sum())
        out[f"RepE-row baseline mean (5 others) L{L}"] = round(float(np.delete(vb[ORD.index('repe_honesty__plain')], ORD.index('repe_honesty__plain')).mean()), 4)
    m40b = df(sha, f"{e(40)}/matrix_baseline.csv", index_col=0)
    m40e = df(sha, f"{e(40)}/matrix_ext_axis1.csv", index_col=0)
    out["L40 cell ID->CG raw-ext"] = round(m40b.loc["instruct_dishonesty__plain", "convince_me__plain"] - m40e.loc["instruct_dishonesty__plain", "convince_me__plain"], 4)
    return out


A, B = q(SH["45b16d90"]), q(SH["16def2de"])
diff = {k: (A[k], B[k]) for k in A if A[k] != B[k]}
print(f"{len(A)} baseline-derived quantities; {len(diff)} differ between 45b16d90 and 16def2de:")
for k, (a, b) in diff.items():
    print(f"  {k:42s} 45b16d90={a}  16def2de={b}")

# row -> quantities it cites (from each row's evidence_key/level values in audit/claims.csv)
USES = {
    "D-008": ["summary baseline off_diag L40"],
    "D-096": ["summary baseline diag L40", "summary baseline off_diag L40", "summary baseline drop L40"],
    "D-097": ["min off-diag baseline cell L40"],
    "D-098": [f"summary baseline off_diag L{L}" for L in L3],
    "D-168": [f"summary baseline off_diag L{L}" for L in L3],
    "D-100": [f"cells raw>ext L{L}" for L in L3] + ["L40 cell ID->CG raw-ext", "summary baseline diag L40"],
    "D-009": [k for k in A if k.startswith("per-test baseline") and k.endswith("L40 (4dp)")],
    "D-027": [k for k in A if k.startswith("per-test")],
    "D-099": [k for k in A if k.startswith("per-test ext>base")],
    "D-195": [k for k in A if k.startswith("per-test baseline") and "(3dp)" in k] + [k for k in A if k.startswith("per-test ext>base")],
    "D-157": ["balacc baseline cells_at_floor L40", "balacc baseline cells exactly 0.5 L40"],
    "D-216": ["RepE-row baseline mean (5 others) L20"],
}
claims = {r["id"]: r for r in csv.DictReader(open(C.ROOT / "audit" / "claims.csv", encoding="utf-8"))}
rows = []
for cid, ks in USES.items():
    ch = [k for k in ks if k in diff]
    rows.append({"id": cid, "version_used": claims[cid]["version_used"][:60],
                 "verdict_stored": claims[cid]["verdict"][:40],
                 "n_quantities_checked": len(ks), "n_changed": len(ch),
                 "changed": "; ".join(f"{k}: {diff[k][0]} -> {diff[k][1]}" for k in ch),
                 "affected": "y" if ch else "n",
                 "evidence": f"{C.RES}/persona_ood_ext_L{{20,40,44}}/ @ {C.PIN_PCOOD[:8]} vs {C.PIN_PAPER[:8]}; script audit/scripts/p17_other_rows.py"})
out = C.ROOT / "audit" / "p17" / "d_rows_other_affected.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nrows checked:", len(rows), "; affected:", [r["id"] for r in rows if r["affected"] == "y"])
for r in rows:
    if r["affected"] == "y":
        print(f"  {r['id']}: {r['changed'][:200]}")
