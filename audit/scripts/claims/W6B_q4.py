"""W6B-thesis2 : O-076, O-077, O-078, O-079.

Checks three rows of the 7.4 Q4 erasure table in
docs/thesis/persona-coordinates-70b-results.md (lines 662, 663, 664) and the
verdict bullet "the assistant direction carries a small but real 1-3% of
transfer AUROC" (line 670), against the file that produced them.

Level 1 : document cell  vs  stored field.
Level 2 : recompute cost = raw - erase_assistant, the random-erasure mean, and
          the cost expressed as a share of raw, from the per-row numbers;
          match every BH q to its named comparison and report the FAMILY SIZE.

Read-only.  Run from the repo root.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
F = os.path.join(ROOT, "run-coords", "q3_q4_bootstrap.json")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("file        :", F)
    print("size, md5   :", os.path.getsize(F), md5(F))
    print("mtime       :", __import__("datetime").datetime.fromtimestamp(
        os.path.getmtime(F)).isoformat())
    d = json.load(open(F, encoding="utf-8"))
    print("boot_rows   :", d["boot_rows"], " boot_datasets:", d["boot_datasets"])
    comps = d["comparisons"]
    print("BH FAMILY   : %d comparisons in this file "
          "(Q3 %d + Q4 %d) -- the q column is corrected over THIS family only"
          % (len(comps),
             sum(1 for c in comps if "erasing" not in c["name"]
                 and "erasure" not in c["name"]),
             sum(1 for c in comps if "erasing" in c["name"]
                 or "erasure" in c["name"])))
    print()

    # document cells, transcribed from the .md table (lines 661-666)
    doc = {
        (20, "mean", "mms"):      (0.652, 0.646, 0.006, (0.513, 0.796), 0.52),
        (20, "mean", "lr_reg10"): (0.660, 0.655, 0.005, (0.558, 0.741), 0.54),
        (40, "mean", "mms"):      (0.669, 0.652, 0.018, (0.510, 0.786), 0.003),
        (40, "mean", "lr_reg10"): (0.686, 0.672, 0.014, (0.586, 0.760), 0.16),
        (44, "last", "mms"):      (0.720, 0.712, 0.008, (0.608, 0.824), 0.003),
        (44, "last", "lr_reg10"): (0.661, 0.656, 0.005, (0.570, 0.752), 0.04),
    }
    mine = {"O-076": (20, "mean", "lr_reg10"),
            "O-077": (40, "mean", "mms"),
            "O-078": (40, "mean", "lr_reg10")}

    shares = []
    print("%-6s %-22s %9s %9s %9s %9s %-20s %9s" %
          ("claim", "setting", "raw", "erased", "cost", "rand.mean",
           "erased dataset CI95", "q_BH"))
    for r in d["q4_table"]:
        key = (r["layer"], r["pooling"], r["probe"])
        cid = next((c for c, k in mine.items() if k == key), "  -")
        cr = sum(r["cost_random"]) / len(r["cost_random"])
        ci = r["erased_dataset_ci95"]
        name = ("cost of erasing Assistant Axis, L%d %s %s"
                % (r["layer"], r["pooling"], r["probe"]))
        q = next(c["dataset_q_bh"] for c in comps if c["name"] == name)
        pval = next(c["dataset_p"] for c in comps if c["name"] == name)
        share = 100.0 * r["cost_assistant"] / r["raw"]
        shares.append((key, share))
        print("%-6s L%-2d %-6s %-11s %9.6f %9.6f %9.6f %9.2e [%.4f, %.4f]  %9.5f"
              % (cid, r["layer"], r["pooling"], r["probe"], r["raw"],
                 r["erase_assistant"], r["cost_assistant"], cr, ci[0], ci[1], q))
        # Level 1
        dr, de, dc, dci, dq = doc[key]
        ok = (abs(round(r["raw"], 3) - dr) < 5e-4
              and abs(round(r["erase_assistant"], 3) - de) < 5e-4
              and abs(round(r["cost_assistant"], 3) - dc) < 5e-4
              and abs(round(ci[0], 3) - dci[0]) < 5e-4
              and abs(round(ci[1], 3) - dci[1]) < 5e-4
              and abs(round(q, len(str(dq).split(".")[1])) - dq) < 5e-4)
        print("       doc says raw %.3f erased %.3f cost %.3f CI [%.3f, %.3f] "
              "q %.3f  -> LEVEL1 %s   (dataset_p %.4f)"
              % (dr, de, dc, dci[0], dci[1], dq, "MATCH" if ok else "MISMATCH",
                 pval))
        # Level 2 : cost is exactly raw - erased?
        recomputed = r["raw"] - r["erase_assistant"]
        print("       L2 recompute raw-erased = %.9f  (stored %.9f, "
              "delta %.2e)   cost as share of raw = %.3f%%"
              % (recomputed, r["cost_assistant"],
                 abs(recomputed - r["cost_assistant"]), share))
        print()

    print("O-079  'the assistant direction carries a small but real 1-3% of "
          "transfer AUROC'")
    vals = [s for _, s in shares]
    print("       cost / raw over the six settings, in percent:")
    for k, s in shares:
        print("         L%-2d %-5s %-9s  %6.3f%%" % (k[0], k[1], k[2], s))
    print("       range %.3f%% .. %.3f%%   (median %.3f%%)"
          % (min(vals), max(vals), sorted(vals)[len(vals) // 2 - 1:][0]))
    print("       rows BELOW 1%%: %d of 6 ;  rows ABOVE 3%%: %d of 6"
          % (sum(1 for v in vals if v < 1.0), sum(1 for v in vals if v > 3.0)))
    print("       absolute cost in AUROC points: %.4f .. %.4f"
          % (min(r["cost_assistant"] for r in d["q4_table"]),
             max(r["cost_assistant"] for r in d["q4_table"])))
    print("       BH-significant across datasets (q<0.05): %d of 6"
          % sum(1 for c in comps
                if c["name"].startswith("cost of erasing")
                and c["dataset_q_bh"] < 0.05))


if __name__ == "__main__":
    sys.exit(main())
