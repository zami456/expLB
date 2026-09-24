"""W6B-thesis2 : O-083, O-084, O-085, O-087, O-089, O-090, O-091, O-092,
O-093, O-094  -- section 7.6 (Q6, "Which layer: 20, 22 or 40?") of
docs/thesis/persona-coordinates-70b-results.md, lines 723-748.

Level 1 : each quoted table cell vs the stored row of run-coords/q6_layers.json.
Level 2 : (a) the six per-method table rows are re-read out of the SOURCE sweep
              files that q6_layers.json says they came from, not out of
              q6_layers.json itself;
          (b) the medians/maxima in the verdict bullets are recomputed from the
              23 per-row gaps, with the raw / persona split rebuilt from the
              method strings;
          (c) the L60 / L68 zero-shot collapse is read from dotproducts_mean.json;
          (d) the 20-vs-22 equivalence CIs are read from significance.json.

Read-only.  Run from the repo root.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
RC = ROOT / "run-coords"


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prov(p: pathlib.Path):
    print("  %-34s %8d B  md5 %s  mtime %s"
          % (p.name, p.stat().st_size, md5(p),
             datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(" ", "seconds")))


def is_persona(method: str, source: str) -> bool:
    m = method.split("/")[0].strip()
    if m in ("raw", "tok_lr", "tok_mms", "pool_mean_mms", "pool_mean_lr",
             "pool_last_mms"):
        return False
    if m in ("axes", "pc:deception", "deception contrast axis", "PC1",
             "axis", "zeroshot_axis"):
        return True
    raise SystemExit("unclassified method: %r (%s)" % (method, source))


def main():
    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    for n in ("q6_layers.json", "layer_sweep_passB_mms.json",
              "layer_sweep_passB_last.json", "layer_sweep_mean.json",
              "layer_sweep_mms.json", "dotproducts_mean.json",
              "pertoken_q8.json", "q8_trained_pertoken.json",
              "significance.json"):
        prov(RC / n)

    d = json.load(open(RC / "q6_layers.json", encoding="utf-8"))
    rows = d["rows"]
    print("\n  q6_layers.json: targets=%s, %d method rows, tol=%s"
          % (d["targets"], len(rows), d["tol"]))

    # -------------------------------------------------- the six claimed rows
    # claim id -> (method string in q6_layers.json, doc L20, L22, L40,
    #              doc best layer, doc best)
    claimed = {
        "O-083": ("raw / mms / mean",              0.652, 0.646, 0.669, 32, 0.717),
        "O-084": ("raw / lr_reg10 / mean",         0.676, None,  0.675, 32, 0.718),
        "O-085": ("axes / mms / mean",             0.570, 0.566, 0.721, 44, 0.731),
        "O-087": ("deception contrast axis / mean", 0.608, 0.630, 0.863, 40, 0.863),
        "O-089": ("pool_mean_mms / pooled",        0.857, 0.854, 0.883, 40, 0.883),
        "O-090": ("tok_lr / mean",                 0.785, 0.779, 0.839, 44, 0.851),
    }
    # O-084 is the Pass A row, and there are TWO "raw / ... / mean" rows;
    # disambiguate by source.
    src_hint = {"O-084": "Pass A", "O-083": "full corpus", "O-085": "full corpus"}

    print("\n" + "=" * 78)
    print("LEVEL 1  --  the six quoted table rows vs q6_layers.json")
    print("=" * 78)
    for cid, (meth, d20, d22, d40, dbl, dbest) in claimed.items():
        cands = [r for r in rows if r["method"] == meth]
        if cid in src_hint:
            cands = [r for r in cands if src_hint[cid] in r["source"]]
        assert len(cands) == 1, (cid, meth, [r["source"] for r in cands])
        r = cands[0]
        at = r["at"]
        got = (at["20"], at["22"], at["40"], r["best_layer"], r["best"])
        print("\n%s  %-28s   source=%r  file=%s" % (cid, meth, r["source"], r["file"]))
        print("      file : L20 %s  L22 %s  L40 %.6f  own best L%d %.6f"
              % ("%.6f" % at["20"] if at["20"] is not None else "   None ",
                 "%.6f" % at["22"] if at["22"] is not None else "   None ",
                 at["40"], r["best_layer"], r["best"]))
        print("      doc  : L20 %s  L22 %s  L40 %.3f     own best L%d %.3f"
              % ("%.3f" % d20, ("%.3f" % d22) if d22 is not None else " --  ",
                 d40, dbl, dbest))
        checks = []
        checks.append(("L20", round(at["20"], 3) == d20))
        checks.append(("L22", (at["22"] is None and d22 is None)
                       or (at["22"] is not None and d22 is not None
                           and round(at["22"], 3) == d22)))
        checks.append(("L40", round(at["40"], 3) == d40))
        checks.append(("best_layer", r["best_layer"] == dbl))
        checks.append(("best", round(r["best"], 3) == dbest))
        print("      LEVEL1 %s   %s"
              % ("MATCH" if all(c[1] for c in checks) else "MISMATCH",
                 " ".join("%s=%s" % (k, "ok" if v else "NO") for k, v in checks)))

        # ---------- Level 2: re-read from the source sweep file where possible
        sf = RC / r["file"]
        if sf.exists() and r["file"].startswith("layer_sweep"):
            sw = json.load(open(sf, encoding="utf-8"))
            feat = meth.split("/")[0].strip()
            key = {"raw": "raw", "axes": "axes", "pc:deception": "pc:deception"}[feat]
            series = {int(e["layer"]): e[key] for e in sw["sweep"]
                      if key in e and e[key] is not None}
            bl = max(series, key=series.get)
            print("      L2   : re-read %s (pooling=%s probe=%s k=%s bank=%s)"
                  % (r["file"], sw.get("pooling"), sw.get("probe"), sw.get("k"),
                     sw.get("bank")))
            print("             -> L20 %.6f  L40 %.6f  argmax over %d layers "
                  "= L%d %.6f"
                  % (series[20], series[40], len(series), bl, series[bl]))
            print("             agrees with q6_layers row: %s"
                  % all([abs(series[20] - at["20"]) < 1e-9,
                         abs(series[40] - at["40"]) < 1e-9,
                         bl == r["best_layer"],
                         abs(series[bl] - r["best"]) < 1e-9]))
        elif r["file"] == "dotproducts_mean.json":
            dp = json.load(open(RC / r["file"], encoding="utf-8"))
            series = {m["layer"]: m["contrast"] for m in dp["matched"]}
            bl = max(series, key=series.get)
            print("      L2   : re-read dotproducts_mean.json contrast -> "
                  "L20 %.6f L22 %.6f L40 %.6f  argmax over %d layers = L%d %.6f"
                  % (series[20], series[22], series[40], len(series), bl,
                     series[bl]))
            print("             agrees with q6_layers row: %s"
                  % all([abs(series[20] - at["20"]) < 1e-9,
                         abs(series[22] - at["22"]) < 1e-9,
                         abs(series[40] - at["40"]) < 1e-9,
                         bl == r["best_layer"]]))
        elif r["file"] == "q8_trained_pertoken.json":
            qt = json.load(open(RC / r["file"], encoding="utf-8"))
            probe, agg = [s.strip() for s in meth.split("/")]
            series = {e["layer"]: e["auroc_mean"] for e in qt["results"]
                      if e["probe"] == probe and e["aggregation"] == agg}
            bl = max(series, key=series.get)
            print("      L2   : re-read q8_trained_pertoken.json "
                  "(probe=%s aggregation=%s, protocol=%s, %d folds) -> "
                  "L20 %.6f L22 %.6f L40 %.6f  argmax over %s = L%d %.6f"
                  % (probe, agg, qt.get("protocol"),
                     len(qt["results"][0]["per_fold"]),
                     series[20], series[22], series[40], sorted(series), bl,
                     series[bl]))
            print("             agrees with q6_layers row: %s"
                  % all([abs(series[20] - at["20"]) < 1e-9,
                         abs(series[22] - at["22"]) < 1e-9,
                         abs(series[40] - at["40"]) < 1e-9,
                         bl == r["best_layer"],
                         abs(series[bl] - r["best"]) < 1e-9]))

    # ------------------------------------------------------- verdict bullets
    print("\n" + "=" * 78)
    print("LEVEL 2  --  the four verdict bullets (7.6, lines 739-748)")
    print("=" * 78)

    g2022 = [(r["method"], r["source"], abs(r["gap_20_22"]))
             for r in rows if r["gap_20_22"] is not None]
    print("\nO-091  '20 vs 22 is immaterial. Median gap over the 17 methods is "
          "0.012 and the maximum 0.037.'")
    print("       rows with both layers: %d" % len(g2022))
    for m, s, g in sorted(g2022, key=lambda t: -t[2]):
        print("         %-52s %.6f" % (m + "  [" + s[:22] + "]", g))
    print("       median |gap| = %.6f  -> %.3f" % (statistics.median(
        [g for _, _, g in g2022]), statistics.median([g for _, _, g in g2022])))
    print("       max    |gap| = %.6f  -> %.3f" % (max(g for _, _, g in g2022),
                                                   max(g for _, _, g in g2022)))
    sig = json.load(open(RC / "significance.json", encoding="utf-8"))
    eq = [c for c in sig["comparisons"] if c.get("kind") == "equiv"]
    print("       + second sentence: 'the four equivalence tests have 90%% CIs "
          "from -0.038 to +0.033'")
    for c in eq:
        print("         %-34s diff %+.6f  dataset_ci90 [%+.5f, %+.5f]  "
              "equivalent_within_margin=%s (margin %s)"
              % (c["name"], c["diff"], c["dataset_ci90"][0], c["dataset_ci90"][1],
                 c.get("equivalent_within_margin"), sig["margin"]))
    print("         union of the four CI90s: [%+.5f, %+.5f]"
          % (min(c["dataset_ci90"][0] for c in eq),
             max(c["dataset_ci90"][1] for c in eq)))

    gaps40 = [(r["method"], r["source"], r["gap_40_minus_best_of_20_22"],
               is_persona(r["method"], r["source"])) for r in rows]
    raw = [g for *_, g, p in [(m, s, g, p) for m, s, g, p in gaps40] if not p]
    per = [g for m, s, g, p in gaps40 if p]
    print("\nO-092  'Raw probes: layer 40 helps a little. Median +0.025 over the "
          "better of 20/22 (range -0.016 to +0.068, 10 methods).'")
    print("       raw-probe rows (%d):" % len(raw))
    for m, s, g, p in gaps40:
        if not p:
            print("         %-52s %+.6f" % (m + "  [" + s[:22] + "]", g))
    print("       median %+.6f -> %+.3f    min %+.6f -> %+.3f    max %+.6f -> %+.3f"
          % (statistics.median(raw), statistics.median(raw), min(raw), min(raw),
             max(raw), max(raw)))

    print("\nO-093  'Persona methods: layer 40 dominates. Median +0.113 "
          "(13 methods), up to +0.234 zero-shot.'")
    print("       persona rows (%d):" % len(per))
    for m, s, g, p in gaps40:
        if p:
            print("         %-52s %+.6f" % (m + "  [" + s[:22] + "]", g))
    print("       median %+.6f -> %+.3f    max %+.6f -> %+.3f (%s)"
          % (statistics.median(per), statistics.median(per), max(per), max(per),
             [m for m, s, g, p in gaps40 if p and g == max(per)][0]))
    print("       zero-shot subset only: max %+.6f"
          % max(g for m, s, g, p in gaps40 if p and "zero-shot" in s))
    print("       ratio persona-median / raw-median = %.2fx  (doc: 'about 4x')"
          % (statistics.median(per) / statistics.median(raw)))

    print("\nO-094  'deception-axis AUROC is 0.491 at L60 and 0.445 at L68, "
          "below chance'")
    dp = json.load(open(RC / "dotproducts_mean.json", encoding="utf-8"))
    print("       dotproducts_mean.json layers: %s" % dp["layers"])
    print("       %6s %10s %10s" % ("layer", "contrast", "pc1"))
    for m in dp["matched"]:
        if m["layer"] >= 40:
            print("       %6d %10.6f %10.6f" % (m["layer"], m["contrast"], m["pc1"]))
    for L in (60, 68):
        v = next(m["contrast"] for m in dp["matched"] if m["layer"] == L)
        print("       L%-3d contrast = %.6f -> %.3f   (below chance: %s)"
              % (L, v, v, v < 0.5))
    below = [m["layer"] for m in dp["matched"] if m["contrast"] < 0.5]
    print("       layers with contrast AUROC < 0.5: %s" % below)


if __name__ == "__main__":
    main()
