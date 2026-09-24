"""W6G local recomputations (laptop, git working tree + audit/cache/labpc).

S-05/S-06  run-coords/q8_trained_pertoken.json      (the full run that superseded the smoke)
S-08       pure arithmetic on the NDIF attention allocation
S-15       run-coords/q10_sycophancy_axis.json      (length controls)
S-23       run-coords/probes_L20.json               (3 probes x 4 features, L20, 5 datasets)
S-30       audit/cache/labpc/run-premise/audit/audit_L40_s42.json
S-32       audit/cache/labpc/run-crossgen/kd11/banks/kd/significance.json (+ git copy)
S-33       run-crossgen/kd11/matrices/mean_L40_s42.json and .../banks/kd/matrices/...
S-34       run-paper16/responses/*.jsonl            (character percentiles)
S-35       audit/cache/labpc/run-crossgen/scale_{3b,8b}/matrices/*  (the `random` column)
"""
import glob
import json
import os
import statistics
import sys

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
os.chdir(ROOT)


def hdr(t):
    print("\n================================ " + t)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


want = sys.argv[1:] or ["S-05", "S-08", "S-15", "S-23", "S-30", "S-32", "S-33", "S-34", "S-35"]

# ------------------------------------------------------------------ S-05 / S-06
if "S-05" in want:
    hdr("S-05 / S-06  run-coords/q8_trained_pertoken.json  (FULL run)")
    d = jload("run-coords/q8_trained_pertoken.json")
    print("protocol=%s control=%s layers=%s folds(held_out)=%d n_rows=%d tokens_per_row=%d max_train_tokens=%d"
          % (d["protocol"], d["control"], d["layers"], len(d["held_out"]), d["n_rows"],
             d["tokens_per_row"], d["max_train_tokens"]))
    print("self_checks:", d["self_checks"])
    rs = [r for r in d["results"] if r["layer"] == 44]
    print("L44 rows: %d" % len(rs))
    for r in sorted(rs, key=lambda x: -x["auroc_mean"]):
        print("   %-14s %-9s AUROC %.3f  TPR@1%%FPR %.3f  folds %d"
              % (r["probe"], r["aggregation"], r["auroc_mean"], r["tpr_mean"], len(r["per_fold"])))
    nulls = [r["auroc_mean"] for r in rs if r["probe"] == "tok_lr_null"]
    print("L44 tok_lr_null range in the FULL run: %.3f .. %.3f" % (min(nulls), max(nulls)))
    print("smoke claim range was 0.392 .. 0.462")

# ------------------------------------------------------------------ S-08
if "S-08" in want:
    hdr("S-08  NDIF eager-attention allocation arithmetic")
    for B, T in [(1, 3000), (16, 400), (2, 3000), (6, 1869)]:
        b = B * 64 * T * T * 4
        print("  B=%-3d T=%-5d  bytes=%d  = %.4f GiB  = %.4f GB(1e9)" % (B, T, b, b / 2 ** 30, b / 1e9))
    print("  memory note records for B=6 T=1869: 'attn 5.36 GiB'")
    print("  NDIF error message quoted in the note: 'Tried to allocate 5.00 GiB'")

# ------------------------------------------------------------------ S-15
if "S-15" in want:
    hdr("S-15  run-coords/q10_sycophancy_axis.json")
    d = jload("run-coords/q10_sycophancy_axis.json")
    print("n:", d["n"], " length_only AUROC:", round(d["length_only"], 4))
    print("random:", json.dumps(d["random"]))
    n_entries = n_with = 0
    for bank, dirs in d["results"].items():
        for dname, v in dirs.items():
            n_entries += len(v["by_layer"])
            n_with += len(v["key_layers"])
            kl = sorted(v["key_layers"])
            print("  %-14s %-12s best L%-3s %.4f | by_layer %d | key_layers %s"
                  % (bank, dname, v["best_layer"], v["best"], len(v["by_layer"]), kl))
            for L in kl:
                e = v["key_layers"][L]
                print("        L%-3s auroc %.4f  within_length_quintiles %.4f  length_regressed_out %.4f"
                      % (L, e["auroc"], e["within_length_quintiles"], e["length_regressed_out"]))
    print("TOTAL by_layer AUROCs = %d ; of those carrying the 2 length controls = %d" % (n_entries, n_with))

# ------------------------------------------------------------------ S-23
if "S-23" in want:
    hdr("S-23  run-coords/probes_L20.json")
    d = jload("run-coords/probes_L20.json")
    print("layer=%s pooling=%s k=%s bank=%s" % (d["layer"], d["pooling"], d["k"], d["bank"]))
    print("datasets:", d["datasets"])
    tab = {}
    for r in d["p1"]:
        tab.setdefault(r["features"], {})[r["probe"]] = (r["off_diagonal"], r["diagonal"], r["n_pairs"])
    order = ["pc:deception", "axes", "raw", "random", "axes_resid"]
    print("  %-14s %-22s %-22s %-22s" % ("feature", "paper", "lr_reg10", "mms"))
    for f in order:
        if f not in tab:
            continue
        row = "  %-14s" % f
        for p in ("paper", "lr_reg10", "mms"):
            if p in tab[f]:
                row += " off %.4f (diag %.4f)" % (tab[f][p][0], tab[f][p][1])
            else:
                row += " %-22s" % "-"
        print(row + "   n_pairs=%d" % tab[f]["paper"][2])
    for p in ("paper", "lr_reg10", "mms"):
        print("  delta pc:deception - raw  (%s) = %+.4f" % (p, tab["pc:deception"][p][0] - tab["raw"][p][0]))
        print("  delta axes         - raw  (%s) = %+.4f" % (p, tab["axes"][p][0] - tab["raw"][p][0]))

# ------------------------------------------------------------------ S-30
if "S-30" in want:
    hdr("S-30  audit/cache/labpc/run-premise/audit/audit_L40_s42.json")
    d = jload("audit/cache/labpc/run-premise/audit/audit_L40_s42.json")
    md = d["mean_of_directions"]
    pf = d["pooled_lodo_full"]
    draws = d["pooled_lodo_match_draws"]
    print("layer=%s probe=%s seed=%s budget=%s draws=%d datasets=%d"
          % (d["layer"], d["probe"], d["seed"], d["budget"], d["draws"], len(d["datasets"])))
    print("mean_of_directions   = %.10f" % md)
    print("pooled_lodo_full     = %.10f" % pf)
    print("margin (Result 2 over pooled_lodo_full) = %+.4f" % (md - pf))
    print("pooled_lodo_match: mean %.4f  min %.4f  max %.4f  range %.4f  sd %.4f"
          % (statistics.mean(draws), min(draws), max(draws), max(draws) - min(draws),
             statistics.stdev(draws)))
    print("margin / range = %.3f ; margin / sd = %.3f" % ((md - pf) / (max(draws) - min(draws)),
                                                          (md - pf) / statistics.stdev(draws)))
    print("baselines:", {k: round(v, 4) for k, v in d["baselines"].items()})
    print("pairs_below_half %d of %d" % (d["pairs_below_half"], d["n_pairs"]))
    print("margin vs mean_over_sources = %+.4f" % (md - d["baselines"]["mean_over_sources"]))

# ------------------------------------------------------------------ S-32
if "S-32" in want:
    hdr("S-32  kd-bank significance.json  (lab cache and git copy)")
    for tag, p in [("labcache", "audit/cache/labpc/run-crossgen/kd11/banks/kd/significance.json"),
                   ("git-tree", "run-crossgen/kd11/banks/kd/significance.json")]:
        if not os.path.exists(p):
            print("  %s MISSING %s" % (tag, p))
            continue
        d = jload(p)
        tests = d["tests"] if isinstance(d, dict) and "tests" in d else d
        print("\n  --- %s %s  (%d tests, mtime %s)"
              % (tag, p, len(tests), __import__("datetime").datetime.fromtimestamp(os.path.getmtime(p))))
        for k, v in (d.items() if isinstance(d, dict) else []):
            if k != "tests" and not isinstance(v, (list, dict)):
                print("      %s = %s" % (k, v))
        sel = [t for t in tests if t.get("layer") == 40 and t.get("pooling") == "mean"]
        print("      L40/mean tests: %d" % len(sel))
        sel.sort(key=lambda t: -t.get("diff", 0))
        for t in sel[:6]:
            print("      %-9s %-26s diff %+.4f  pairs %s/%s  CI [%+.3f, %+.3f]  q_BH %.3f  sig=%s  block=%s"
                  % (t.get("probe"), t.get("comparison"), t.get("diff"),
                     t.get("pairs_won"), t.get("n_pairs"),
                     t.get("dataset_ci95", [float("nan")] * 2)[0], t.get("dataset_ci95", [float("nan")] * 2)[1],
                     t.get("dataset_q_bh", float("nan")), t.get("significant_datasets"), t.get("block")))
        cleared = [t for t in tests if t.get("significant_datasets")]
        print("      cleared the DATASET-level bootstrap: %d / %d" % (len(cleared), len(tests)))

# ------------------------------------------------------------------ S-33
if "S-33" in want:
    hdr("S-33  paper16 (kd11 root) vs kd bank, L40 mean s42")
    for tag, p in [("paper16(kd11 root)", "run-crossgen/kd11/matrices/mean_L40_s42.json"),
                   ("kd bank", "run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json"),
                   ("paper16 lab", "audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json"),
                   ("kd bank lab", "audit/cache/labpc/run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json")]:
        if not os.path.exists(p):
            print("  %-20s MISSING %s" % (tag, p))
            continue
        d = jload(p)
        ds = d["datasets"]
        print("\n  --- %s  %s" % (tag, p))
        print("      bank=%s bank_name=%s k=%s layer=%s pooling=%s seed=%s n_datasets=%d"
              % (d.get("bank"), d.get("bank_name"), d.get("k"), d["layer"], d["pooling"], d["seed"], len(ds)))
        print("      block key:", [k for k in d if k.startswith("block")])
        base = None
        for r in d["results"]:
            if r.get("probe") != "paper":
                continue
            m = r["auroc"]
            off = [m[i][j] for i in range(len(ds)) for j in range(len(ds))
                   if i != j and m[i][j] == m[i][j]]
            if r["feature"] == "raw":
                base = sum(off) / len(off)
        for r in d["results"]:
            if r.get("probe") != "paper":
                continue
            m = r["auroc"]
            off = [m[i][j] for i in range(len(ds)) for j in range(len(ds))
                   if i != j and m[i][j] == m[i][j]]
            mu = sum(off) / len(off)
            nan = sum(1 for i in range(len(ds)) for j in range(len(ds))
                      if i != j and m[i][j] != m[i][j])
            print("      %-14s off-diag %.4f  d vs raw %s  pairs %d (nan %d)"
                  % (r["feature"], mu, ("%+.4f" % (mu - base)) if base is not None else "  n/a  ",
                     len(off), nan))

# ------------------------------------------------------------------ S-34
if "S-34" in want:
    hdr("S-34  run-paper16/responses  character-length percentiles")
    import numpy as np
    for store in ["run-paper16/responses", "audit/cache/labpc/run-paper16/responses"]:
        fs = sorted(glob.glob(store + "/*.jsonl"))
        if not fs:
            print("  %s : no files" % store)
            continue
        keys = {}
        L = []
        for f in fs:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for k in r:
                    keys[k] = keys.get(k, 0) + 1
                v = r.get("response", r.get("completion", r.get("text", "")))
                L.append(len(v) if isinstance(v, str) else 0)
        a = np.array(L)
        print("  %s : %d files, %d rows, record keys=%s" % (store, len(fs), len(a), sorted(keys)))
        print("     chars p50=%.0f p90=%.0f p99=%.0f max=%.0f  (F-33 70B row: 1480 / 2542 / 2825 / 3059)"
              % (np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 99), a.max()))
        print("     rows over 12,000 chars: %d" % int((a > 12000).sum()))

# ------------------------------------------------------------------ S-35
if "S-35" in want:
    hdr("S-35  the `random` feature column in the scale matrices")
    for rung in ("scale_3b", "scale_8b"):
        base = "audit/cache/labpc/run-crossgen/%s/matrices" % rung
        print("\n  === %s (mean pooling, probe=paper)" % rung)
        print("      %4s %8s %8s %8s %8s %8s %8s  %s" %
              ("L", "raw", "axes", "pc:dec", "datapca", "random", "length", "bank/k/n_ds"))
        for f in sorted(glob.glob(base + "/mean_L*_s42.json"),
                        key=lambda x: int(x.split("_L")[1].split("_")[0])):
            d = jload(f)
            ds = d["datasets"]
            vals = {}
            for r in d["results"]:
                if r.get("probe") != "paper":
                    continue
                m = r["auroc"]
                off = [m[i][j] for i in range(len(ds)) for j in range(len(ds))
                       if i != j and m[i][j] == m[i][j]]
                vals[r["feature"]] = sum(off) / len(off)
            g = lambda k: ("%8.4f" % vals[k]) if k in vals else "     -  "
            print("      %4d %s %s %s %s %s %s  %s/k=%s/%d"
                  % (d["layer"], g("raw"), g("axes"), g("pc:deception"), g("datapca"),
                     g("random"), g("length"), d.get("bank_name"), d.get("k"), len(ds)))
