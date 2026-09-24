"""W6C-briefing: claims that live in the run-premise store (cached from the lab PC).

Covers O-016, O-017 (0.916 vs our pooled 0.868 / 0.893 on SIX datasets),
O-020 (110 pairs: 21 below 0.5, 14 below 0.4, min 0.055),
O-021 (the four single-source baselines 0.676 / 0.722 / 0.735 / 0.897),
O-026 (the trained-LODO persona table and its CIs),
O-027 (pc:unified@2 gains +0.118 from pooling),
O-028 / O-029 (mean-of-directions +0.028, CI [+0.005, +0.052]),
O-032 / O-034 (persona features lose to raw pooled LR by 0.11-0.16).

All inputs are the audit's cached copies of lab-PC files under
audit/cache/labpc/run-premise/ (the lab PC is unreachable this session).
Read-only. Run from the repo root.
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

CACHE = "audit/cache/labpc/run-premise"
p = sys.stdout.write


def load(rel):
    path = os.path.join(CACHE, rel)
    with io.open(path, "rb") as f:
        b = f.read()
    return json.loads(b.decode("utf-8")), path, hashlib.md5(b).hexdigest(), len(b)


def banner(t):
    p("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78 + "\n")


def dsboot(a, b, B=10000, seed=42):
    """Paired dataset-level bootstrap of mean(a) - mean(b) over targets."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    d = (a[idx] - b[idx]).mean(axis=1)
    obs = float(a.mean() - b.mean())
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


FILES = []

banner("O-016 / O-017 : the pooled probe on OUR SIX overlapping datasets (L32, L40)")
six = {}
for L in (20, 32, 40):
    d, path, md5, n = load("six/premise_L%d_mean_s42.json" % L)
    FILES.append((path, n, md5))
    six[L] = d
    s = d["summary"]
    p("  L%-3d datasets=%d  single %.4f   pooled_lodo %.4f   pooled_lodo_match %.4f\n"
      % (L, len(d["datasets"]), s["single"], s["pooled_lodo"], s["pooled_lodo_match"]))
    p("        n_train: single_mean %s  pooled_full %s  pooled_match %s\n"
      % (d["n_train"]["single_mean"], d["n_train"]["pooled_full"], d["n_train"]["pooled_match"]))
p("  datasets in the six-set: %s\n" % ", ".join(six[40]["datasets"]))
ABS = 0.9158352734669751  # collaborator's ext_axis1 off_diagonal_mean at L40 (his file)
p("\n  briefing 0.916 = his ext_axis1 L40 = %.10f\n" % ABS)
for L in (32, 40):
    s = six[L]["summary"]
    p("  L%-3d  0.916 - pooled_lodo %.4f       = %+.4f   (briefing +0.048 at L40)\n"
      % (L, s["pooled_lodo"], ABS - s["pooled_lodo"]))
    p("  L%-3d  0.916 - pooled_lodo_match %.4f = %+.4f   (briefing +0.023 at L40, +0.015 at L32)\n"
      % (L, s["pooled_lodo_match"], ABS - s["pooled_lodo_match"]))

banner("O-020 / O-021 : the 110 source->target pairs and the four single-source baselines")
aud, path, md5, n = load("audit/audit_L40_s42.json")
FILES.append((path, n, md5))
b = aud["baselines"]
p("  stored baselines block (audit_L40_s42.json):\n")
for k in ("mean_over_sources", "median_over_sources", "oriented_mean", "best_source_oracle"):
    p("    %-22s %.10f\n" % (k, b[k]))
p("    pairs_below_half %s   n_pairs %s\n" % (aud["pairs_below_half"], aud["n_pairs"]))

pairs = aud["pairs"]
vals = np.array([r["auroc"] for r in pairs], float)
p("\n  recomputed from the 110 stored pairs:\n")
p("    n                       %d\n" % len(vals))
p("    mean over sources       %.10f\n" % vals.mean())
p("    median over sources     %.10f\n" % float(np.median(vals)))
p("    sign-oriented mean      %.10f\n" % float(np.maximum(vals, 1 - vals).mean()))
p("    minimum                 %.10f\n" % vals.min())
p("    pairs < 0.5             %d\n" % int((vals < 0.5).sum()))
p("    pairs < 0.4             %d\n" % int((vals < 0.4).sum()))
p("    pairs < 0.3             %d\n" % int((vals < 0.3).sum()))
p("  NOTE on 'median over sources': the stored 0.7220 is the mean over targets of the\n"
  "  per-target median (per_target.median_over_sources), NOT the median of all 110 pairs.\n")
pt = aud["per_target"]
p("    mean of per-target medians  %.10f\n" % float(np.mean(pt["median_over_sources"])))
p("    mean of per-target oriented %.10f\n" % float(np.mean(pt["oriented_mean"])))
p("    mean of per-target oracle   %.10f\n" % float(np.mean(pt["best_source_oracle"])))
p("    mean of per-target means    %.10f\n" % float(np.mean(pt["mean_over_sources"])))
p("  the seven worst pairs:\n")
for r in sorted(pairs, key=lambda r: r["auroc"])[:7]:
    p("    %.4f   %-32s -> %s\n" % (r["auroc"], r["source"], r["target"]))

banner("O-026 / O-027 / O-032 / O-034 : the trained leave-one-dataset-out persona table")
prem, path, md5, n = load("premise_L40_mean_s42.json")
FILES.append((path, n, md5))
raw_pt = np.array(aud["per_target"]["pooled_lodo_full"], float)
raw_single = aud["baselines"]["mean_over_sources"]
raw_lodo = aud["pooled_lodo_full"]
p("  raw 8192-d:  single %.4f   pooled LODO %.4f   (briefing 0.676 / 0.854)\n"
  % (raw_single, raw_lodo))
p("  cross-check against premise_L40_mean_s42.json summary: single %.4f  pooled_lodo %.4f\n"
  % (prem["summary"]["single"], prem["summary"]["pooled_lodo"]))
p("\n  %-14s %10s %10s %12s %10s %-26s\n"
  % ("feature", "single", "pooledLODO", "vs raw pool", "gain", "own dataset bootstrap CI95"))
for feat in ("axes", "pc:deception", "pc:unified@2"):
    f = aud["persona_lodo"][feat]
    ptv = np.array(f["lodo_per_target"], float)
    obs, lo, hi, pv = dsboot(ptv, raw_pt)
    p("  %-14s %10.4f %10.4f %12.4f %10.4f  [%+.3f, %+.3f]  P(>=0) %.4f\n"
      % (feat, f["single"], f["lodo"], f["lodo"] - raw_lodo, f["lodo"] - f["single"],
         lo, hi, 1 - pv))
p("\n  briefing states: axes 0.679/0.745/-0.109 CI[-0.187,-0.039];"
  " pc:deception 0.621/0.731/-0.124 CI[-0.203,-0.055];\n"
  "                   pc:unified@2 0.574/0.692/-0.162 CI[-0.288,-0.059];"
  " 'pc:unified@2 gains +0.118'\n")
p("  range of the three 'vs raw pooled' losses: %.4f to %.4f  (briefing '0.11-0.16')\n"
  % (min(aud["persona_lodo"][f]["lodo"] - raw_lodo for f in aud["persona_lodo"]),
     max(aud["persona_lodo"][f]["lodo"] - raw_lodo for f in aud["persona_lodo"])))

banner("O-028 / O-029 : mean-of-directions vs pooled LODO")
bs = aud["bootstrap"]["mean_of_directions-pooled_lodo_full"]
p("  stored: diff %.10f  lo %.10f  hi %.10f   (briefing +0.028, CI [+0.005, +0.052])\n"
  % (bs["diff"], bs["lo"], bs["hi"]))
mod = np.array(aud["per_target"]["mean_of_directions"], float)
obs, lo, hi, pv = dsboot(mod, raw_pt)
p("  recomputed from the 11 per-target values: %+.6f  own CI95 [%+.6f, %+.6f]  P(<=0) %.4f\n"
  % (obs, lo, hi, pv))
p("  scalar means: mean_of_directions %.10f   pooled_lodo_full %.10f   diff %+.10f\n"
  % (aud["mean_of_directions"], aud["pooled_lodo_full"],
     aud["mean_of_directions"] - aud["pooled_lodo_full"]))
p("  also in why_L40_s42.json: ")
why, path2, md52, n2 = load("why/why_L40_s42.json")
FILES.append((path2, n2, md52))
p("mean_of_directions %.10f  single_offdiag %.10f\n"
  % (why["mean_of_directions"], why["single_offdiag"]))

banner("FILES READ (path, bytes, md5)")
for path, n, md5 in FILES:
    p("  %-62s %9d  %s\n" % (path.replace("\\", "/"), n, md5))
