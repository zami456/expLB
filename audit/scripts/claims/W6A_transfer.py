"""W6A-thesis1 -- transfer / probe / protocol claims.

Covers O-058 (Q4 erasure cost), O-059 (Q6 layer medians), O-060 (Q8),
O-061 (Q11), O-062 (top-10% definition), O-063 (P1 diagonal/off-diagonal),
O-066 (length-only AUROC per dataset), O-070/O-071/O-072/O-073 (Q2 layer
tables), O-074 (the Q4 table).

Read-only; every number comes from a file under run-coords/ in the git working
tree, or from audit/cache/labpc/run-coords/ for the one file that is lab-only.

Usage: python audit/scripts/claims/W6A_transfer.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
RC = ROOT / "run-coords"
CACHE = ROOT / "audit/cache/labpc/run-coords"


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def md5(p):
    h = hashlib.md5()
    h.update(pathlib.Path(p).read_bytes())
    return h.hexdigest()


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


hdr("FILE PROVENANCE (md5 + size of every file this script reads)")
FILES = ["layer_sweep_passB_mms.json", "layer_sweep_passB_last.json",
         "layer_sweep_mms.json", "layer_sweep_mean.json", "q6_layers.json",
         "significance.json", "q3_q4_bootstrap.json", "pertoken_q8.json",
         "q8_trained_pertoken.json", "q11_sleeper_L44_last.json",
         "zeroshot_nulls.json", "splits.json", "p1_L44_last_withsyc.json"]
for f in FILES:
    p = RC / f
    print(f"{f:<32} {p.stat().st_size:>9} B  md5 {md5(p)}")
p = CACHE / "confounds_L44_last.json"
print(f"{'(cache) confounds_L44_last.json':<32} {p.stat().st_size:>9} B  md5 {md5(p)}")

# ---------------------------------------------------------------- O-059  Q6
hdr("O-059  Q6 -- median layer-40 gain over the better of 20/22")
q6 = load(RC / "q6_layers.json")
rows = q6["rows"]
print("n rows =", len(rows), " targets =", q6["targets"], " tol =", q6["tol"])
PERSONA_TOK = ("axes", "pc:", "PC1", "axis")
raw_rows, per_rows = [], []
for r in rows:
    m = r["method"]
    persona = any(t in m for t in PERSONA_TOK)
    (per_rows if persona else raw_rows).append(r)
print("\nclassified as PERSONA (%d):" % len(per_rows))
for r in per_rows:
    print(f"   {r['source']:<36} {r['method']:<42} "
          f"gain40={r['gap_40_minus_best_of_20_22']:+.4f}")
print("classified as RAW (%d):" % len(raw_rows))
for r in raw_rows:
    print(f"   {r['source']:<36} {r['method']:<42} "
          f"gain40={r['gap_40_minus_best_of_20_22']:+.4f}")
for tag, rs in (("raw", raw_rows), ("persona", per_rows)):
    g = sorted(r["gap_40_minus_best_of_20_22"] for r in rs)
    print(f"\n{tag}: n={len(g)} median={statistics.median(g):+.4f} "
          f"min={min(g):+.4f} max={max(g):+.4f}")
both = [r for r in rows if r["at"].get("22") is not None]
gaps = sorted(abs(r["gap_20_22"]) for r in both)
print(f"\n20-vs-22: n rows with both = {len(both)}  median |gap| = "
      f"{statistics.median(gaps):.4f}  max |gap| = {max(gaps):.4f}")
sig = load(RC / "significance.json")
print("Q6 equivalence tests (dataset bootstrap):")
for c in sig["comparisons"]:
    if c["kind"] == "equiv":
        ci = c.get("dataset_ci90") or c.get("dataset_ci95") or "(no CI key)"
        print("   ", c["name"], "diff %+.4f" % c["diff"],
              "equivalent_within_margin", c["equivalent_within_margin"],
              "row_equiv", c["row_equivalent_within_margin"],
              "keys", [k for k in c if k not in ("name", "question", "kind")])

# ------------------------------------------------- O-070 / O-072 / O-073  Q2
hdr("O-070  best-vs-best gaps (Pass B, full corpus, mms)")
mean_s = {r["layer"]: r for r in load(RC / "layer_sweep_passB_mms.json")["sweep"]}
last_s = {r["layer"]: r for r in load(RC / "layer_sweep_passB_last.json")["sweep"]}
for tag, s in (("mean", mean_s), ("last", last_s)):
    braw = max(s, key=lambda L: s[L]["raw"])
    bax = max(s, key=lambda L: s[L]["axes"])
    print(f"{tag}: raw peaks L{braw} {s[braw]['raw']:.4f}; axes peak L{bax} "
          f"{s[bax]['axes']:.4f}; axes@{bax} - raw@{braw} = "
          f"{s[bax]['axes'] - s[braw]['raw']:+.6f}")
print("\nsignificance.json best-vs-best comparisons:")
for c in sig["comparisons"]:
    if "each at its best" in c["name"]:
        print("   ", c["name"], "diff %+.6f" % c["diff"], "row_p", c["row_p"],
              "dataset_p", c["dataset_p"], "q_bh", c["dataset_q_bh"],
              "significant", c["significant"])

hdr("O-073  pc:deception minus raw, every stored layer and setting")
srcs = {"PassB mms mean": RC / "layer_sweep_passB_mms.json",
        "PassB mms last": RC / "layer_sweep_passB_last.json",
        "PassA mms mean": RC / "layer_sweep_mms.json",
        "PassA lr_reg10 mean": RC / "layer_sweep_mean.json"}
for tag, f in srcs.items():
    d = load(f)
    sw = d["sweep"]
    pos_ge20 = [(r["layer"], r["pc:deception"] - r["raw"]) for r in sw
                if r["layer"] >= 20 and r["pc:deception"] > r["raw"]]
    pos_lt20 = [(r["layer"], r["pc:deception"] - r["raw"]) for r in sw
                if r["layer"] < 20 and r["pc:deception"] > r["raw"]]
    best = max((r["pc:deception"] - r["raw"] for r in sw if r["layer"] >= 20))
    print(f"{tag:<22} probe={d['probe']:<9} layers={[r['layer'] for r in sw][:3]}..."
          f"{[r['layer'] for r in sw][-1]}  n={len(sw)}")
    print(f"   layers>=20 where pc beats raw: {pos_ge20}   "
          f"max(pc-raw) at layers>=20 = {best:+.4f}")
    print(f"   layers<20  where pc beats raw: "
          f"{[(l, round(v,4)) for l, v in pos_lt20]}")
    if pos_lt20:
        print(f"   max gain below 20 = {max(v for _, v in pos_lt20):+.4f}")

hdr("O-071 / O-072  Pass A (2,000 rows)")
for tag, f in (("Pass A mms mean", RC / "layer_sweep_mms.json"),
               ("Pass A lr_reg10 mean", RC / "layer_sweep_mean.json")):
    d = load(f)
    s = {r["layer"]: r for r in d["sweep"]}
    braw = max(s, key=lambda L: s[L]["raw"])
    bax = max(s, key=lambda L: s[L]["axes"])
    print(f"\n{tag} (probe={d['probe']}, bank={d['bank']}, pooling={d['pooling']}):")
    print(f"   raw peak  L{braw} {s[braw]['raw']:.4f}")
    print(f"   axes peak L{bax} {s[bax]['axes']:.4f}")
    print(f"   best-vs-best axes - raw = {s[bax]['axes'] - s[braw]['raw']:+.4f}")
    print("   layer   raw      axes     axes-raw")
    for L in sorted(s):
        if L >= 20:
            print(f"   {L:<6} {s[L]['raw']:.4f}  {s[L]['axes']:.4f}  "
                  f"{s[L]['axes'] - s[L]['raw']:+.4f}")

# -------------------------------------------------------- O-058 / O-074  Q4
hdr("O-058 / O-074  Q4 erasure table")
q34 = load(RC / "q3_q4_bootstrap.json")
q4 = q34["q4_table"]
print("rows in q4_table:", len(q4))
print(f"{'layer':>5} {'pool':>5} {'probe':>9} {'raw':>8} {'erased':>8} {'cost':>8} "
      f"{'cost_random(mean)':>18} {'erased dataset CI95':>26} {'q_bh':>8}")
for r in q4:
    cr = r.get("cost_random")
    crm = (sum(cr) / len(cr)) if isinstance(cr, list) else cr
    ci = r.get("erased_dataset_ci95")
    want = (f"cost of erasing Assistant Axis, L{r['layer']} {r['pooling']} "
            f"{r['probe']}")
    q = next((c["dataset_q_bh"] for c in q34["comparisons"]
              if c["name"] == want), None)
    print(f"{r['layer']:>5} {r['pooling']:>5} {r['probe']:>9} {r['raw']:>8.4f} "
          f"{r['erase_assistant']:>8.4f} {r['cost_assistant']:>8.4f} "
          f"{crm:>18.6f} {str([round(x,3) for x in ci]):>26} "
          f"{('%.4f' % q) if q is not None else 'NA':>8}")
costs = [r["cost_assistant"] for r in q4]
print("\ncost range over the 6 settings: %.4f .. %.4f" % (min(costs), max(costs)))
print("all costs positive:", all(c > 0 for c in costs))
rel = [(r["cost_assistant"] / r["raw"] * 100) for r in q4]
print("cost as %% of the raw transfer AUROC: %.2f%% .. %.2f%%  (doc's verdict says "
      "'1-3%%')" % (min(rel), max(rel)))
print("   per setting:", [f"{r['layer']}{r['pooling'][0]}/{r['probe']}={v:.2f}%"
                         for r, v in zip(q4, rel)])
print("\nQ4 comparisons in the same file:")
for c in q34["comparisons"]:
    if "eras" in c["name"].lower():
        print("   ", c["name"], "| diff %+.5f" % c.get("diff", float("nan")),
              "| row_p", c.get("row_p"), "| dataset_p", c.get("dataset_p"),
              "| q_bh %.4f" % c["dataset_q_bh"], "| significant", c["significant"])
print("\nBH family size in this file (n comparisons):", len(q34["comparisons"]))

# ---------------------------------------------------------------- O-060  Q8
hdr("O-060  Q8 -- zero-shot half (pertoken_q8.json)")
pt = load(RC / "pertoken_q8.json")
print("axis =", pt["axis"], " bank =", pt["bank"])
print(f"{'layer':>6} {'mean':>8} {'top10pct':>10} {'top10-mean':>11} "
      f"{'max':>8} {'last':>8}")
for a in pt["aggregations"]:
    print(f"{a['layer']:>6} {a['mean']:>8.4f} {a['top10pct']:>10.4f} "
          f"{a['top10pct']-a['mean']:>+11.4f} {a['max']:>8.4f} {a['last']:>8.4f}")
print("top10pct > mean at every stored layer:",
      all(a["top10pct"] > a["mean"] for a in pt["aggregations"]))
print("top10pct is the best aggregation at layer:",
      [a["layer"] for a in pt["aggregations"]
       if a["top10pct"] == max(a[k] for k in
                               ("mean", "last", "last8", "max", "top10pct", "first"))])

hdr("O-060  Q8 -- trained half (q8_trained_pertoken.json)")
q8 = load(RC / "q8_trained_pertoken.json")
res = q8["results"]
by = {}
for r in res:
    by.setdefault(r["layer"], []).append(r)
print("layers:", q8["layers"], " held_out:", len(q8["held_out"]),
      " tokens_per_row:", q8["tokens_per_row"], " n_rows:", q8["n_rows"])
print(f"\n{'layer':>6} {'best pooled':>32} {'best token probe':>34} {'gap':>8}")
for L in q8["layers"]:
    pool = [r for r in by[L] if r["probe"].startswith("pool_")]
    tok = [r for r in by[L] if r["probe"].startswith("tok_")]
    bp = max(pool, key=lambda r: r["auroc_mean"])
    bt = max(tok, key=lambda r: r["auroc_mean"])
    print(f"{L:>6} {bp['probe']+'/'+bp['aggregation']+' '+format(bp['auroc_mean'],'.4f'):>32} "
          f"{bt['probe']+'/'+bt['aggregation']+' '+format(bt['auroc_mean'],'.4f'):>34} "
          f"{bp['auroc_mean']-bt['auroc_mean']:>+8.4f}")
print("\npool_mean_mms vs EVERY token probe, per layer (min gap):")
for L in q8["layers"]:
    pm = [r for r in by[L] if r["probe"] == "pool_mean_mms"][0]
    tok = [r for r in by[L] if r["probe"].startswith("tok_")]
    print(f"   L{L:<3} pooled-mean {pm['auroc_mean']:.4f}  worst-case gap "
          f"{min(pm['auroc_mean']-t['auroc_mean'] for t in tok):+.4f}  "
          f"(n token rows {len(tok)})")
print("\nzero-shot axis rows (not token PROBES) for context:")
for L in q8["layers"]:
    zs = [r for r in by[L] if r["probe"] == "zeroshot_axis"]
    pm = [r for r in by[L] if r["probe"] == "pool_mean_mms"][0]
    bz = max(zs, key=lambda r: r["auroc_mean"])
    print(f"   L{L:<3} best zero-shot {bz['aggregation']:>8} {bz['auroc_mean']:.4f} "
          f"vs pooled-mean {pm['auroc_mean']:.4f} -> "
          f"{pm['auroc_mean']-bz['auroc_mean']:+.4f}")
print("\nQ8 significance tests:")
for c in sig["comparisons"]:
    if c["question"] == "Q8":
        print("   ", c["name"], "| diff %+.4f" % c["diff"], "| dataset_p",
              c["dataset_p"], "| folds_pooled_wins %d/%d" % (c["folds_pooled_wins"],
                                                            c["folds"]),
              "| q_bh %.5f" % c["dataset_q_bh"], "| significant", c["significant"])
print("\nself checks:", json.dumps(q8["self_checks"])[:400])

# --------------------------------------------------------------- O-061  Q11
hdr("O-061  Q11 sleeper agents")
q11 = load(RC / "q11_sleeper_L44_last.json")
print("layer", q11["layer"], "pooling", q11["pooling"], "probe", q11["probe"],
      "bank", q11["bank"])
ct = q11["tests"]["cross_trigger"]
wo = q11["tests"]["within_organism"]
raw_ct = [v for k, v in ct.items() if k.startswith("raw|")]
raw_wo = [v for k, v in wo.items() if k.startswith("raw|soft-trigger")]
print("cross_trigger raw:", {k: round(v, 4) for k, v in ct.items() if k.startswith("raw|")},
      "mean %.4f" % (sum(raw_ct) / len(raw_ct)))
print("within_organism raw (soft-trigger only):",
      {k: round(v, 4) for k, v in wo.items() if k.startswith("raw|soft-trigger")},
      "mean %.4f" % (sum(raw_wo) / len(raw_wo)))
print("within_organism raw|gender-secret: %.4f" % wo["raw|gender-secret"])
print("token_only (answer token alone):",
      {k.split('|')[1]: v for k, v in q11["tests"]["token_stratified"].items()
       if k.startswith("token_only")})
ts = q11["tests"]["token_stratified"]
within_tok = [v["probe_from_instructed-deception"] for k, v in ts.items()
              if k.startswith("raw|soft-trigger") and isinstance(v, dict)]
print("probe from instructed-deception scored WITHIN one answer token, raw: "
      "%.4f .. %.4f" % (min(within_tok), max(within_tok)))
print("zero-shot persona axis on sleepers:",
      {k: round(v, 4) for k, v in q11["tests"]["zeroshot"].items()})
nulls = load(RC / "zeroshot_nulls.json")["sleeper_L44_last"]
for k, v in nulls.items():
    da = v["deception_axis"]
    print(f"   {k:<24} auroc {da['auroc']:.4f}  random p95 "
          f"{v['random_p95_auroc_dev']:.4f}  p_vs_random {da['p_vs_random']:.4f}  "
          f"beats_random_p95 {da['beats_random_p95']}")

# ---------------------------------------------------- O-062 / O-063 protocol
hdr("O-062  top-10% definition, read from the producing code")
src = (ROOT / "src/assistant_axis_ndif/14_pertoken.py").read_text(encoding="utf-8")
i = src.index("def aggregate")
print("14_pertoken.py:")
print(src[i:i + 620])
src2 = (ROOT / "src/assistant_axis_ndif/21_q8_trained_pertoken.py").read_text(encoding="utf-8")
j = src2.index("def aggregate")
print("21_q8_trained_pertoken.py:")
print(src2[j:j + 330])

hdr("O-063  P1 diagonal / off-diagonal, read from 9_persona_probes.py + splits.json")
src3 = (ROOT / "src/assistant_axis_ndif/9_persona_probes.py").read_text(encoding="utf-8")
k = src3.index("def group_split")
print(src3[k:k + 620])
k2 = src3.index("for s in src:")
print("... P1 loop ...")
print(src3[k2:k2 + 1150])
sp = load(RC / "splits.json")
print("\nsplits.json realised sizes (key 'splits'):")
s = sp["splits"]
if isinstance(s, dict):
    for k3, v in s.items():
        print("   ", k3, v)

print("\nLEVEL 2 -- recompute 'off_diagonal' and 'diagonal' from the stored matrix "
      "(p1_L44_last_withsyc.json):")
import math  # noqa: E402
pw = load(RC / "p1_L44_last_withsyc.json")
print("   datasets in file (%d):" % len(pw["datasets"]), sorted(pw["datasets"]))
for r in pw["p1"]:
    m = r["matrix"]
    off = [v for kk, v in m.items()
           if kk.split("->")[0] != kk.split("->")[1]
           and not (isinstance(v, float) and math.isnan(v))]
    dia = [v for kk, v in m.items()
           if kk.split("->")[0] == kk.split("->")[1]
           and not (isinstance(v, float) and math.isnan(v))]
    print(f"   {r['features']:<14} {r['probe']:<9} n_pairs={r['n_pairs']:<4} "
          f"off stored {r['off_diagonal']:.6f} = recomputed "
          f"{sum(off)/len(off):.6f} over {len(off)} cells | diag stored "
          f"{r['diagonal']:.6f} = {sum(dia)/len(dia):.6f} over {len(dia)} cells")

# ------------------------------------------------------------- O-066 length
hdr("O-066  length-only AUROC per dataset")
print("The doc's path run-coords/confounds_L44_last.json exists in the working tree:",
      (RC / "confounds_L44_last.json").exists())
print("read instead from the audit cache of the lab copy:",
      CACHE / "confounds_L44_last.json")
cf = load(CACHE / "confounds_L44_last.json")
print("top-level keys:", list(cf.keys()),
      "| layer", cf["layer"], "| pooling", cf["pooling"], "| bank", cf["bank"])
print("the doc names the key 'length_only_auroc'; it is actually nested at "
      "confounds.length_only_auroc, a LIST of {dataset, auroc_from_length, "
      "median_pos, median_neg}")
la = {r["dataset"]: r for r in cf["confounds"]["length_only_auroc"]}
doc = {"sandbagging_v2": 0.854, "harm-pressure-knowledge-report": 0.776,
       "sycophancy_mt": 0.664, "mask": 0.625, "ai_liar": 0.613,
       "insider-trading": 0.595, "instructed-deception": 0.560,
       "convincing-game": 0.507, "harm-pressure-choice": 0.502,
       "roleplaying": 0.441}
print("\nDOC vs FILE (tolerance 5e-4):")
allok = True
for ds, dv in doc.items():
    fv = la.get(ds, {}).get("auroc_from_length")
    ok = (fv is not None and abs(fv - dv) <= 5e-4)
    allok &= ok
    print(f"   {ds:<34} doc {dv:<7} file "
          f"{('%.6f' % fv) if fv is not None else 'ABSENT':<12} "
          f"{'OK' if ok else 'MISMATCH'}")
print("   all ten match:", allok)
print("   datasets in the file but not in the doc list:",
      [d for d in la if d not in doc])
print("   n datasets with a length AUROC:", len(la), "| above 0.5:",
      sum(1 for v in la.values() if v["auroc_from_length"] > 0.5),
      "| below 0.5:", [d for d, v in la.items() if v["auroc_from_length"] < 0.5])
print("   the file's own dataset list (%d):" % len(cf["datasets"]),
      sorted(cf["datasets"]))
gap = [(d, v["median_pos"] - v["median_neg"]) for d, v in la.items()]
print("\n   MEDIAN token gap (deceptive - honest) in this same file:")
for d, g in sorted(gap, key=lambda x: -x[1]):
    print(f"      {d:<34} {g:+.1f}")
print("   deceptive longer by median in %d of %d; equal in %d; shorter in %d"
      % (sum(1 for _, g in gap if g > 0), len(gap),
         sum(1 for _, g in gap if g == 0), sum(1 for _, g in gap if g < 0)))

print("\nDONE")
