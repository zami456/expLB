"""W6D-ownerdocs: pooled-training claims O-037, O-042, O-043, O-044, O-045.

Reads the md5-verified lab copies cached under audit/cache/labpc/run-premise/.
Read-only. Prints every number the owner docs quote, with its file and key.
"""
import json, hashlib, os, sys

ROOT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
CACHE = os.path.join(ROOT, "audit/cache/labpc/run-premise")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(rel):
    p = os.path.join(CACHE, rel)
    print(f"\n--- {rel}  md5={md5(p)}  size={os.path.getsize(p)}")
    return json.load(open(p))


print("=" * 90)
print("O-042  REVIEW_REQUEST_POOLING.md:15  'the setting'")
print("=" * 90)
d = load("premise_L40_mean_s42.json")
print("layer        =", d["layer"])
print("pooling      =", d["pooling"])
print("probe        =", d["probe"])
print("seed         =", d["seed"])
print("n datasets   =", len(d["datasets"]))
print("datasets     =", d["datasets"])
print("n_train      =", d["n_train"])
print("n pairs      =", len(d["pairs"]))

print("\n-- split code, src/assistant_axis_ndif/30_premise.py (lines 79, 102-108, 114) --")
src = open(os.path.join(ROOT, "src/assistant_axis_ndif/30_premise.py")).read().splitlines()
for i in (78, 101, 102, 103, 104, 105, 106, 107, 113):
    print(f"{i+1:>4}: {src[i]}")
print("-> --frac default 0.8 (80/20) CONFIRMED")
print("-> split is rng.permutation(n) over ROWS; P.group_split (item id) is NOT called")
print("-> P.balance() is called on every training index set -> class-balanced CONFIRMED")
gs = [l for l in open(os.path.join(ROOT, "src/assistant_axis_ndif/9_persona_probes.py")).read().splitlines()
      if "def group_split" in l or "def balance" in l]
print("   helpers available in 9_persona_probes.py:", gs)
import subprocess
out = subprocess.run(["grep", "-c", "group_split", os.path.join(ROOT, "src/assistant_axis_ndif/30_premise.py")],
                     capture_output=True, text=True)
print("   occurrences of 'group_split' in 30_premise.py:", out.stdout.strip() or out.stderr.strip())

print("\n" + "=" * 90)
print("O-043  ':40'  full pooling vs budget-matched are indistinguishable, p = 0.38")
print("=" * 90)
s = d["summary"]
for k in ["single", "single_own_mean", "pooled_full", "pooled_match", "pooled_lodo", "pooled_lodo_match"]:
    print(f"  summary.{k:<20} = {s[k]:.6f}")
print("  n_train (single_mean / pooled_full / pooled_match) =", d["n_train"])

# dataset-level paired bootstrap, unit = target dataset, B=10000, seed 42 (F-80's recipe)
import numpy as np
per = d["per_dataset"]
tg = list(per.keys())
lodo = np.array([per[t]["pooled_lodo"] for t in tg])
lodom = np.array([per[t]["pooled_lodo_match"] for t in tg])
single = np.array([per[t]["single_offdiag"] for t in tg])
rng = np.random.default_rng(42)
B = 10000
n = len(tg)


def boot(a, b):
    obs = float(np.mean(a - b))
    draws = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        draws[i] = np.mean(a[idx] - b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p_le0 = float(np.mean(draws <= 0))
    return obs, lo, hi, p_le0


o, lo, hi, p = boot(lodo, lodom)
print(f"\n  VOLUME CONTROL  pooled_lodo - pooled_lodo_match = {o:+.4f}  [{lo:+.4f},{hi:+.4f}]  P(<=0) = {p:.4f}")
o2, lo2, hi2, p2 = boot(lodo, single)
print(f"  HEADLINE        pooled_lodo - single            = {o2:+.4f}  [{lo2:+.4f},{hi2:+.4f}]  P(<=0) = {p2:.4f}")
print(f"  wins (lodo > single) = {int(np.sum(lodo > single))}/{n}")

print("\n" + "=" * 90)
print("O-044  ':63'  'I compare it against full pooling (0.854) rather than the matched row (0.806)'")
print("=" * 90)
w = load("why/why_L40_s42.json")
print("  why_L40_s42.json  budget              =", w["budget"])
print("  why_L40_s42.json  pooled_lodo_matched =", w["pooled_lodo_matched"])
print("  why_L40_s42.json  mean_of_directions  =", w["mean_of_directions"])
print("  why_L40_s42.json  single_offdiag      =", w["single_offdiag"])
print("  premise_L40       summary.pooled_lodo_match =", s["pooled_lodo_match"])
print("  premise_L40       summary.pooled_lodo       =", s["pooled_lodo"])
print(f"  -> SAME NAME, TWO VALUES: 0.806 (30d_why_pooling) vs {s['pooled_lodo_match']:.4f} (30_premise), "
      f"diff {abs(w['pooled_lodo_matched'] - s['pooled_lodo_match']):.4f}")

a = load("audit/audit_L40_s42.json")
draws = np.array(a["pooled_lodo_match_draws"])
print(f"  30f audit: {a['draws']} independent budget-matched draws -> mean {draws.mean():.4f} "
      f"range [{draws.min():.4f},{draws.max():.4f}] sd(ddof=0) {draws.std():.4f}")
print(f"  both published figures inside that range? 0.806 -> {draws.min() <= 0.8062 <= draws.max()}; "
      f"0.850 -> {draws.min() <= 0.8500 <= draws.max()}")

print("\n" + "=" * 90)
print("O-045  ':95'  'our mean-of-directions IS a single direction at 0.883'")
print("=" * 90)
print("  why_L40_s42.json   mean_of_directions =", w["mean_of_directions"])
print("  audit_L40_s42.json mean_of_directions =", a["mean_of_directions"])
print("  audit pooled_lodo_full                =", a["pooled_lodo_full"])
bs = a["bootstrap"]["mean_of_directions-pooled_lodo_full"]
print("  stored bootstrap mean_of_directions - pooled_lodo_full =", json.dumps(bs))
pt = a["per_target"]
mod = np.asarray(pt["mean_of_directions"], dtype=float)
pf = np.asarray(pt["pooled_lodo_full"], dtype=float)
tgs = a["targets"]
print(f"  recomputed mean over 11 targets: mean_of_directions {mod.mean():.6f}  pooled_lodo_full {pf.mean():.6f}")
print(f"  wins {int(np.sum(mod > pf))}/11; losses:",
      [(tgs[i], round(float(mod[i] - pf[i]), 4)) for i in range(len(tgs)) if mod[i] < pf[i]])
print("  cos(single,single) =", w["cos_single_to_single_mean"], " cos(pooled,single) =", w["cos_pooled_to_single_mean"])

print("\n" + "=" * 90)
print("O-037  ROADMAP_TO_DEFENCE.md:27  median (0.722) / sign-oriented mean (0.735)")
print("=" * 90)
for k, v in a["baselines"].items():
    print(f"  baselines.{k:<20} = {v:.6f}")
print("  n_pairs =", a["n_pairs"], " pairs_below_half =", a["pairs_below_half"],
      " n datasets =", len(a["datasets"]))
print("  datasets in this file:", a["datasets"])
for k, v in a["bootstrap"].items():
    if k.startswith("pooled_lodo_full-"):
        print(f"  bootstrap.{k} = {json.dumps(v)}")
print("\n  NOTE: this file is the ELEVEN-dataset corpus (110 pairs). ROADMAP:27 places these two")
print("        numbers in section 0.1, whose baseline (0.6478/0.7320) is the COLLABORATOR'S SIX.")
