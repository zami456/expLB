#!/usr/bin/env python
"""Cluster-based permutation test on the kd matched-slice layer profile.

The one test `docs/LAYER_PREREGISTRATION.md` committed to that has never run. Quoting it:

    Cluster-based permutation test (Maris & Oostenveld 2007), which is the standard for exactly
    this shape of problem. Each label permutation is applied IDENTICALLY to all layers, so the
    null preserves the between-layer correlation rather than treating 80 layers as 80
    independent tests.

WHY IT IS A DIFFERENT TEST FROM F-76. The dataset bootstrap asks "would this survive a different
choice of evaluation datasets". This asks "could a profile of this shape arise from labels that
carry no signal at all". A claim can fail one and pass the other; neither substitutes.

WHY IT NEEDS REFITS. The null is over LABELS, so each permutation has to retrain the probe on
shuffled labels at every layer. That is the expense the earlier runs deferred.

THE STATISTIC. For each layer, the margin of `axes` over the matched control. Contiguous runs of
positive margin form clusters; a cluster's mass is the sum of its margins. The observed largest
cluster mass is compared against the distribution of largest cluster masses under permutation.
This is what makes it a test of the BAND rather than of any single layer, and it is why
contiguity -- which F-76 showed is not evidence on its own -- becomes interpretable here.

Prediction, written before the run: F-76 found the margin at the pre-specified layer does not
clear the dataset bootstrap and the argmax is unidentified across 93% of the network. If that
reading is right, the observed cluster mass should sit WELL INSIDE the permutation null, p > 0.2.
A small p would mean the band is real against label noise even though it is fragile against
dataset choice -- which would be a more interesting result than either test alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
DS = "harm-pressure-kr-matched"


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cluster_masses(margins):
    """Contiguous runs of positive margin -> summed mass of each."""
    out, cur = [], 0.0
    for v in margins:
        if v > 0:
            cur += v
        elif cur > 0:
            out.append(cur); cur = 0.0
    if cur > 0:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="run-coords/passA")
    ap.add_argument("--out_root", default="run-premise")
    ap.add_argument("--layers", nargs="+", type=int,
                    default=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                             64, 68, 72, 76])
    ap.add_argument("--others", nargs="+", default=[
        "instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "convincing-game", "insider-trading", "ai_liar", "sandbagging_v2", "roleplaying",
        "mask", "sycophancy_mt"])
    ap.add_argument("-B", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bank", default="run-kd/axes_vectors_s23.pt")
    args = ap.parse_args()

    P = module("persona_probes", "9_persona_probes.py")
    keys = [DS] + list(args.others)
    rng = np.random.default_rng(args.seed)

    # Load every layer once; the permutation loop only refits.
    per_layer = {}
    for L in args.layers:
        data = P.build(Path(args.run_dir), keys, L, "mean")
        if DS not in data:
            continue
        per_layer[L] = {k: (np.asarray(v[0], dtype=np.float32), v[1]) for k, v in data.items()}
    layers = sorted(per_layer)
    print(f"{len(layers)} layers loaded; B={args.B} permutations, k={args.k}")

    import torch
    bank = torch.load(REPO_ROOT / args.bank, map_location="cpu", weights_only=False)

    def margin_profile(labels_by_layer):
        """Margin of the bank's stacked axes over a MATCHED RANDOM control, per layer.

        The control is `random` at the same k, built by make_features from the same seed --
        a genuine matched control that needs no per-source refitting. (An earlier draft fitted
        the "control" on the identical features as the probe, which makes the margin
        identically zero; caught before it ran.)
        """
        out = []
        for L in layers:
            d = per_layer[L]
            ys = labels_by_layer[L]
            tr = P.balance(np.arange(len(ys)), ys, args.seed)
            vals = {}
            for kind in ("axes", "random"):
                fs = P.make_features(d[DS][0], kind, bank, L, args.k, args.seed)
                clf = P.make_probe("paper", args.seed).fit(fs[tr], ys[tr])
                sc = [P.auroc(clf, P.make_features(d[t][0], kind, bank, L, args.k, args.seed),
                              d[t][1]) for t in args.others if t in d]
                vals[kind] = float(np.nanmean(sc))
            out.append(vals["axes"] - vals["random"])
        return np.array(out)

    true_labels = {L: per_layer[L][DS][1] for L in layers}
    obs = margin_profile(true_labels)
    obs_clusters = cluster_masses(obs)
    obs_max = max(obs_clusters) if obs_clusters else 0.0
    print(f"observed: {int((obs > 0).sum())}/{len(layers)} layers positive, "
          f"largest cluster mass {obs_max:.4f}")

    null = []
    y0 = true_labels[layers[0]]
    for b in range(args.B):
        perm = rng.permutation(len(y0))
        # THE SAME permutation at every layer, so the null keeps between-layer correlation
        lab = {L: true_labels[L][perm] for L in layers}
        m = margin_profile(lab)
        cl = cluster_masses(m)
        null.append(max(cl) if cl else 0.0)
        if (b + 1) % 25 == 0:
            print(f"  {b + 1}/{args.B}")
    null = np.array(null)
    p = float((null >= obs_max).mean())

    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    (out / "cluster_perm_kd_matched.json").write_text(json.dumps({
        "layers": layers, "observed_margins": obs.tolist(),
        "observed_clusters": obs_clusters, "observed_max_mass": obs_max,
        "null_max_mass": null.tolist(), "p": p, "B": args.B}, indent=1), encoding="utf-8")

    print(f"\ncluster-mass p = {p:.4f}   (null mean {null.mean():.4f}, "
          f"95th pct {np.percentile(null, 95):.4f})")
    print("=> band is not distinguishable from label noise" if p > 0.05
          else "=> band survives the label permutation")


if __name__ == "__main__":
    main()
