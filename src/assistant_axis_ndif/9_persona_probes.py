#!/usr/bin/env python
"""Persona-coordinate probes over the Pass B activations (plan questions Q2-Q6, Q10).

Reads run-coords/passB/<dataset>/{rows.jsonl, chunk_*.pt} and asks whether a probe on k
persona coordinates transfers across datasets better than one on the raw 8192-d pooled
activation -- the paper's central claim -- and whether any of it survives removing the
Assistant Axis.

    python 9_persona_probes.py --protocol p1 --layer 40 --pooling mean
    python 9_persona_probes.py --protocol p2 --layer 40
    python 9_persona_probes.py --protocol p3 --layer 40

Design decisions that matter, and why:

* **Splits group on item_id, never row_id.** ai_liar is 54 scenarios x 5 samples, and
  instructed-deception recycles dataset_index across five subsets; splitting on rows would
  put near-duplicates of one scenario on both sides and inflate every diagonal.
* **label=None rows are dropped, never coerced.** roleplaying's 3-5 grades are genuinely
  ambiguous under Apollo's own rule.
* **alpaca is never a training source.** It is 2,000 honest rows with no positives; it exists
  to set the TPR@1%FPR threshold, and a "probe" trained on one class is not a probe.
* **Axes are normalised before projection.** They arrive with norms 1.0-2.5 because
  residual-stream norm grows with depth, so an unnormalised projection would rank layers by
  norm rather than by signal.
* **Two baselines, reported side by side.** The paper's own C=1 with no scaler will overfit
  8192 dims on splits as small as 268 rows -- which is exactly the strong-diagonal /
  weak-off-diagonal pattern that PC projection then appears to repair. Apollo's tuned
  settings are the fair comparison. If persona-PC beats the paper's baseline and loses to
  Apollo's, that is the finding, and it gets said plainly.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# sklearn 1.8 deprecated the `penalty` kwarg and emits a paragraph per fit; with a
# transfer matrix that is hundreds of fits and it buries every result line.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL = "alpaca"          # honest-only: threshold source, never a training source


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_dataset(run_dir: Path, key: str, layer: int, pooling: str):
    """-> X (n, H) float32, meta list. One layer and one pooling at a time.

    The store is 21 layers x 8192 x 2 poolings x 2 bytes = 688 KB per row, about 10 GB over
    the corpus. A probe needs one layer and one pooling, which is ~500 MB as float32, so
    slice on the way in rather than loading everything and indexing after.
    """
    d = run_dir / key
    rows_path = d / "rows.jsonl"
    chunks = sorted(d.glob("chunk_*.pt"))
    if not rows_path.exists() or not chunks:
        return None, None

    meta = {}
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["row_id"]] = r

    xs, keep, pool_idx = [], [], None
    for c in chunks:
        blob = torch.load(c, weights_only=False)
        layers = list(blob["layers"])
        if layer not in layers:
            raise SystemExit(f"layer {layer} not stored in {c.name}; have {layers}")
        li = layers.index(layer)
        if pool_idx is None:
            pools = list(blob.get("pooling", ["mean", "last"]))
            if pooling not in pools:
                raise SystemExit(f"pooling {pooling!r} not stored; have {pools}")
            pool_idx = pools.index(pooling)
        pooled = blob["pooled"][:, li, :, pool_idx].to(torch.float32).numpy()
        for i, rid in enumerate(blob["row_ids"]):
            m = meta.get(rid)
            if m is None or m.get("label") is None:
                continue
            xs.append(pooled[i])
            keep.append(m)
    if not xs:
        return None, None
    return np.stack(xs), keep


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_features(X, kind: str, bank, layer: int, k: int, seed: int):
    """kind: raw | pc:<behaviour> | axes | axes_resid | random"""
    if kind == "raw":
        return X
    if kind == "random":
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], k)))
        return X @ Q
    if kind.startswith("pc:"):
        beh = kind.split(":", 1)[1]
        comps = bank["pca"][beh]["components"][layer, :k].numpy()
        comps = np.stack([unit(c) for c in comps])
        return X @ comps.T
    if kind.startswith("erase:"):
        # Erase a direction from the ACTIVATIONS, then probe the full 8192-d space, rather
        # than residualising the axis and probing its projection. The nnsight probing skill
        # is explicit about why: after any projection the ORIGINAL probe fails by
        # arithmetic, since P w is nearly zero, so only a probe refitted on erased
        # activations says whether the concept is gone for ANY linear probe. axes_resid
        # answers the weaker question.
        #
        # This is a representational claim, not a causal one -- the erasure happens after
        # the forward pass, so it says what is linearly decodable from the state, not what
        # the model does with it. Q4 asks exactly the representational question: is there
        # deception signal in the activations beyond the assistant direction?
        what = kind.split(":", 1)[1]
        if what == "random":
            # rank-matched control: remove ONE arbitrary direction, same rank as the real
            # erasure. It tells you the result is about WHICH direction, not how many.
            rng = np.random.default_rng(seed)
            d = rng.standard_normal(X.shape[1])
        else:
            d = bank["axes"][what][layer].numpy()
        d = unit(d)
        return X - np.outer(X @ d, d)
    if kind in ("axes", "axes_resid"):
        ax = bank["axes"]
        dec = unit(ax["deception"][layer].numpy())
        asst = unit(ax["assistant"][layer].numpy())
        syc = unit(ax["sycophancy"][layer].numpy())
        uni = unit(ax["unified"][layer].numpy())
        if kind == "axes_resid":
            # d - <d,a>a. Whether ANY deception signal survives removing the assistant
            # component is the single highest-information experiment in the plan (Q4):
            # if detection collapses here, the axis was measuring assistant-ness all along.
            dec = unit(dec - float(dec @ asst) * asst)
            syc = unit(syc - float(syc @ asst) * asst)
            uni = unit(uni - float(uni @ asst) * asst)
            dirs = [dec, syc, uni]
        else:
            dirs = [dec, syc, uni, asst]
        return X @ np.stack(dirs).T
    raise ValueError(f"unknown feature kind {kind!r}")


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
class MeanMassShift(BaseEstimator, ClassifierMixin):
    """Apollo's "mms" method: score along the difference of class means.

    descriptive.yaml uses this rather than logistic regression, and it is the standard
    strong baseline in this literature -- no fitting beyond two means, so it cannot overfit
    8192 dimensions the way the paper's C=1 does.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.w_ = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        return self

    def decision_function(self, X):
        return X @ self.w_


def make_probe(name: str, seed: int):
    if name == "paper":
        # the persona-coordinates paper's own setting: C=1, intercept, no scaler
        return LogisticRegression(C=1.0, fit_intercept=True,
                                  max_iter=1000, random_state=seed)
    if name in ("lr_reg10", "apollo"):
        # THE deception-probe literature's standard config, and it is shared: Apollo's
        # roleplaying.yaml and descriptive.yaml use detect_layers [22], Liars' Bench's own
        # configs use detect_layers [20], but BOTH are method "lr" with reg_coeff 10. So the
        # difference between "the Apollo probe" and "the Liars' Bench probe" is the layer,
        # not the estimator -- which is why the layer is a separate axis here (--layer) and
        # is exactly what Q6 asks about.
        #
        # fit_intercept=False, matching detectors.py:476 -- LogisticRegression(
        # C=1/reg_coeff, random_state=42, fit_intercept=False) -- after StandardScaler,
        # which centres the features so an intercept is largely redundant anyway.
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, fit_intercept=False,
                                                max_iter=1000, random_state=seed))
    if name == "mms":
        return make_pipeline(StandardScaler(), MeanMassShift())
    raise ValueError(name)


def _scores(clf, X):
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def auroc(clf, X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _scores(clf, X)))


def tpr_at_fpr(clf, X_pos, X_control, fpr: float = 0.01):
    """TPR at a threshold set on the honest-only control, not on the test set itself."""
    if len(X_control) == 0 or len(X_pos) == 0:
        return float("nan")
    thr = float(np.quantile(_scores(clf, X_control), 1.0 - fpr))
    return float((_scores(clf, X_pos) > thr).mean())


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def group_split(meta, frac: float, seed: int):
    """Split indices by item_id so no scenario straddles train and test."""
    by_item = defaultdict(list)
    for i, m in enumerate(meta):
        by_item[m["item_id"]].append(i)
    items = sorted(by_item)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    cut = max(1, int(round(len(items) * frac)))
    tr = [i for it in items[:cut] for i in by_item[it]]
    te = [i for it in items[cut:] for i in by_item[it]]
    return np.array(sorted(tr), dtype=int), np.array(sorted(te), dtype=int)


def balance(idx, y, seed: int, cap: int = 0):
    """Equal positives and negatives, deterministically."""
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    if cap:
        n = min(n, cap // 2)
    if n == 0:
        return np.array([], dtype=int)
    return np.concatenate([rng.choice(pos, n, replace=False),
                           rng.choice(neg, n, replace=False)])


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def build(run_dir: Path, keys, layer: int, pooling: str):
    data = {}
    for k in keys:
        X, meta = load_dataset(run_dir, k, layer, pooling)
        if X is None:
            logger.warning("  %s: no usable rows yet, skipping", k)
            continue
        y = np.array([m["label"] for m in meta], dtype=int)
        data[k] = (X, y, meta)
        sem = {m.get("label_semantics") for m in meta}
        if len(sem) > 1:
            logger.warning("  %s mixes label semantics %s", k, sorted(sem))
        logger.info("  %-32s n=%-6d %dD/%dH", k, len(y), int(y.sum()), int((1 - y).sum()))
    return data


def sources(data):
    """Datasets usable for TRAINING: both classes present, and not the control."""
    return [k for k, (_, y, _) in data.items()
            if k != CONTROL and len(np.unique(y)) == 2]


def p1_transfer(data, bank, args):
    """75/25 within each source, then test on every other dataset's held-out pool."""
    feats = args.features
    rows = []
    src = sources(data)
    sem_of = {k: (meta[0].get("label_semantics") or "?") for k, (_, _, meta) in data.items()}
    present = sorted(set(sem_of.values()))
    if len(present) < 2:
        logger.warning("  only one label semantics present (%s) -- the within/across split "
                       "is vacuous until roleplaying (judge_grade) or mask "
                       "(belief_inconsistency) finish extracting", present[0])
    for fkind in feats:
        for probe in args.probes:
            mat = {}
            for s in src:
                Xs, ys, ms = data[s]
                tr, _ = group_split(ms, 0.75, args.seed)
                tr = balance(tr, ys, args.seed)
                if len(tr) < 8:
                    continue
                F = make_features(Xs, fkind, bank, args.layer, args.k, args.seed)
                clf = make_probe(probe, args.seed)
                clf.fit(F[tr], ys[tr])
                for t in data:
                    Xt, yt, mt = data[t]
                    if t == s:
                        _, te = group_split(mt, 0.75, args.seed)
                    else:
                        te = np.arange(len(yt))
                    te = balance(te, yt, args.seed + 1)
                    if len(te) < 8:
                        mat[(s, t)] = float("nan")
                        continue
                    Ft = make_features(Xt, fkind, bank, args.layer, args.k, args.seed)
                    mat[(s, t)] = auroc(clf, Ft[te], yt[te])
            diag = [v for (s, t), v in mat.items() if s == t and not np.isnan(v)]
            off = [v for (s, t), v in mat.items() if s != t and not np.isnan(v)]
            # Q5: transfer WITHIN a label definition vs ACROSS one. The three label
            # semantics are different constructs -- instructed_intent (the prompt told it to
            # lie), judge_grade (a judge said it lied), belief_inconsistency (it contradicted
            # its own elicited belief) -- so an off-diagonal mean that pools them can look
            # healthy purely because most pairs share a definition. Splitting the mean is
            # the actual test, and it is the difference between "the method transfers" and
            # "the method transfers within one notion of deception".
            same = [v for (s, t), v in mat.items()
                    if s != t and not np.isnan(v) and sem_of[s] == sem_of[t]]
            cross = [v for (s, t), v in mat.items()
                     if s != t and not np.isnan(v) and sem_of[s] != sem_of[t]]
            rows.append({"features": fkind, "probe": probe,
                         "diagonal": float(np.mean(diag)) if diag else float("nan"),
                         "off_diagonal": float(np.mean(off)) if off else float("nan"),
                         "within_semantics": float(np.mean(same)) if same else float("nan"),
                         "across_semantics": float(np.mean(cross)) if cross else float("nan"),
                         "n_within": len(same), "n_across": len(cross),
                         "n_pairs": len(off),
                         "matrix": {f"{s}->{t}": v for (s, t), v in mat.items()}})
            logger.info("  %-16s %-7s diag=%.3f off=%.3f  within-sem=%.3f (%d) "
                        "across-sem=%.3f (%d)",
                        fkind, probe, rows[-1]["diagonal"], rows[-1]["off_diagonal"],
                        rows[-1]["within_semantics"], len(same),
                        rows[-1]["across_semantics"], len(cross))
    return rows


def p2_curves(data, bank, args):
    """Train on n, test on the FULL balanced pool of every other dataset. 5 seeds."""
    ns = [16, 32, 64, 128, 256, 512, 1024]
    out = []
    src = sources(data)
    for fkind in args.features:
        for probe in args.probes:
            for n in ns:
                vals = []
                for seed in range(args.seeds):
                    for s in src:
                        Xs, ys, ms = data[s]
                        tr, _ = group_split(ms, 0.75, seed)
                        tr = balance(tr, ys, seed, cap=n)
                        if len(tr) < min(8, n):
                            continue
                        F = make_features(Xs, fkind, bank, args.layer, args.k, seed)
                        clf = make_probe(probe, seed)
                        clf.fit(F[tr], ys[tr])
                        for t in data:
                            if t == s:
                                continue
                            Xt, yt, _ = data[t]
                            te = balance(np.arange(len(yt)), yt, seed + 1)
                            if len(te) < 8:
                                continue
                            Ft = make_features(Xt, fkind, bank, args.layer, args.k, seed)
                            vals.append(auroc(clf, Ft[te], yt[te]))
                vals = [v for v in vals if not np.isnan(v)]
                if not vals:
                    continue
                out.append({"features": fkind, "probe": probe, "n": n,
                            "auroc_mean": float(np.mean(vals)),
                            "auroc_sd": float(np.std(vals)), "n_obs": len(vals)})
                logger.info("  %-16s %-7s n=%-5d off-diag AUROC %.3f +/- %.3f",
                            fkind, probe, n, out[-1]["auroc_mean"], out[-1]["auroc_sd"])
    return out


def p3_loso(data, bank, args):
    """Leave one dataset out: train on all other sources pooled, test on the held-out one."""
    out = []
    src = sources(data)
    for fkind in args.features:
        for probe in args.probes:
            for held in data:
                tr_X, tr_y = [], []
                for s in src:
                    if s == held:
                        continue
                    Xs, ys, ms = data[s]
                    idx = balance(np.arange(len(ys)), ys, args.seed)
                    if len(idx) == 0:
                        continue
                    tr_X.append(make_features(Xs, fkind, bank, args.layer,
                                              args.k, args.seed)[idx])
                    tr_y.append(ys[idx])
                if not tr_X:
                    continue
                clf = make_probe(probe, args.seed)
                clf.fit(np.concatenate(tr_X), np.concatenate(tr_y))
                Xh, yh, _ = data[held]
                te = balance(np.arange(len(yh)), yh, args.seed + 1)
                if len(te) < 8:
                    continue
                Fh = make_features(Xh, fkind, bank, args.layer, args.k, args.seed)
                a = auroc(clf, Fh[te], yh[te])
                tpr = float("nan")
                if CONTROL in data:
                    Xc, _, _ = data[CONTROL]
                    Fc = make_features(Xc, fkind, bank, args.layer, args.k, args.seed)
                    pos = Fh[te][yh[te] == 1]
                    tpr = tpr_at_fpr(clf, pos, Fc)
                out.append({"features": fkind, "probe": probe, "held_out": held,
                            "auroc": a, "tpr_at_1pct_fpr": tpr, "n_test": int(len(te))})
                logger.info("  %-16s %-7s held=%-30s AUROC=%.3f TPR@1%%=%.3f",
                            fkind, probe, held, a, tpr)
    return out


def zeroshot(data, bank, args):
    """No training at all: project onto a direction and score AUROC directly.

    This is the paper's Figure 3 and the only number of theirs that can be compared without
    matching their train/test protocol, because there is no protocol to match -- the
    direction comes from persona prompts and never sees an eval label.

    Sign is pinned per direction by construction (6_axes.py orients each PC against its own
    behaviour contrast), but a direction that lands upside-down on a dataset would read as
    1-AUROC, so both are reported and the flag is explicit rather than silently absolved.
    """
    ax = bank["axes"]
    dirs = {"contrast": unit(ax["deception"][args.layer].numpy()),
            "sycophancy": unit(ax["sycophancy"][args.layer].numpy()),
            "unified": unit(ax["unified"][args.layer].numpy()),
            "assistant": unit(ax["assistant"][args.layer].numpy())}
    comps = bank["pca"]["deception"]["components"][args.layer].numpy()
    for j in range(min(3, comps.shape[0])):
        dirs[f"PC{j + 1}"] = unit(comps[j])
    dirs["contrast_resid"] = unit(dirs["contrast"]
                                  - float(dirs["contrast"] @ dirs["assistant"])
                                  * dirs["assistant"])

    out = []
    logger.info("  %-16s" % "dataset" + "".join(f"{k:>16}" for k in dirs))
    for k, (X, y, _) in data.items():
        if len(np.unique(y)) < 2:
            continue
        row = {"dataset": k}
        cells = []
        for name, d in dirs.items():
            a = float(roc_auc_score(y, X @ d))
            row[name] = a
            cells.append(f"{a:>16.3f}" + ("*" if a < 0.5 else " ")[:0])
        out.append(row)
        logger.info("  %-16s" % k + "".join(cells))
    if out:
        logger.info("  %-16s" % "MEAN" + "".join(
            f"{np.mean([r[k] for r in out]):>16.3f}" for k in dirs))
    return out


def confounds(data, bank, args):
    """Does the projection just track response length? And is the null where it should be?"""
    out = []
    for fkind in args.features:
        if fkind == "raw":
            continue
        for k, (X, y, meta) in data.items():
            F = make_features(X, fkind, bank, args.layer, args.k, args.seed)
            ntok = np.array([m.get("span_len", 0) for m in meta], dtype=float)
            if ntok.std() == 0:
                continue
            r2 = []
            for j in range(F.shape[1]):
                c = np.corrcoef(F[:, j], ntok)[0, 1]
                r2.append(0.0 if np.isnan(c) else c * c)
            out.append({"features": fkind, "dataset": k,
                        "max_r2_vs_n_tok": float(max(r2)),
                        "mean_r2_vs_n_tok": float(np.mean(r2))})
    for r in sorted(out, key=lambda d: -d["max_r2_vs_n_tok"])[:8]:
        logger.info("  length confound: %-16s %-30s max R2=%.3f",
                    r["features"], r["dataset"], r["max_r2_vs_n_tok"])
    return out


def shuffle_null(data, bank, args):
    """Labels shuffled within dataset: anything far from 0.50 means a leak in the pipeline."""
    vals = []
    src = sources(data)
    for s in src[:4]:
        Xs, ys, ms = data[s]
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            yp = rng.permutation(ys)
            tr, te = group_split(ms, 0.75, seed)
            tr, te = balance(tr, yp, seed), balance(te, yp, seed + 1)
            if len(tr) < 8 or len(te) < 8:
                continue
            F = make_features(Xs, "raw" if "raw" in args.features else args.features[0],
                              bank, args.layer, args.k, seed)
            clf = make_probe("apollo", seed)
            clf.fit(F[tr], yp[tr])
            vals.append(auroc(clf, F[te], yp[te]))
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        logger.info("  label-shuffle null: %.3f +/- %.3f over %d draws (want ~0.50)",
                    float(np.mean(vals)), float(np.std(vals)), len(vals))
    return {"mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals)) if vals else None, "n": len(vals)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3.pt"),
                    help="axes .pt from 6_axes.py; paper16 score-3 is the controlled bank")
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--k", type=int, default=5, help="persona coordinates to keep")
    ap.add_argument("--features", nargs="+",
                    default=["raw", "pc:deception", "axes", "axes_resid", "random"])
    ap.add_argument("--probes", nargs="+", default=["paper", "lr_reg10", "mms"],
                    choices=["paper", "lr_reg10", "apollo", "mms"],
                    help="paper = the paper's C=1 no-scaler probe; lr_reg10 = the "
                         "lr/reg_coeff-10 config shared by Apollo and Liars' Bench "
                         "(they differ only in layer); mms = Apollo's mean-mass shift")
    ap.add_argument("--protocol", nargs="+", default=["p1"],
                    choices=["p1", "p2", "p3", "zeroshot", "confounds", "null", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=5, help="repeats for p2 and the null")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    keys = args.datasets
    if keys == ["all"]:
        keys = sorted(d.name for d in run_dir.iterdir()
                      if d.is_dir() and (d / "rows.jsonl").exists())

    logger.info("layer %d, pooling %s, k=%d, bank=%s",
                args.layer, args.pooling, args.k, Path(args.bank).name)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)

    logger.info("loading:")
    data = build(run_dir, keys, args.layer, args.pooling)
    if not data:
        raise SystemExit("no datasets with extracted activations yet")
    logger.info("training sources: %s", ", ".join(sources(data)) or "(none)")

    protos = args.protocol
    if "all" in protos:
        protos = ["p1", "p2", "p3", "zeroshot", "confounds", "null"]

    # Realised split sizes, per the plan: balancing and item_id grouping both shrink a
    # split, and by how much is not predictable from the row count -- convincing-game is
    # 167D/100H, so a balanced train split is bounded by the 100, not the 267.
    splits = {}
    for k, (_, y, meta) in data.items():
        tr, te = group_split(meta, 0.75, args.seed)
        btr, bte = balance(tr, y, args.seed), balance(te, y, args.seed + 1)
        ball = balance(np.arange(len(y)), y, args.seed + 1)
        splits[k] = {"rows": int(len(y)), "items": len({m["item_id"] for m in meta}),
                     "pos": int(y.sum()), "neg": int((1 - y).sum()),
                     "train_items_75pct": int(len(tr)), "test_items_25pct": int(len(te)),
                     "train_balanced": int(len(btr)), "test_balanced_ondiag": int(len(bte)),
                     "test_balanced_offdiag": int(len(ball)),
                     "label_semantics": meta[0].get("label_semantics")}
    logger.info("realised splits:")
    logger.info("  %-24s%7s%7s%9s%11s%12s%12s", "dataset", "rows", "items", "D/H",
                "train(bal)", "test-diag", "test-off")
    for k, v in splits.items():
        logger.info("  %-24s%7d%7d%9s%11d%12d%12d", k, v["rows"], v["items"],
                    f'{v["pos"]}/{v["neg"]}', v["train_balanced"],
                    v["test_balanced_ondiag"], v["test_balanced_offdiag"])

    results = {"layer": args.layer, "pooling": args.pooling, "k": args.k,
               "bank": Path(args.bank).name,
               "datasets": {k: int(len(v[1])) for k, v in data.items()},
               "splits": splits}
    if "p1" in protos:
        logger.info("P1 transfer matrix:")
        results["p1"] = p1_transfer(data, bank, args)
    if "p2" in protos:
        logger.info("P2 sample-efficiency curves:")
        results["p2"] = p2_curves(data, bank, args)
    if "p3" in protos:
        logger.info("P3 leave-one-dataset-out:")
        results["p3"] = p3_loso(data, bank, args)
    if "zeroshot" in protos:
        logger.info("zero-shot direction projections (no training; the paper's Figure 3):")
        results["zeroshot"] = zeroshot(data, bank, args)
    if "confounds" in protos:
        logger.info("confound checks:")
        results["confounds"] = confounds(data, bank, args)
    if "null" in protos:
        logger.info("label-shuffle null:")
        results["null"] = shuffle_null(data, bank, args)

    out = Path(args.out) if args.out else (
        REPO_ROOT / "run-coords" / f"probes_L{args.layer}_{args.pooling}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
