#!/usr/bin/env python
"""Cross-generalization matrix: one probe per dataset, trained on 80% of it, tested on all ten.

Row = trained on, column = tested on, over the ten labelled Pass B datasets.

  * diagonal     -- the probe's OWN dataset, scored on its held-out 20% test split
  * off-diagonal -- every OTHER dataset, scored on 100% of its rows (no split: the probe never
                    saw any of it)

Only the training dataset is split. Splits are by item_id, so paraphrases of one item never
straddle train and test, and stratified by label, so no test split can come out single-class.
Training rows are class-balanced by downsampling; test rows are not, because AUROC does not
depend on the class ratio and balancing would only throw test rows away. alpaca is honest-only:
it is never trained or scored, and sets the 1%-FPR threshold for the TPR matrices.

Writes ONLY under run-crossgen/ and refuses any other output root, so run-coords/ -- every
earlier result -- cannot be touched.

    python 27_crossgen.py --smoke                    # layer 44, last pooling, seed 42 + checks
    python 27_crossgen.py --stage compute            # every layer x pooling, resumable
    python 27_crossgen.py --stage summarize          # CSVs, summary.json, significance.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

# grouped by label construct, the order every matrix and figure uses
DATASETS = ["instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
            "convincing-game", "insider-trading", "ai_liar", "sandbagging_v2",
            "roleplaying", "mask", "sycophancy_mt"]
BLOCK9 = [d for d in DATASETS if d != "sycophancy_mt"]
# Every persona feature is compared against raw AND against a MATCHED control at the same k.
# Matched controls are the honest comparison: a low-dimensional feature sits closer to 0.5 than an
# 8192-d probe, so it "wins" on cells where raw transfers badly even when the directions are
# meaningless -- a random 2-d projection gains +0.056 where raw < 0.6 and loses 0.44 where raw is
# above 0.85 (corr(raw, delta) = -0.76). Only the gain over datapca/random at the same k says
# anything about persona structure.
# THE BASELINES AND CONTROLS. Everything else in a --features list is a persona feature and
# gets bootstrapped against its matched control.
BASELINE_FEATURES = {"raw", "random", "datapca", "length"}

# Paper16's names, kept only as the default for a bank that names nothing else.
PERSONA_FEATURES = ["axes", "pc:deception", "pc:unified", "pc:unified@2"]


def comparisons_for(features):
    """Persona features vs raw and BOTH matched controls at matched k, plus control vs raw.

    This used to be a module-level constant over the literal paper16 names, which meant a bank
    naming anything else got NO persona comparisons at all: on 2026-09-21 the cap bank's
    significance.json contained only the four control-vs-raw pairs, so the pipeline's own
    dataset-level bootstrap never tested `pc:capitulation` -- the number the bank exists to
    produce. Seventh hardcoding of paper16's names in this file family; each was reachable only
    after the previous one was fixed. Derived from the feature list now, so it cannot happen to
    the next bank.
    """
    persona = [f for f in features if f.partition("@")[0] not in BASELINE_FEATURES]
    out = [(f, c) for f in persona
           for c in (["raw", "datapca@2", "random@2"] if f.endswith("@2")
                     else ["raw", "datapca", "random"])]
    out += [("random", "raw"), ("datapca", "raw"), ("random@2", "raw"), ("datapca@2", "raw")]
    seen, uniq = set(), []
    for p in out:
        if p not in seen and p[0] != p[1]:
            seen.add(p); uniq.append(p)
    return uniq


COMPARISONS = comparisons_for(PERSONA_FEATURES)

# label construct per dataset, for the block summaries
CONSTRUCT = {"instructed-deception": "intent", "harm-pressure-choice": "intent",
             "harm-pressure-knowledge-report": "intent", "convincing-game": "intent",
             "insider-trading": "intent", "ai_liar": "intent", "sandbagging_v2": "intent",
             "roleplaying": "judge", "mask": "belief", "sycophancy_mt": "belief"}


def parse_feature(spec: str, default_k: int):
    """'random@2' -> ('random', 2); 'axes' -> ('axes', default_k)."""
    base, _, kk = spec.partition("@")
    return base, (int(kk) if kk else default_k)


def datapca_projections(X32, rows_train, targets, kmax, seed):
    """The paper's dataset-PCA control: PCA fitted on the SOURCE's own training rows only, then
    every dataset projected onto that source-specific basis."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=kmax, svd_solver="randomized", random_state=seed)
    pca.fit(X32[targets[0]][rows_train])
    comps = pca.components_.astype(np.float32)
    shift = (pca.mean_.astype(np.float32) @ comps.T)
    return {t: X32[t] @ comps.T - shift for t in targets}
DESIGN = ("train on 80% of the row dataset; diagonal = its own 20% test split; "
          "off-diagonal = 100% of the column dataset")


def module(name, fname):
    """Reuse the existing harness instead of restating it. Safe: no trace body here."""
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def guard(out_root: Path) -> Path:
    r = out_root.resolve()
    if "run-crossgen" not in r.parts or any(p in ("run-coords", "persona-coords") for p in r.parts):
        raise SystemExit(f"refusing to write to {r}: results must live under a run-crossgen/ "
                         "folder, never beside the earlier results")
    return r


def crossgen_root(out_root: Path) -> Path:
    """Where the baselines (raw / random / dataset PCA / length) and the reference splits live.

    A bank folder is <baseline root>/banks/<name>, and each activation store gets its own baseline
    root: run-crossgen for passB, run-crossgen/passA for passA, and so on. Resolving to the
    top-level run-crossgen unconditionally would merge Pass B's baselines into a Pass A summary.
    """
    p = out_root
    if p.parent.name == "banks":
        p = p.parent.parent
    return p


def clean(o):
    """NaN/inf -> None so the JSON is standard."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.floating):
        return clean(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(path: Path, obj) -> None:
    """Atomic write, with a retry because Windows will not replace an OPEN file.

    On 2026-09-20 a passC root run died at the last line with
        PermissionError: [WinError 5] Access is denied: 'significance.json.tmp' -> '...json'
    after computing every matrix. The rename loses to anything holding the target: the rclone
    sync that runs every 15 minutes, or the two antivirus services on this machine. POSIX
    rename is atomic and silent about readers; Windows is not, so the retry is platform
    behaviour rather than a race in our own code.

    Retry rather than fall back to a non-atomic write: a partially written results file that
    looks complete is far worse than a slow one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1), encoding="utf-8")
    delay = 0.5
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            logger.warning("  %s is locked (attempt %d/6); retrying in %.1fs",
                           path.name, attempt + 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 8.0)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def stratified_group_split(meta, y, frac: float, seed: int):
    """80/20 by item_id, stratified on each item's label make-up. -> (train, test, attempt)."""
    by_item = defaultdict(list)
    for i, m in enumerate(meta):
        by_item[str(m["item_id"])].append(i)
    strata = defaultdict(list)
    for it, idx in by_item.items():
        lab = y[idx]
        strata["pos" if lab.min() == 1 else "neg" if lab.max() == 0 else "mixed"].append(it)
    for attempt in range(50):
        rng = np.random.default_rng(seed + 7919 * attempt)
        test_items = set()
        for key in sorted(strata):
            items = sorted(strata[key])
            rng.shuffle(items)
            test_items.update(items[:int(round(len(items) * (1.0 - frac)))])
        te = np.array(sorted(i for it in test_items for i in by_item[it]), dtype=int)
        tr = np.array(sorted(i for it, idx in by_item.items() if it not in test_items
                             for i in idx), dtype=int)
        if len(te) and len(tr) and len(np.unique(y[te])) == 2 and len(np.unique(y[tr])) == 2:
            shared = {str(meta[i]["item_id"]) for i in tr} & {str(meta[i]["item_id"]) for i in te}
            assert not shared, f"item_ids on both sides: {sorted(shared)[:5]}"
            return tr, te, attempt
    raise SystemExit("could not draw a split with both classes on both sides")


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class Resampler:
    """Stratified bootstrap of one test set, shared by every method scored on it: paired.

    Resampling is stored as multinomial COUNTS per row, so a bootstrap AUROC is a weighted
    Mann-Whitney statistic: sort the negatives once, take cumulative negative weight, and look
    each positive up by searchsorted (ties count one half). O(B x n) per call instead of
    ranking a (B, n) matrix, which matters when the test set is all 5,432 rows of a dataset.
    """

    def __init__(self, y, B, rng):
        self.pos = np.where(y == 1)[0]
        self.neg = np.where(y == 0)[0]
        npos, nneg = len(self.pos), len(self.neg)
        self.cp = rng.multinomial(npos, np.full(npos, 1.0 / npos), size=B).astype(np.float64)
        self.cn = rng.multinomial(nneg, np.full(nneg, 1.0 / nneg), size=B).astype(np.float64)

    @staticmethod
    def weighted_auc(sp, sn, cp, cn):
        order = np.argsort(sn, kind="stable")
        sn_sorted = sn[order]
        C = np.zeros((cn.shape[0], cn.shape[1] + 1))
        np.cumsum(cn[:, order], axis=1, out=C[:, 1:])
        lo = np.searchsorted(sn_sorted, sp, side="left")
        hi = np.searchsorted(sn_sorted, sp, side="right")
        g = C[:, lo] + 0.5 * (C[:, hi] - C[:, lo])
        return (cp * g).sum(axis=1) / (cp.sum(axis=1) * cn.sum(axis=1))

    def boot(self, s):
        return self.weighted_auc(s[self.pos], s[self.neg], self.cp, self.cn)


def self_check_weighted_auc(rng) -> float:
    """Unit weights must reproduce sklearn's AUROC exactly, ties included."""
    worst = 0.0
    for _ in range(20):
        n = int(rng.integers(20, 400))
        y = rng.integers(0, 2, n)
        if len(np.unique(y)) < 2:
            continue
        s = np.round(rng.standard_normal(n), 1)          # rounding forces ties
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        a = Resampler.weighted_auc(s[pos], s[neg], np.ones((1, len(pos))), np.ones((1, len(neg))))[0]
        worst = max(worst, abs(a - roc_auc_score(y, s)))
    if worst > 1e-9:
        raise SystemExit(f"weighted AUROC disagrees with sklearn by {worst}")
    return worst


def tpr_at(scores_pos, ctrl_scores, fpr=0.01):
    if len(scores_pos) == 0 or len(ctrl_scores) == 0:
        return float("nan")
    thr = float(np.quantile(ctrl_scores, 1.0 - fpr))
    return float((scores_pos > thr).mean())


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #
def load_pooling(S, run_dir, keys, pooling):
    data, layers = {}, None
    for k in keys:
        X16, meta, L = S.load_all_layers(run_dir, k, pooling)
        if X16 is None:
            raise SystemExit(f"{k}: no rows in {run_dir}")
        layers = L if layers is None else layers
        if list(L) != list(layers):
            raise SystemExit(f"{k}: stored layers differ from the other datasets")
        y = np.array([m["label"] for m in meta], dtype=int)
        data[k] = (X16, y, meta)
        logger.info("  %-32s n=%-6d %dD/%dH", k, len(y), int(y.sum()), int((1 - y).sum()))
    return data, [int(x) for x in layers]


def make_splits(data, seeds, frac):
    return {seed: {k: stratified_group_split(data[k][2], data[k][1], frac, seed)
                   for k in DATASETS} for seed in seeds}


def split_table(data, splits, P):
    rows = {}
    for seed, per in splits.items():
        rows[str(seed)] = {}
        for k, (tr, te, attempt) in per.items():
            _, y, meta = data[k]
            btr = P.balance(tr, y, seed)
            rows[str(seed)][k] = {
                "rows": int(len(y)), "items": len({str(m["item_id"]) for m in meta}),
                "pos": int(y.sum()), "neg": int((1 - y).sum()),
                "train_rows": int(len(tr)), "test_rows": int(len(te)),
                "test_fraction": float(len(te) / len(y)),
                "train_pos": int(y[tr].sum()), "train_neg": int((1 - y[tr]).sum()),
                "test_pos": int(y[te].sum()), "test_neg": int((1 - y[te]).sum()),
                "train_balanced": int(len(btr)), "shared_item_ids": 0,
                "split_attempts": int(attempt) + 1,
                "test_row_ids": sorted(meta[i]["row_id"] for i in te)}
    return rows


def test_rows(data, splits_seed, s, t):
    """The rows a probe trained on s is scored on in column t."""
    return splits_seed[t][1] if t == s else np.arange(len(data[t][1]))


def compute_cell_block(P, data, X32, splits_seed, bank, L, seed, features, probes, args,
                       control, with_boot, res_split, res_full):
    """Everything for one (pooling, layer, seed). -> result dict."""
    n = len(DATASETS)
    results = []
    boot_sums = {}           # (feature, probe) -> {"off10","off9","diag10","diag9": (B,)}
    kmax_pca = max([parse_feature(f, args.k)[1] for f in features
                    if parse_feature(f, args.k)[0] == "datapca"] or [0])
    pca_cache = {}
    for fkind in features:
        base, kf = parse_feature(fkind, args.k)
        shared = None
        if base != "datapca":
            shared = {k: P.make_features(X32[k], base, bank, L, kf, seed, meta=data[k][2])
                      for k in DATASETS + [control]}
        for probe in probes:
            A = np.full((n, n), np.nan)
            LO = np.full((n, n), np.nan)
            HI = np.full((n, n), np.nan)
            TPR = np.full((n, n), np.nan)
            train_n = []
            sums = {k: np.zeros(args.boot_rows) for k in ("off10", "off9", "diag10", "diag9")}
            cnt = defaultdict(int)
            for i, s in enumerate(DATASETS):
                _, ys, _ = data[s]
                if base == "datapca":
                    if s not in pca_cache:
                        order = [s] + [t for t in DATASETS + [control] if t != s]
                        pca_cache[s] = datapca_projections(X32, splits_seed[s][0], order,
                                                           kmax_pca, seed)
                    feats = {t: v[:, :kf] for t, v in pca_cache[s].items()}
                else:
                    feats = shared
                tr = P.balance(splits_seed[s][0], ys, seed)
                train_n.append(int(len(tr)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf = P.make_probe(probe, seed).fit(feats[s][tr], ys[tr])
                ctrl = P._scores(clf, feats[control])
                for j, t in enumerate(DATASETS):
                    rows = test_rows(data, splits_seed, s, t)
                    yt = data[t][1][rows]
                    sc = P._scores(clf, feats[t][rows])
                    A[i, j] = safe_auc(yt, sc)
                    TPR[i, j] = tpr_at(sc[yt == 1], ctrl)
                    if with_boot:
                        b = (res_split[t] if t == s else res_full[t]).boot(sc)
                        LO[i, j], HI[i, j] = np.percentile(b, [2.5, 97.5])
                        key = "diag" if t == s else "off"
                        sums[key + "10"] += b
                        cnt[key + "10"] += 1
                        if s in BLOCK9 and t in BLOCK9:
                            sums[key + "9"] += b
                            cnt[key + "9"] += 1
            rec = {"feature": fkind, "probe": probe, "auroc": A.tolist(),
                   "tpr_at_1pct_fpr": TPR.tolist(), "train_n_balanced": train_n}
            if with_boot:
                rec["ci95_lo"], rec["ci95_hi"] = LO.tolist(), HI.tolist()
                boot_sums[(fkind, probe)] = {k: sums[k] / max(cnt[k], 1) for k in sums}
            results.append(rec)
            off = ~np.eye(n, dtype=bool)
            # BLOCK9 is the first nine datasets -- the "without sycophancy_mt" comparison that
            # F-13 turns on. It is only meaningful when there ARE nine; with a shorter
            # --datasets list (the 5-dataset scale runs reproduce the paper's Table 1 deception
            # set) the slice is smaller than the mask and this raised IndexError. Clamp, and
            # report NaN rather than a misleading number when the block is not a proper subset.
            b = min(9, n)
            blk = (np.nanmean(A[:b, :b][~np.eye(b, dtype=bool)])
                   if b > 1 and b < n else float("nan"))
            logger.info("    %-13s %-8s diag %.3f  off-diag %.3f  (first-%d block %.3f)",
                        fkind, probe, np.nanmean(np.diag(A)), np.nanmean(A[off]), b, blk)
        del shared

    # zero-shot rows: no training, so nothing is held out -- every dataset scored on 100%
    #
    # The zero-shot direction is the bank's PRIMARY BEHAVIOUR CONTRAST, not literally
    # "deception". Every bank until 2026-09-21 was paper16 or a superset of it, so hardcoding
    # the name always worked; the cap bank's contrasts are capitulation / cap_vs_default /
    # cap_only and this raised KeyError. Unlike the trained features this block is not
    # controlled by --features, so it cannot be worked around from the call site.
    # Prefer "deception" when present so every existing result is bit-identical.
    _cands = [n for n in ("deception",) if n in bank["axes"] and n in bank["pca"]]
    if not _cands:
        _cands = [n for n in bank["axes"] if n != "assistant" and n in bank["pca"]]
    if not _cands:
        raise SystemExit("bank has no behaviour axis with a matching PCA basis for zero-shot")
    _zs_axis = _cands[0]
    zs = []
    # The primary axis keeps the bare names "contrast"/"PC1" so every existing result stays
    # bit-identical and the readers keep working. EVERY OTHER behaviour axis is now scored too,
    # under "contrast:<name>" / "PC1:<name>".
    #
    # This is the only place a CONTRAST direction reaches a number. The trained `pc:<name>`
    # features project onto the named PCA *basis* and never touch the contrast, so two variants
    # that share a role set are identical by construction. The cap bank's A/B pair is exactly
    # that: designed to isolate the contrast, and unable to at any trained k. Scoring one
    # direction per bank left that question unanswerable; scoring all of them answers it.
    dirs = {"contrast": P.unit(bank["axes"][_zs_axis][L].numpy().astype(np.float32)),
            "PC1": P.unit(bank["pca"][_zs_axis]["components"][L][0].numpy().astype(np.float32))}
    for _n in bank["axes"]:
        if _n == "assistant" or _n == _zs_axis or _n not in bank["pca"]:
            continue
        dirs[f"contrast:{_n}"] = P.unit(bank["axes"][_n][L].numpy().astype(np.float32))
        dirs[f"PC1:{_n}"] = P.unit(
            bank["pca"][_n]["components"][L][0].numpy().astype(np.float32))
    for name, d in dirs.items():
        ctrl = X32[control] @ d
        row = {"direction": name, "auroc": [], "tpr_at_1pct_fpr": [], "ci95_lo": [], "ci95_hi": []}
        for t in DATASETS:
            yt = data[t][1]
            sc = X32[t] @ d
            row["auroc"].append(safe_auc(yt, sc))
            row["tpr_at_1pct_fpr"].append(tpr_at(sc[yt == 1], ctrl))
            if with_boot:
                lo, hi = np.percentile(res_full[t].boot(sc), [2.5, 97.5])
                row["ci95_lo"].append(float(lo))
                row["ci95_hi"].append(float(hi))
        zs.append(row)

    out = {"results": results, "zeroshot": zs}
    if with_boot:
        rb = {}
        for probe in probes:
            for fa, fb in comparisons_for(features):
                if (fa, probe) not in boot_sums or (fb, probe) not in boot_sums:
                    continue
                for block in ("10", "9"):
                    d = boot_sums[(fa, probe)]["off" + block] - boot_sums[(fb, probe)]["off" + block]
                    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                    rb[f"{probe}|{fa} - {fb}|{block}"] = {
                        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "p": float(max(min(p, 1.0), 1.0 / len(d)))}
        out["row_bootstrap_offdiag_diff"] = rb
        out["row_bootstrap_offdiag_ci95"] = {
            f"{f}|{p}|{block}": [float(np.percentile(v["off" + block], 2.5)),
                                 float(np.percentile(v["off" + block], 97.5))]
            for (f, p), v in boot_sums.items() for block in ("10", "9")}
    return out


def smoke_checks(P, data, X32, splits_seed, seed, args, result):
    """Controls and the consistency check against the earlier sweep."""
    checks = {"weighted_auc_vs_sklearn_worst": self_check_weighted_auc(np.random.default_rng(0))}
    logger.info("  weighted-AUROC self-check vs sklearn: worst |diff| %.1e",
                checks["weighted_auc_vs_sklearn_worst"])
    nulls = []
    for s in DATASETS[:2]:
        _, ys, _ = data[s]
        yp = np.random.default_rng(1000 + seed).permutation(ys)
        tr = P.balance(splits_seed[s][0], yp, seed)
        clf = P.make_probe("mms", seed).fit(X32[s][tr], yp[tr])
        te = splits_seed[s][1]
        nulls.append({"source": s, "auroc": safe_auc(yp[te], P._scores(clf, X32[s][te]))})
    checks["label_shuffle_null"] = nulls
    logger.info("  label-shuffle null: %s", ", ".join(f"{x['source']} {x['auroc']:.3f}"
                                                      for x in nulls))
    # off-diagonal cells use the same test set as the earlier sweep (100% of the target), so on
    # the same 9 datasets the means must land close to it; only the training split moved 75 -> 80
    sweep_file = REPO_ROOT / "run-coords" / ("layer_sweep_passB_last.json" if args.smoke_pooling
                                             == "last" else "layer_sweep_passB_mms.json")
    if sweep_file.exists():
        sw = json.loads(sweep_file.read_text(encoding="utf-8"))
        ref = next((r for r in sw["sweep"] if r["layer"] == args.smoke_layer), None)
        cons = {}
        for rec in result["results"]:
            if rec["probe"] != "mms" or rec["feature"] not in ("raw", "axes", "pc:deception"):
                continue
            A_full = np.array(rec["auroc"], dtype=float)
            b = min(9, A_full.shape[0])
            A9 = A_full[:b, :b]
            new = (float(np.nanmean(A9[~np.eye(b, dtype=bool)]))
                   if b > 1 else float("nan"))
            old = ref[rec["feature"]] if ref else None
            cons[rec["feature"]] = {"new_9block": new, "earlier_sweep": old,
                                    "abs_diff": None if old is None else abs(new - old)}
            logger.info("  consistency %-13s new %.3f  earlier %.3f  |diff| %.3f",
                        rec["feature"], new, old if old is not None else float("nan"),
                        abs(new - old) if old is not None else float("nan"))
        checks["consistency_with_earlier_sweep"] = cons
    for rec in result["results"]:
        if rec["feature"] == "random":
            A = np.array(rec["auroc"], dtype=float)
            logger.info("  random control %-8s diag %.3f off-diag %.3f", rec["probe"],
                        np.nanmean(np.diag(A)), np.nanmean(A[~np.eye(len(A), dtype=bool)]))
    return checks


def stage_compute(args, out_root):
    P = module("probes", "9_persona_probes.py")
    S = module("sweep", "11_layer_sweep.py")
    control = P.CONTROL
    run_dir = Path(args.run_dir)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    poolings = [args.smoke_pooling] if args.smoke else args.poolings
    main_seed = args.main_seed
    all_seeds = [main_seed] + [s for s in args.extra_seeds if s != main_seed]
    t_start = time.time()
    self_check_weighted_auc(np.random.default_rng(0))

    splits_path = out_root / "splits.json"
    root_splits = Path(args.baseline_root or crossgen_root(out_root)) / "splits.json"
    for pooling in poolings:
        logger.info("loading pooling=%s", pooling)
        data, layers = load_pooling(S, run_dir, DATASETS + [control], pooling)
        splits = make_splits(data, all_seeds, args.frac)
        table = split_table(data, splits, P)
        # a bank run must use exactly the baseline run's splits, or its matrices cannot be
        # compared cell by cell with raw / random / dataset PCA from the root folder
        if root_splits.exists() and root_splits.resolve() != splits_path.resolve():
            root_doc = json.loads(root_splits.read_text(encoding="utf-8"))
            # only meaningful when both runs read the same activation store: passA and passC hold a
            # 200-row subsample, so their splits are different rows by construction
            if root_doc.get("run_dir") in (None, str(run_dir)):
                ref = root_doc["splits"]
                for seed, per in table.items():
                    for k, v in per.items():
                        if seed in ref and k in ref[seed] and ref[seed][k]["test_row_ids"] != v["test_row_ids"]:
                            raise SystemExit(f"split for {k} seed {seed} differs from {root_splits}")
                logger.info("splits identical to the baseline run in %s", root_splits.parent)
            else:
                logger.info("baseline splits are from %s, this run is %s -- not comparable, "
                            "skipping the cross-check", root_doc.get("run_dir"), run_dir)
        if splits_path.exists():
            prev = json.loads(splits_path.read_text(encoding="utf-8"))
            for seed, per in table.items():
                for k, v in per.items():
                    if seed in prev["splits"] and k in prev["splits"][seed]:
                        if prev["splits"][seed][k]["test_row_ids"] != v["test_row_ids"]:
                            raise SystemExit(f"split for {k} seed {seed} differs between "
                                             "runs/poolings -- refusing to mix them")
            prev["splits"].update({s: {**prev["splits"].get(s, {}), **per}
                                   for s, per in table.items()})
            prev["design"] = DESIGN
            prev["run_dir"] = str(run_dir)
            table_all = prev
        else:
            table_all = {"frac_train": args.frac, "design": DESIGN, "datasets": DATASETS,
                         "control": control, "run_dir": str(run_dir), "splits": table}
        write_json(splits_path, table_all)
        logger.info("realised splits of the TRAINING dataset (seed %d); other datasets are "
                    "always tested on 100%% of their rows:", main_seed)
        logger.info("  %-32s%7s%7s%7s%8s%10s%10s%9s", "dataset", "rows", "train", "test",
                    "test%", "test D/H", "train bal", "attempts")
        for k, v in table[str(main_seed)].items():
            logger.info("  %-32s%7d%7d%7d%7.1f%%%10s%10d%9d", k, v["rows"], v["train_rows"],
                        v["test_rows"], 100 * v["test_fraction"],
                        f'{v["test_pos"]}/{v["test_neg"]}', v["train_balanced"],
                        v["split_attempts"])

        want_layers = [args.smoke_layer] if args.smoke else (args.layers or layers)
        for L in want_layers:
            if L not in layers:
                raise SystemExit(f"layer {L} not stored; have {layers}")
            li = layers.index(L)
            X32 = None
            for seed in all_seeds:
                is_main = seed == main_seed
                if not is_main and args.extra_layers and L not in args.extra_layers:
                    continue
                path = out_root / "matrices" / f"{pooling}_L{L}_s{seed}.json"
                if path.exists() and not args.smoke and not args.recompute:
                    logger.info("  skip %s (exists)", path.name)
                    continue
                if X32 is None:
                    X32 = {k: data[k][0][:, li, :].to(torch.float32).numpy()
                           for k in DATASETS + [control]}
                t0 = time.time()
                feats = args.features if is_main else args.extra_features
                probes = args.probes if is_main else args.extra_probes
                logger.info("  pooling %s layer %d seed %d: %d features x %d probes%s",
                            pooling, L, seed, len(feats), len(probes),
                            " + row bootstrap" if is_main else "")
                res_split = res_full = None
                if is_main:
                    rng = np.random.default_rng(10_000 + seed * 101 + L)
                    res_split = {t: Resampler(data[t][1][splits[seed][t][1]], args.boot_rows, rng)
                                 for t in DATASETS}
                    res_full = {t: Resampler(data[t][1], args.boot_rows, rng) for t in DATASETS}
                res = compute_cell_block(P, data, X32, splits[seed], bank, L, seed, feats, probes,
                                         args, control, is_main, res_split, res_full)
                doc = {"pooling": pooling, "layer": L, "seed": seed, "frac_train": args.frac,
                       "design": DESIGN, "datasets": DATASETS, "block9": BLOCK9,
                       "control": control, "bank": Path(args.bank).name,
                       "bank_name": args.bank_name, "k": args.k,
                       "rows": "trained on (80% split)",
                       "columns": "tested on: own 20% split on the diagonal, 100% elsewhere",
                       **res, "seconds": time.time() - t0}
                if args.smoke:
                    doc["smoke_checks"] = smoke_checks(P, data, X32, splits[seed], seed, args, res)
                    path = out_root / "smoke" / f"{pooling}_L{L}_s{seed}.json"
                write_json(path, doc)
                logger.info("  wrote %s   %.0fs   (total %.0f min)", path.relative_to(out_root),
                            time.time() - t0, (time.time() - t_start) / 60)
            del X32
        del data


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def stage_summarize(args, out_root):
    SIG = module("sig", "23_significance.py")
    files = sorted((out_root / "matrices").glob("*.json"))
    if not files:
        raise SystemExit("no matrices yet")
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    bad = [f.name for f, d in zip(files, docs) if d.get("design") != DESIGN]
    if bad:
        raise SystemExit(f"matrices from a different test design present: {bad[:5]}")
    # a bank folder holds only persona features; take raw / random / dataset PCA / length from
    # the baseline run so every comparison is cell-by-cell on identical splits
    root = Path(args.baseline_root or crossgen_root(out_root))
    if root.resolve() != out_root.resolve():
        for f, d in zip(files, docs):
            base = root / "matrices" / f.name
            if not base.exists():
                continue
            bd = json.loads(base.read_text(encoding="utf-8"))
            have = {(r["feature"], r["probe"]) for r in d["results"]}
            d["results"] += [r for r in bd["results"]
                             if (r["feature"], r["probe"]) not in have
                             and r["feature"] in ("raw", "random", "random@2", "datapca",
                                                  "datapca@2", "length")]
    n = len(DATASETS)
    off10 = ~np.eye(n, dtype=bool)
    idx9 = [DATASETS.index(d) for d in BLOCK9]
    off9 = ~np.eye(len(idx9), dtype=bool)

    def arr(x):
        return np.array([[np.nan if v is None else v for v in row] for row in x], dtype=float)

    def block_means(A, D=None):
        """Mean AUROC (and mean delta vs raw) per construct-pair block, plus the sycophancy
        row/column on their own. One mean over 90 heterogeneous cells hides that transfer inside
        the instructed family behaves nothing like transfer across a construct."""
        out = {}
        pairs = [(i, j) for i in range(len(DATASETS)) for j in range(len(DATASETS)) if i != j]
        groups = defaultdict(list)
        for i, j in pairs:
            ci, cj = CONSTRUCT.get(DATASETS[i], "?"), CONSTRUCT.get(DATASETS[j], "?")
            groups[f"{ci}->{cj}"].append((i, j))
            if DATASETS[i] == "sycophancy_mt":
                groups["sycophancy as source"].append((i, j))
            if DATASETS[j] == "sycophancy_mt":
                groups["sycophancy as target"].append((i, j))
        for name, cells_ in sorted(groups.items()):
            v = [A[i, j] for i, j in cells_]
            rec = {"n_cells": len(cells_), "mean_auroc": float(np.nanmean(v))}
            if D is not None:
                rec["mean_delta_vs_raw"] = float(np.nanmean([D[i, j] for i, j in cells_]))
            out[name] = rec
        return out

    def sign_flips(A, raw, lo=None, hi=None):
        """Below-chance cells are not 'no signal' -- they are the same feature with the label
        mapping reversed (convincing-game -> harm-pressure-knowledge-report reads 0.05). Counted
        separately from weak cells, and split into ones a method rescues or breaks."""
        m = ~np.eye(len(DATASETS), dtype=bool)
        rec = {"below_chance": int(np.nansum(A[m] < 0.5)),
               "rescued_vs_raw": int(np.nansum((raw[m] < 0.5) & (A[m] > 0.5))),
               "broken_vs_raw": int(np.nansum((raw[m] > 0.5) & (A[m] < 0.5)))}
        if hi is not None:
            rec["significantly_inverted"] = int(np.nansum(hi[m] < 0.5))
            rec["significantly_above_chance"] = int(np.nansum(lo[m] > 0.5))
        worst = [(DATASETS[i], DATASETS[j], float(A[i, j]))
                 for i in range(len(DATASETS)) for j in range(len(DATASETS))
                 if i != j and A[i, j] == A[i, j] and A[i, j] < 0.35]
        rec["cells_below_0.35"] = sorted(worst, key=lambda t: t[2])[:8]
        return rec

    records, cells = [], defaultdict(list)
    csv_dir = out_root / "csv"
    for d in docs:
        raw_of = {r["probe"]: arr(r["auroc"]) for r in d["results"] if r["feature"] == "raw"}
        for rec in d["results"]:
            A, T = arr(rec["auroc"]), arr(rec["tpr_at_1pct_fpr"])
            A9 = A[np.ix_(idx9, idx9)]
            r = {"pooling": d["pooling"], "layer": d["layer"], "seed": d["seed"],
                 "feature": rec["feature"], "probe": rec["probe"],
                 "diag_mean": float(np.nanmean(np.diag(A))),
                 "offdiag_mean_10": float(np.nanmean(A[off10])),
                 "offdiag_mean_9": float(np.nanmean(A9[off9])),
                 "tpr_diag_mean": float(np.nanmean(np.diag(T))),
                 "tpr_offdiag_mean_10": float(np.nanmean(T[off10])),
                 "per_source_offdiag": {s: float(np.nanmean(np.delete(A[i], i)))
                                        for i, s in enumerate(DATASETS)},
                 "per_target_offdiag": {t: float(np.nanmean(np.delete(A[:, j], j)))
                                        for j, t in enumerate(DATASETS)},
                 "diagonal": {s: float(A[i, i]) for i, s in enumerate(DATASETS)}}
            r["generalization_gap_10"] = r["diag_mean"] - r["offdiag_mean_10"]
            base = raw_of.get(rec["probe"])
            if base is not None:
                r["blocks"] = block_means(A, A - base)
                r["sign_flips"] = sign_flips(
                    A, base,
                    arr(rec["ci95_lo"]) if "ci95_lo" in rec else None,
                    arr(rec["ci95_hi"]) if "ci95_hi" in rec else None)
            records.append(r)
            cells[(d["pooling"], d["layer"], rec["feature"], rec["probe"])].append(A)
            if d["seed"] == args.main_seed:
                for tag, M in (("", A), ("tpr_", T)):
                    fname = (f"{tag}{d['pooling']}_L{d['layer']}_"
                             f"{rec['feature'].replace(':', '-').replace('@', '_k')}_"
                             f"{rec['probe']}.csv")
                    p = csv_dir / fname
                    p.parent.mkdir(parents=True, exist_ok=True)
                    lines = ["trained_on (80%) \\ tested_on (own 20% | others 100%),"
                             + ",".join(DATASETS)]
                    for i, s in enumerate(DATASETS):
                        lines.append(s + "," + ",".join("" if np.isnan(v) else f"{v:.4f}"
                                                        for v in M[i]))
                    if tag == "":
                        for z in d["zeroshot"]:
                            lines.append(f"zero-shot {z['direction']} (100%)," + ",".join(
                                "" if v is None else f"{v:.4f}" for v in z["auroc"]))
                    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    seed_stats = []
    for (pool, L, f, p), mats in sorted(cells.items()):
        if len(mats) < 2:
            continue
        M = np.stack(mats)
        rs = [r for r in records
              if (r["pooling"], r["layer"], r["feature"], r["probe"]) == (pool, L, f, p)]

        def ms(key):
            v = [r[key] for r in rs]
            return float(np.mean(v)), float(np.std(v, ddof=1))

        seed_stats.append({"pooling": pool, "layer": L, "feature": f, "probe": p,
                           "n_seeds": len(mats),
                           "offdiag_mean_10": ms("offdiag_mean_10")[0],
                           "offdiag_sd_10": ms("offdiag_mean_10")[1],
                           "offdiag_mean_9": ms("offdiag_mean_9")[0],
                           "offdiag_sd_9": ms("offdiag_mean_9")[1],
                           "diag_mean": ms("diag_mean")[0], "diag_sd": ms("diag_mean")[1],
                           "cell_mean": np.nanmean(M, axis=0).tolist(),
                           "cell_sd": np.nanstd(M, axis=0, ddof=1).tolist()})

    zs = [{"pooling": d["pooling"], "layer": d["layer"], "direction": z["direction"],
           "auroc": dict(zip(DATASETS, z["auroc"])),
           "mean": float(np.nanmean([np.nan if v is None else v for v in z["auroc"]]))}
          for d in docs if d["seed"] == args.main_seed for z in d["zeroshot"]]
    write_json(out_root / "summary.json", {"design": DESIGN, "datasets": DATASETS,
                                           "block9": BLOCK9, "main_seed": args.main_seed,
                                           "records": records, "seed_stats": seed_stats,
                                           "zeroshot": zs})

    # dataset bootstrap on the mean off-diagonal, paired across methods (as 23_significance)
    rng = np.random.default_rng(args.main_seed)
    tests = []
    for block, ids in (("10", list(range(n))), ("9", idx9)):
        D = len(ids)
        counts = rng.multinomial(D, np.full(D, 1.0 / D), size=args.boot_datasets)
        W = (counts[:, :, None] * counts[:, None, :]).astype(np.float64)
        W[:, np.arange(D), np.arange(D)] = 0

        def dboot(A, ids=ids, W=W, D=D):
            A = A[np.ix_(ids, ids)]
            mask = ~np.isnan(A) & ~np.eye(D, dtype=bool)
            Aw = np.where(mask, A, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (W * Aw[None]).sum(axis=(1, 2)) / (W * mask[None]).sum(axis=(1, 2))

        for d in docs:
            if d["seed"] != args.main_seed:
                continue
            by = {(r["feature"], r["probe"]): arr(r["auroc"]) for r in d["results"]}
            rb = d.get("row_bootstrap_offdiag_diff", {})
            # from the matrix, not a constant -- a bank carries whatever features it was run with
            cmps = comparisons_for(sorted({f for f, _ in by}))
            for probe in sorted({p for _, p in by}):
                for fa, fb in cmps:
                    if (fa, probe) not in by or (fb, probe) not in by:
                        continue
                    A = by[(fa, probe)][np.ix_(ids, ids)]
                    B = by[(fb, probe)][np.ix_(ids, ids)]
                    offm = ~np.eye(D, dtype=bool)
                    dd = dboot(by[(fa, probe)]) - dboot(by[(fb, probe)])
                    dd = dd[~np.isnan(dd)]
                    row = rb.get(f"{probe}|{fa} - {fb}|{block}", {})
                    tests.append({"block": block, "pooling": d["pooling"], "layer": d["layer"],
                                  "probe": probe, "comparison": f"{fa} - {fb}",
                                  "diff": float(np.nanmean(A[offm]) - np.nanmean(B[offm])),
                                  "pairs_won": int(np.nansum(A[offm] > B[offm])),
                                  "pairs": int(offm.sum()),
                                  "row_ci95": row.get("ci95"), "row_p": row.get("p"),
                                  "dataset_ci95": [float(np.percentile(dd, 2.5)),
                                                   float(np.percentile(dd, 97.5))],
                                  "dataset_p": SIG.two_sided_p(dd)})
    for block in ("10", "9"):
        fam = [t for t in tests if t["block"] == block]
        if fam:
            for t, qq in zip(fam, SIG.bh([t["dataset_p"] for t in fam])):
                t["dataset_q_bh"] = float(qq)
                t["significant_datasets"] = bool(qq < 0.05)
    write_json(out_root / "significance.json", {"design": DESIGN, "boot_datasets": args.boot_datasets,
                                                "note": "BH within each block over every "
                                                        "pooling x layer x probe x comparison",
                                                "tests": tests})

    logger.info("mean off-diagonal AUROC, seed %d (10 datasets | 9 without sycophancy_mt)",
                args.main_seed)
    for pool in args.poolings:
        logger.info("  pooling=%s", pool)
        for L in sorted({r["layer"] for r in records if r["pooling"] == pool}):
            parts = []
            for f in ("raw", "axes", "pc:deception", "random"):
                r = next((r for r in records if (r["pooling"], r["layer"], r["feature"],
                                                 r["probe"], r["seed"]) ==
                          (pool, L, f, "mms", args.main_seed)), None)
                if r:
                    parts.append(f"{f} {r['offdiag_mean_10']:.3f}|{r['offdiag_mean_9']:.3f}")
            logger.info("    L%-2d mms  %s", L, "   ".join(parts))
    logger.info("wrote summary.json, significance.json, %d CSVs",
                len(list(csv_dir.glob("*.csv"))))


def main() -> None:
    global DATASETS, BLOCK9
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="passA and passC hold 9 labelled datasets (no sycophancy_mt), so those "
                         "stores run 9 x 9; pass the list explicitly there")
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--out_root", default=str(REPO_ROOT / "run-crossgen"),
                    help="run-crossgen/ for the baseline + paper16 run; "
                         "run-crossgen/banks/<name> for another persona bank")
    ap.add_argument("--bank_name", default="paper16")
    ap.add_argument("--baseline_root", default="",
                    help="folder holding the baseline matrices and reference splits for this "
                         "activation store; default: the parent of banks/<name>, else out_root")
    ap.add_argument("--stage", default="all", choices=["compute", "summarize", "all"])
    ap.add_argument("--poolings", nargs="+", default=["mean", "last"])
    ap.add_argument("--layers", nargs="*", type=int, default=[], help="default: every stored layer")
    ap.add_argument("--features", nargs="+",
                    default=["raw", "pc:deception", "axes", "axes_resid", "random", "length",
                             "pc:unified", "datapca",
                             "pc:unified@2", "random@2", "datapca@2"],
                    help="'@k' overrides --k for that feature; datapca = PCA fitted on the "
                         "source's own training rows (the paper's dataset-PCA control)")
    ap.add_argument("--probes", nargs="+", default=["paper", "lr_reg10", "mms"])
    ap.add_argument("--main_seed", type=int, default=42)
    ap.add_argument("--extra_seeds", nargs="*", type=int, default=[],
                    help="additional 80/20 splits for error bands (no row bootstrap)")
    ap.add_argument("--extra_features", nargs="+", default=["raw", "axes", "pc:deception"])
    ap.add_argument("--extra_probes", nargs="+", default=["mms", "lr_reg10"])
    ap.add_argument("--extra_layers", nargs="*", type=int, default=[],
                    help="restrict extra seeds to these layers; default every layer")
    ap.add_argument("--frac", type=float, default=0.8, help="train fraction of the training dataset")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot_rows", type=int, default=1000)
    ap.add_argument("--boot_datasets", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild matrices that already exist instead of skipping them. The "
                         "resume-by-skipping default is right for a chain that may be "
                         "interrupted, but it silently pins a bank to whatever the code did on "
                         "its first run -- adding a zero-shot direction or a feature changes "
                         "nothing until the matrices are rebuilt. Same seed, so the trained "
                         "numbers should reproduce exactly; a change there is a bug.")
    ap.add_argument("--smoke_layer", type=int, default=44)
    ap.add_argument("--smoke_pooling", default="last")
    args = ap.parse_args()

    if list(args.datasets) != DATASETS:
        DATASETS = list(args.datasets)
        BLOCK9 = [d for d in DATASETS if d != "sycophancy_mt"]
        logger.info("datasets overridden: %d -> %s", len(DATASETS), ", ".join(DATASETS))
    out_root = guard(Path(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("output root: %s", out_root)
    logger.info("design: %s", DESIGN)
    if args.stage in ("compute", "all") or args.smoke:
        stage_compute(args, out_root)
    if args.stage in ("summarize", "all") and not args.smoke:
        stage_summarize(args, out_root)


if __name__ == "__main__":
    main()
