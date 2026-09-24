"""Independent reimplementation of the dataset-level paired bootstrap used by
src/assistant_axis_ndif/27_crossgen.py (lines 778-824) and by cap/scripts/cap_significance.py.

Written for the audit so a 'clears / does not clear' verdict can be recomputed rather than read.
Read-only: takes AUROC matrices as nested lists.
"""
import numpy as np


def arr(a):
    return np.array([[np.nan if v is None else float(v) for v in row] for row in a], dtype=float)


def make_weights(D, B, rng):
    """The 27_crossgen weight scheme: multinomial dataset counts -> outer-product cell weights,
    diagonal zeroed."""
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=B)
    W = (counts[:, :, None] * counts[:, None, :]).astype(np.float64)
    W[:, np.arange(D), np.arange(D)] = 0
    return W


def dboot(A, ids, W, D):
    A = A[np.ix_(ids, ids)]
    mask = ~np.isnan(A) & ~np.eye(D, dtype=bool)
    Aw = np.where(mask, A, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * Aw[None]).sum(axis=(1, 2)) / (W * mask[None]).sum(axis=(1, 2))


def two_sided_p(d):
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(min(p, 1.0), 1.0 / len(d)))


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def offdiag_mean(A, ids):
    A = A[np.ix_(ids, ids)]
    m = ~np.eye(len(ids), dtype=bool)
    return float(np.nanmean(A[m]))
