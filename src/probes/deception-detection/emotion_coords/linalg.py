"""Subspace helpers for the emotion-coordinate study: bases, keep-k, erase-k, random subspaces, LEACE.

A basis is a (k, hidden) array whose rows are orthonormal. "Keep-k" maps activations to their coordinates
in the first k rows (`acts @ U_k.T`); "erase-k" removes those directions and keeps the rest of the
activation (`acts - acts @ U_k.T @ U_k`). The two reconstruct the input: erase_k(x) + keep_k(x) @ U_k = x.
"""

from dataclasses import dataclass

import numpy as np


def orthonormal(rows: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """An orthonormal basis (as rows) of the span of `rows`, in their order (QR, rank-revealing by `tol`)."""
    q, r = np.linalg.qr(np.asarray(rows, dtype=np.float64).T)
    keep = np.abs(np.diag(r)) > tol * max(1.0, float(np.abs(np.diag(r)).max(initial=0.0)))
    return q[:, keep].T


def keep_k(acts: np.ndarray, basis: np.ndarray, k: int | None = None) -> np.ndarray:
    """Coordinates of `acts` in the first k basis rows: (n, hidden) -> (n, k)."""
    return acts @ basis[:k].T


def erase_k(acts: np.ndarray, basis: np.ndarray, k: int | None = None) -> np.ndarray:
    """`acts` with the first k basis directions projected out (orthogonal projection)."""
    u = basis[:k]
    return acts - (acts @ u.T) @ u


def projector(basis: np.ndarray, k: int | None = None) -> np.ndarray:
    """The (hidden, hidden) orthogonal projector onto the first k basis rows. For tests and small bases."""
    u = basis[:k]
    return u.T @ u


def random_subspace(hidden: int, k: int, seed: int) -> np.ndarray:
    """k random orthonormal directions (QR of a Gaussian matrix), seeded."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
    return q.T


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pca_basis(vectors: np.ndarray, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Principal directions of a set of vectors (rows) and their explained-variance ratios, by SVD."""
    x = np.asarray(vectors, dtype=np.float64)
    if center:
        x = x - x.mean(0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    keep = var > 1e-12 * var.max()
    return vt[keep], (var / var.sum())[keep]


def variance_basis(vectors: np.ndarray, share: float) -> np.ndarray:
    """The top principal directions of `vectors` that together explain at least `share` of their variance."""
    basis, ratio = pca_basis(vectors)
    k = int(np.searchsorted(np.cumsum(ratio), share) + 1)
    return basis[:k]


# --- LEACE (Belrose et al. 2023), with covariance shrinkage --------------------------------------


@dataclass
class Leace:
    """The least-squares concept eraser for a categorical concept, fitted with a shrunk covariance.

    r(x) = x - A (B (x - mean)), with A = W^+ Q and B = Q^T W, where W = Sigma_xx^{-1/2} (whitening) and Q
    is an orthonormal basis of the column space of W Sigma_xz (one column per class indicator). Whatever
    invertible W is used, cov(r(X), Z) = 0 on the fitting distribution, so no linear probe of r(X) can
    recover the classes there. Sigma_xx is shrunk towards a scaled identity (Ledoit-Wolf), since there are
    fewer samples than dimensions.
    """

    mean: np.ndarray  # (hidden,)
    a: np.ndarray  # (hidden, rank)
    b: np.ndarray  # (rank, hidden)
    shrinkage: float

    @property
    def rank(self) -> int:
        return self.a.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x - ((x - self.mean) @ self.b.T) @ self.a.T


def ledoit_wolf(xc: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the covariance of centred rows `xc` towards mu * I. Returns (cov, delta)."""
    n, d = xc.shape
    s = xc.T @ xc / n
    mu = float(np.trace(s) / d)
    # delta = min(1, b2 / d2) with d2 = ||S - mu I||^2 and b2 = (1/n^2) sum_k ||x_k x_k^T - S||^2, which
    # expands to (sum_k ||x_k||^4 / n - ||S||^2) / n
    s_sq = float((s ** 2).sum())
    d2 = s_sq - 2 * mu * float(np.trace(s)) + mu ** 2 * d
    b2 = (float(((xc ** 2).sum(1) ** 2).sum()) / n - s_sq) / n
    delta = min(1.0, b2 / d2) if d2 > 0 else 1.0
    return (1 - delta) * s + delta * mu * np.eye(d), delta


def fit_leace(x: np.ndarray, labels: np.ndarray) -> Leace:
    """Fit a categorical LEACE eraser on activations `x` (n, hidden) with class `labels` (n,)."""
    x = np.asarray(x, dtype=np.float64)
    classes, z_idx = np.unique(labels, return_inverse=True)
    z = np.eye(len(classes))[z_idx]
    mean = x.mean(0)
    xc = x - mean
    cov, delta = ledoit_wolf(xc)
    sigma_xz = xc.T @ (z - z.mean(0)) / len(x)  # (hidden, classes)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, evals.max() * 1e-12, None)
    w = (evecs * evals ** -0.5) @ evecs.T  # Sigma^{-1/2}
    w_inv = (evecs * evals ** 0.5) @ evecs.T  # Sigma^{1/2}
    q = orthonormal((w @ sigma_xz).T).T  # (hidden, rank), rank <= classes - 1
    return Leace(mean=mean, a=w_inv @ q, b=q.T @ w, shrinkage=delta)
