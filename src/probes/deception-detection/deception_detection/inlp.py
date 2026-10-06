"""Iterative nullspace projection (INLP), ported from the reference implementation.

Ravfogel et al. 2020, "Null It Out: Guarding Protected Attributes by Iterative Nullspace
Projection". Reference code: https://github.com/shauli-ravfogel/nullspace_projection, whose
`src/debias.py::get_debiasing_projection` this follows step for step (MIT licence, below):

    for i in 1..num_classifiers:
        fit a fresh linear classifier on the projected X_train, score its accuracy on X_dev
        below min_accuracy: skip it
        W_i = its coef_                                    the direction it reads
        P   = projection onto N(W_1) ∩ ... ∩ N(W_i)        Ben-Israel 2013
        X_train, X_dev = P X_train, P X_dev                always from the ORIGINAL X

One change, forced by scale. The reference materialises every rowspace projection P_R(W_i), and P
itself, as dense d x d matrices, keeps all of the former in a list, and runs scipy.linalg.orth (an
SVD) on a d x d matrix every iteration. That is fine at d = 300 and not at d = 8192, where each
matrix is 512 MiB in float64: the list alone outgrows a 16 GB machine's free memory in about ten
iterations. Ben-Israel's identity

    N(W_1) ∩ ... ∩ N(W_n) = N(P_R(W_1) + ... + P_R(W_n))

only uses the range of that sum, which is span{W_1, ..., W_n}. So P = I - B B^T for B an
orthonormal basis of the stacked W's, a d x n matrix, and P x = x - B (B^T x). The port keeps B and
never forms anything d x d. tests/test_inlp.py runs the verbatim reference beside it at small d and
checks that the two agree.

Dropped from the reference: `by_class` (it needs main-task labels, and there is no main task here)
and `dropout_rate` (off by default, and the reference itself warns that it breaks W_i ⟂ W_j).
`is_autoregressive` is always on.

One addition: the loop stops at the first all-zero classifier. With an L2-penalised solver such as
lbfgs this is where the signal runs out -- once the projected class-mean difference is small enough,
the gradient at w = 0 is already inside the solver's tolerance and it returns w = 0 exactly. That
is a fixed point: a zero W leaves P unchanged, so every later iteration refits the same zero
classifier. The reference keeps appending those zeros, which change nothing; the port stops.

Used here the other way round from the paper. INLP removes what its classifiers find; this keeps
it. The W_i are an ordered set of mutually orthogonal directions, each still able to decode the
label once every earlier one is gone, and the span of the first k is a candidate label subspace to
project onto -- the counterpart of the persona PCs in persona_pcs.py.
"""

# Adapted from https://github.com/shauli-ravfogel/nullspace_projection (src/debias.py,
# src/classifier.py), under the following licence:
#
# MIT License
#
# Copyright (c) 2022 Shauli Ravfogel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.metrics import roc_auc_score

from deception_detection.log import logger


class SKlearnClassifier:
    """The reference's classifier wrapper (src/classifier.py), unchanged in behaviour."""

    def __init__(self, model: Any):
        self.model = model

    def train_network(
        self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray
    ) -> float:
        """Fit on train; return accuracy on dev."""
        self.model.fit(X_train, Y_train)
        return float(self.model.score(X_dev, Y_dev))

    def get_weights(self) -> np.ndarray:
        """The fitted coef_, always 2-d."""
        w = self.model.coef_
        if len(w.shape) == 1:
            w = np.expand_dims(w, 0)
        return w


def rowspace_basis(W: np.ndarray) -> np.ndarray:
    """Orthonormal basis [d, r] of W's rowspace -- the factor of the reference's
    `get_rowspace_projection`, which returns basis @ basis.T.

    Each row is normalised first. The reference sums one unit-rank projection per W_i before its
    SVD, so every classifier counts equally however large its weights; normalising the rows gives
    the SVD here the same conditioning.
    """
    if np.allclose(W, 0):
        return np.zeros((W.shape[1], 0))
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    unit = W[norms.ravel() > 0] / norms[norms.ravel() > 0]
    return scipy.linalg.orth(unit.T)


def project_to_nullspace(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """P X for P = I - basis basis^T, applied without forming P."""
    return X - (X @ basis) @ basis.T


@dataclass
class INLPResult:
    """The classifiers INLP found, in order, and how each one did."""

    Ws: np.ndarray  # [T, d] each kept classifier's coef_, in INLP order
    trace: pd.DataFrame  # one row per iteration: accuracies, dev AUROC, weight norm, kept or not

    def basis(self, k: int | None = None) -> np.ndarray:
        """Orthonormal rows [k, d] spanning Ws[:k], nested in INLP order.

        QR rather than `rowspace_basis`: an SVD orders its vectors by singular value, which spans
        the same space but scrambles the order, and the order is what makes k a nested sweep. Row
        i is W_i with the parts along W_1..W_{i-1} removed -- a no-op up to solver tolerance, since
        each W_i was fitted on data with those directions already gone.

        Rows are normalised first. Weight norms fall by orders of magnitude as the signal runs
        out, and the independence check below should see directions, not magnitudes.
        """
        W = self.Ws if k is None else self.Ws[:k]
        if len(W) == 0:
            return np.zeros((0, self.Ws.shape[1]))
        unit = W / np.linalg.norm(W, axis=1, keepdims=True)
        q, r = np.linalg.qr(unit.T)
        if np.abs(np.diag(r)).min() < 1e-8:
            raise ValueError("classifier weights are linearly dependent; the basis is degenerate")
        signs = np.sign(np.diag(r))  # QR may flip a column; keep each row pointing along its W_i
        return (q * signs).T

    def project(self, x: np.ndarray, k: int | None = None) -> np.ndarray:
        """Coordinates of x in the span of the first k classifiers. [n, d] -> [n, k]."""
        return x @ self.basis(k).T

    def nullspace(self, x: np.ndarray, k: int | None = None) -> np.ndarray:
        """x with the first k classifiers' span removed -- the reference's P x."""
        b = self.basis(k)
        return x - (x @ b.T) @ b


def get_debiasing_projection(
    classifier_class: type,
    cls_params: dict[str, Any],
    num_classifiers: int,
    input_dim: int,
    min_accuracy: float,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
) -> INLPResult:
    """Run INLP. The reference's name and argument order, so the two read side by side.

    Returns the classifiers rather than the reference's (P, rowspace_projections, Ws): P is never
    materialised (see module docstring), and `INLPResult.nullspace` applies it on demand.

    `min_accuracy` keeps the reference's `continue` semantics: a classifier below it is recorded in
    the trace and skipped. With a deterministic solver such as lbfgs, every later iteration then
    refits the identical classifier on identical data and is skipped too, so a threshold above 0
    acts as an early stop. An all-zero classifier ends the loop (see module docstring), so the
    trace can be shorter than `num_classifiers`.
    """
    if X_train.shape[1] != input_dim or X_dev.shape[1] != input_dim:
        raise ValueError(f"expected {input_dim}-d inputs, got {X_train.shape} and {X_dev.shape}")

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    Ws: list[np.ndarray] = []
    rows = []

    for i in range(num_classifiers):
        clf = SKlearnClassifier(classifier_class(**cls_params))
        acc = clf.train_network(X_train_cp, Y_train, X_dev_cp, Y_dev)

        # diagnostics beyond the reference, which records dev accuracy only
        dev_auroc = (
            float(roc_auc_score(Y_dev, clf.model.decision_function(X_dev_cp)))
            if hasattr(clf.model, "decision_function") and len(set(Y_dev.tolist())) == 2
            else float("nan")
        )
        W = clf.get_weights()
        zero = bool(np.allclose(W, 0))  # the reference's own test for an empty rowspace
        rows.append(
            {
                "iteration": i + 1,
                "train_accuracy": float(clf.model.score(X_train_cp, Y_train)),
                "dev_accuracy": acc,
                "dev_auroc": dev_auroc,
                "weight_norm": float(np.linalg.norm(W)),
                "kept": acc >= min_accuracy and not zero,
            }
        )
        logger.info(f"INLP iteration {i + 1}: dev accuracy {acc:.4f}, dev AUROC {dev_auroc:.4f}")
        if zero:
            logger.info(f"INLP: classifier {i + 1} has zero weights, a fixed point; stopping")
            break
        if acc < min_accuracy:
            continue

        Ws.append(W)

        # projection onto the intersection of all nullspaces so far, recomputed from the original
        # X each time, as the reference does to stop W_i's inexact orthogonality from compounding
        basis = rowspace_basis(np.vstack(Ws))
        X_train_cp = project_to_nullspace(X_train, basis)
        X_dev_cp = project_to_nullspace(X_dev, basis)

    return INLPResult(
        Ws=np.vstack(Ws) if Ws else np.zeros((0, input_dim)),
        trace=pd.DataFrame(rows),
    )


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between two subspaces given as orthonormal rows.

    1 means a shared direction, 0 an orthogonal one. Descending, min(len(a), len(b)) of them.
    """
    return np.linalg.svd(a @ b.T, compute_uv=False)
