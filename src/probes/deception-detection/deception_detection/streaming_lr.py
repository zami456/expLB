"""The fit `LogisticRegressionDetector` performs, for data too large for it to copy.

`LogisticRegressionDetector.fit` standardises with sklearn's `StandardScaler` and then fits sklearn's
`LogisticRegression` (lbfgs, L2, no intercept). For lbfgs, sklearn copies the standardised matrix to
float64. At ~130k response tokens x 8192 dimensions, that copy (8.6 GB) plus the two float32 matrices
before it exceeds a 16 GB machine.

`fit` solves the same problem on a single float32 matrix, read in row blocks and standardised on the
fly. It follows sklearn 1.6's lbfgs path step for step:

- the objective, (1/n) * sum of log-losses + ||w||^2 / (2 C n);
- a zero start;
- scipy's L-BFGS-B with maxls 50, gtol `tol` and ftol 64 * eps;
- `StandardScaler`'s rule for constant features.

So it returns sklearn's solution to optimiser tolerance. `tests/test_streaming_lr.py` checks that
against sklearn itself.

`X` may also be a torch tensor, such as bfloat16 activations kept at the precision they were cached
in. Every block is converted to float64 before use, so the fit is the same as on a float32 copy,
at half its memory.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = np.finfo(np.float64).eps


@dataclass
class StreamingFit:
    coef: np.ndarray  # [d], in standardised units, as sklearn's coef_
    mean: np.ndarray  # [d], StandardScaler.mean_
    scale: np.ndarray  # [d], StandardScaler.scale_
    n_iter: int
    converged: bool
    message: str


def rows(X, i: int, block: int) -> np.ndarray:
    """Rows i to i + block of an array or a torch tensor, as float64."""
    part = X[i : i + block]
    return part.double().numpy() if hasattr(part, "double") else part.astype(np.float64)


def standardiser(X, block: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """`StandardScaler`'s mean_ and scale_: float64 accumulators, population variance, two passes."""
    n, d = X.shape
    total = np.zeros(d)
    for i in range(0, n, block):
        total += rows(X, i, block).sum(axis=0)
    mean = total / n
    squares = np.zeros(d)
    for i in range(0, n, block):
        centred = rows(X, i, block) - mean
        squares += (centred * centred).sum(axis=0)
    var = squares / n
    scale = np.sqrt(var)
    # sklearn.preprocessing._data._is_constant_feature, then _handle_zeros_in_scale
    constant = var <= n * EPS * var + (n * mean * EPS) ** 2
    scale[constant] = 1.0
    return mean, scale


def fit(X, y: np.ndarray, C: float, max_iter: int = 1000, tol: float = 1e-4,
        block: int = 4096) -> StreamingFit:
    """L2 logistic regression without intercept on standardised `X`, as sklearn's lbfgs fits it.

    `X` is [n, d], a float array or torch tensor, and is never copied whole. `y` is [n], in {0, 1}.
    """
    n, d = X.shape
    y = np.asarray(y, dtype=np.float64)
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be 0/1")
    mean, scale = standardiser(X, block)
    l2 = 1.0 / (C * n)

    def loss_and_gradient(w: np.ndarray) -> tuple[float, np.ndarray]:
        # ((X - mean) / scale) @ w, without forming the standardised matrix
        v = w / scale
        offset = mean @ v
        loss, residual_sum, gradient = 0.0, 0.0, np.zeros(d)
        for i in range(0, n, block):
            part = rows(X, i, block)
            raw = part @ v - offset
            target = y[i : i + block]
            loss += float(np.sum(np.logaddexp(0.0, raw) - target * raw))
            residual = expit(raw) - target
            gradient += part.T @ residual
            residual_sum += float(residual.sum())
        gradient = (gradient - mean * residual_sum) / scale / n + l2 * w
        return loss / n + 0.5 * l2 * float(w @ w), gradient

    result = minimize(
        loss_and_gradient, np.zeros(d), method="L-BFGS-B", jac=True,
        options={"maxiter": max_iter, "maxls": 50, "gtol": tol, "ftol": 64 * EPS},
    )
    return StreamingFit(coef=result.x, mean=mean, scale=scale, n_iter=int(result.nit),
                        converged=bool(result.success), message=str(result.message))
