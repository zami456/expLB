"""Shared helpers for the trajectory study: the D9 contrast at every layer, AUROC under bootstrap
weights, and the freeze.

D9 is the thesis's extended persona contrast (12 deception and knowledge-denial roles against 13
honesty and candour roles, `persona_pcs._axes_from_groups`). The bank stores every role at all 80
layers, so the raw contrast c[l] exists at every depth. Because a contrast of means is linear, the
contrast of what the layers write is c[l'] - c[l]: "D9 on differences" projects h[l'] - h[l] onto
unit(c[l'] - c[l]).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from deception_detection import persona_pcs as pcs
from deception_detection.scripts.trajectory import spec as T

POS = pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS
NEG = pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def bank_dir() -> Path:
    from deception_detection.scripts import recall_denial as rd

    return Path(rd.Inputs("llama").spec.bank_dir("ext"))


def extended_contrasts(layers, folder: Path | None = None) -> dict[int, np.ndarray]:
    """Raw D9 contrast c[l] = mean(POS) - mean(NEG) at each requested layer, from one read of each
    role file (the bank's per-layer loader re-reads all 36 files per layer)."""
    folder = folder or bank_dir()
    layers = [int(x) for x in layers]
    sums = {}
    for group, names in (("pos", POS), ("neg", NEG)):
        acc = None
        for name in names:
            payload = torch.load(folder / f"{name}.pt", weights_only=True)
            stored = payload["vector"]
            listed = [int(x) for x in payload.get("layers", range(stored.shape[0]))]
            rows = stored[[listed.index(l) for l in layers]].double().numpy()
            acc = rows if acc is None else acc + rows
        sums[group] = acc / len(names)
    c = sums["pos"] - sums["neg"]
    return {l: c[i] for i, l in enumerate(layers)}


# --- AUROC -------------------------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def weighted_auroc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC of score columns `s` (n,) or (n, k) with row weights `w` (draws, n) -- a bootstrap draw
    is the count of each row in the resample, so one sort per column serves every draw. Ties count
    one half. Returns (draws,) or (draws, k)."""
    s2 = s[:, None] if s.ndim == 1 else s
    y = np.asarray(y).astype(bool)
    w = np.atleast_2d(w).astype(np.float64)
    out = np.empty((w.shape[0], s2.shape[1]))
    for j in range(s2.shape[1]):
        col = s2[:, j]
        order = np.argsort(col, kind="mergesort")
        sv = col[order]
        yv = y[order]
        new = np.r_[True, sv[1:] != sv[:-1]]
        if new.all():  # no ties: negatives below each positive, one cumulative sum per draw
            wo = w[:, order]
            wneg = wo * (~yv)
            below = np.cumsum(wneg, axis=1) - wneg
            wpos = wo * yv
            out[:, j] = (wpos * below).sum(1) / (wpos.sum(1) * wneg.sum(1))
            continue
        # group equal scores
        gid = np.cumsum(new) - 1
        ng = gid[-1] + 1
        wp = np.zeros((w.shape[0], ng))
        wn = np.zeros((w.shape[0], ng))
        wo = w[:, order]
        np.add.at(wp.T, gid[yv], wo[:, yv].T)
        np.add.at(wn.T, gid[~yv], wo[:, ~yv].T)
        below = np.cumsum(wn, axis=1) - wn  # negatives strictly below each group
        num = (wp * (below + 0.5 * wn)).sum(1)
        out[:, j] = num / (wp.sum(1) * wn.sum(1))
    return out[:, 0] if s.ndim == 1 else out


def stratified_counts(y: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(draws, n) counts of a resample drawn within each class (HP-C and HP-KR are ~6% lies)."""
    y = np.asarray(y).astype(bool)
    counts = np.zeros((n_draws, len(y)), np.int32)
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)
        picks = rng.integers(0, len(idx), size=(n_draws, len(idx)))
        for b in range(n_draws):
            np.add.at(counts[b], idx[picks[b]], 1)
    return counts


# --- freeze ------------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files() -> list[Path]:
    """The files whose hashes the freeze records: this package and the modules it imports."""
    scripts = T.HERE.parent
    dd = scripts.parent
    own = sorted(p for p in T.HERE.glob("*.py")) + [T.HERE / "PREREGISTRATION.md"]
    imported = [scripts / "persona_ood.py", scripts / "recall_denial.py", scripts / "layer_sweep.py",
                scripts / "narcbench" / "score.py", scripts / "narcbench" / "rose.py",
                scripts / "narcbench" / "nulls.py", scripts / "narcbench" / "rows.py",
                scripts / "narcbench" / "spec.py", scripts / "narcbench" / "freeze.py", dd / "persona_pcs.py"]
    return own + imported


def freeze_block() -> dict[str, str]:
    """{path: sha256} of the last FREEZE block in freeze.txt."""
    if not T.FREEZE.exists():
        raise SystemExit(f"{T.FREEZE} does not exist: freeze the design first")
    text = T.FREEZE.read_text(encoding="utf-8")
    if "== FREEZE" not in text:
        raise SystemExit("no FREEZE block in freeze.txt")
    block = text.split("== FREEZE")[-1]
    out = {}
    for line in block.splitlines():
        if line.startswith("sha256 "):
            _, digest, path = line.split(" ", 2)
            out[path] = digest
    return out


def check_freeze() -> None:
    """Refuse to score if any frozen file changed since the last FREEZE block (a change needs a logged
    POST-FREEZE EDIT followed by a new block)."""
    want = freeze_block()
    changed = [p for p, d in want.items() if not Path(p).exists() or sha256(Path(p)) != d]
    if changed:
        raise SystemExit("changed since the freeze: " + ", ".join(changed))
