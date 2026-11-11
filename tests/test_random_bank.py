"""`random_bank.py`: the loader, the contrast builder, splits, scoring shortcuts and the null p."""

import numpy as np
import pytest
import torch

from experiments.external import random_bank as RB


def _write_bank(folder, names, layers=4, hidden=6, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        torch.save({"vector": torch.tensor(rng.standard_normal((layers, hidden)), dtype=torch.float32),
                    "layers": list(range(layers)), "type": "pos_3", "role": n, "n": 5}, folder / f"{n}.pt")


def _groups(names):
    out, k = {}, 0
    for g, size in RB.GROUP_SIZES.items():
        out[g] = names[k:k + size]
        k += size
    return out


def test_loader_takes_any_role_names(tmp_path):
    names = [f"role{i}" for i in range(35)] + ["default"]
    _write_bank(tmp_path, names)
    got, X, counts = RB.load_roles(tmp_path, layers=[1, 3])
    assert got == sorted(names) and X.shape == (36, 2, 6) and counts["role0"] == 5
    with pytest.raises(IndexError):
        RB.load_roles(tmp_path, layers=[9])


def test_builder_matches_persona_pcs_axes(tmp_path):
    from deception_detection import persona_pcs as pcs

    tg = RB.thesis_groups()
    roles = sorted({r for v in tg.values() for r in v} | set(pcs.PUBLISHED_PERSONAS if hasattr(pcs, "PUBLISHED_PERSONAS") else ()))
    _write_bank(tmp_path, roles + ["default"])
    names, X, _ = RB.load_roles(tmp_path)
    vec = {n: X[i, 2] for i, n in enumerate(names)}
    axes = pcs._axes_from_groups(vec)
    for c in RB.CONTRASTS:
        assert np.allclose(RB.contrast(vec, tg, c), axes[RB.AXIS_OF[c]], atol=1e-12)


def test_weights_equal_the_contrast():
    rng = np.random.default_rng(1)
    names = [f"r{i}" for i in range(35)] + ["default"]
    X = rng.standard_normal((36, 7))
    g = _groups(names[:35])
    vec = {n: X[i] for i, n in enumerate(names)}
    for c in RB.CONTRASTS:
        w = RB.weights(names, g, c)
        assert w[names.index("default")] == 0 and abs(w.sum()) < 1e-12
        assert np.allclose(w @ X, RB.contrast(vec, g, c))


def test_split_checks_and_seed_rule():
    roles = [f"r{i}" for i in range(35)]
    g = RB.regenerate_split(roles, 3)
    RB.check_split(g, roles)
    assert g == RB.regenerate_split(roles, 3)
    bad = {k: list(v) for k, v in g.items()}
    bad["A"][0] = bad["B"][0]
    with pytest.raises(ValueError):
        RB.check_split(bad, roles)


def test_auroc_and_transfer_shortcuts():
    from sklearn.metrics import roc_auc_score

    from experiments.persona import persona_ood as P

    rng = np.random.default_rng(2)
    order = ["a", "b", "c"]
    data = {}
    for n in order:
        y = (rng.random(200) < 0.3).astype(int)
        data[n] = (y, y * rng.normal(0.4, 1, 200) + rng.normal(0, 1, 200))
    S = np.stack([data["a"][1], np.round(data["a"][1], 1)], 1)
    assert np.allclose(RB.auroc_columns(data["a"][0], S),
                       [roc_auc_score(data["a"][0], S[:, j]) for j in range(2)])
    au = {n: roc_auc_score(y, x) for n, (y, x) in data.items()}
    signs = {n: RB.fitted_sign(x, y) for n, (y, x) in data.items()}
    assert all(RB.shortcut_sign(x, y) == signs[n] for n, (y, x) in data.items())
    pooled = {n: type("Pooled", (), {"z": x[:, None], "y": y})() for n, (y, x) in data.items()}
    m = P.transfer_matrix(lambda z: z, pooled, order)
    assert abs(RB.transfer_1d(au, signs, order) - P.off_diagonal_mean(m)) < 1e-12


def test_permutation_p():
    null = np.array([0.1, 0.2, 0.3])
    assert RB.permutation_p(0.25, null) == 2 / 4
    assert RB.permutation_p(0.5, null) == 1 / 4
