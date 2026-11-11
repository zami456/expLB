import numpy as np
import torch

from extensions.narcbench import qwen_d4_pipeline as QP


def _cell(path, vec, n, layers=(26, 27)):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": torch.tensor(vec, dtype=torch.float32), "layers": list(layers), "n": n}, path)


def test_combine_is_the_mean_over_kept_replies(tmp_path):
    store, out = tmp_path / "store", tmp_path / "bank"
    a, b = np.ones((2, 4)), np.full((2, 4), 3.0)
    _cell(store / "pc_x" / "q000.pt", a, 2)
    _cell(store / "pc_x" / "q001.pt", b, 3)
    _cell(store / "pc_y" / "q000.pt", a, 1)  # one kept reply: ineligible at min_count 2
    _cell(store / "default" / "q000.pt", b, 1)  # default is always kept
    (store / "pc_z").mkdir()  # no kept reply at all
    out.mkdir()
    torch.save({"stale": 1}, out / "pc_old.pt")  # a role from an earlier combine must not survive
    counts = QP.combine(store, out, min_count=2)
    assert counts == {"default": 1, "pc_x": 5, "pc_y": 1, "pc_z": 0}
    got = torch.load(out / "pc_x.pt", weights_only=True)
    assert np.allclose(got["vector"].numpy(), (2 * a + 3 * b) / 5) and got["n"] == 5 and got["layers"] == [26, 27]
    assert sorted(p.stem for p in out.glob("*.pt")) == ["default", "pc_x"]


def test_identity_projection_is_exact():
    X = np.random.default_rng(0).standard_normal((7, 16)).astype(np.float32)
    assert np.array_equal(QP.project(X, np.eye(16)), X)
    B = QP.random_subspace(0, 26, hidden=16)
    assert QP.project(X, B).shape == (7, 3) and QP.project(X, B).dtype == np.float32


def test_random_subspaces_are_seeded_and_orthonormal():
    B = QP.random_subspace(3, 28, hidden=64)
    assert np.allclose(B @ B.T, np.eye(3))
    assert np.array_equal(B, QP.random_subspace(3, 28, hidden=64))
    assert not np.allclose(B, QP.random_subspace(4, 28, hidden=64))
