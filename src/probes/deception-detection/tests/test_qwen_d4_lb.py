import json

import numpy as np

from deception_detection.scripts.narcbench import qwen_d4_lb as QL
from deception_detection.scripts.narcbench import qwen_d4_pipeline as QP


def test_lb_meta_is_seeded_and_maps_roles():
    y = np.array([1, 0, 0, 1, 0])
    order, meta = QL.lb_meta(y)
    assert sorted(order) == list(range(5)) and np.array_equal(order, QL.lb_meta(y)[0])
    assert [m["role"] for m in meta] == ["colluder" if y[j] else "honest" for j in order]
    assert all(m["mode"] == "collusion" and m["channel"] == "public" and m["phase"] == "discussion" for m in meta)


def test_project_lb_streams_chunks_like_project(tmp_path):
    rng = np.random.default_rng(0)
    H, n_layers = 16, len(QL.LAYERS)
    rows = np.arange(100, 106)
    pts = rng.standard_normal((6, 2, n_layers, H)).astype(np.float32)
    acts = tmp_path / "acts"
    acts.mkdir()
    np.savez(acts / "000100_4.npz", rows=rows[:4], points=pts[:4], content=np.zeros((4, 0), np.float32))
    np.savez(acts / "000104_2.npz", rows=rows[4:], points=pts[4:], content=np.zeros((2, 0), np.float32))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"rows": {str(r): {"label": "HONEST", "set": "x"} for r in rows}}))
    bases = {"a": {L: QP.random_subspace(0, L, hidden=H) for L in QL.LAYERS},
             "b": {L: QP.random_subspace(1, L, hidden=H) for L in QL.LAYERS}}
    rids, got = QL.project_lb(bases, point=1, acts=acts, meta_path=meta)
    assert rids == list(rows)
    for k in bases:
        for i, L in enumerate(QL.LAYERS):
            assert np.allclose(got[k][L], QP.project(pts[:, 1, i], bases[k][L]), atol=1e-6)


def test_patch_returns_the_same_fit_for_every_fold():
    class Mod:
        pass

    fits = {(26, 0): "fit26", (27, 3): "fit27"}
    m = Mod()
    QL.patch(m, fits)
    assert m.train_deception_probe(None, None, 26, exclude_domain="audit", seed=0) == "fit26"
    assert m.train_deception_probe(None, None, 27, seed=3) == "fit27"
