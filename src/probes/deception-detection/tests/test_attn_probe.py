import numpy as np
import torch

from deception_detection.scripts.narcbench import attn_probe as AP


def _spans(n=40, seed=0):
    rng = np.random.default_rng(seed)
    spans = [rng.standard_normal((int(rng.integers(3, 30)), 3)).astype(np.float32) for _ in range(n)]
    y = np.arange(n) % 2
    for i in range(n):
        spans[i][-1, 0] += 3.0 * y[i]
    return spans, y


def test_padding_and_masking_do_not_change_a_row():
    spans, _ = _spans()
    mean, sd = AP.standardiser(spans)
    torch.manual_seed(0)
    probe = AP.AttentionProbe(3, 4)
    with torch.no_grad():
        probe.q.weight.normal_()
        probe.position_weight.normal_()
    together = AP.scores(probe, AP.batch(spans, mean, sd))
    alone = np.array([AP.scores(probe, AP.batch([s], mean, sd))[0] for s in spans])
    assert np.allclose(together, alone, atol=1e-5)


def test_zero_queries_are_mean_pooling_and_score_is_linear_in_pooled_vectors():
    spans, _ = _spans()
    mean, sd = AP.standardiser(spans)
    b = AP.batch(spans, mean, sd)
    probe = AP.AttentionProbe(3, 2)
    z = probe.pooled(b.x, b.mask, b.position).detach().numpy()
    means = np.stack([((s - mean) / sd).mean(0) for s in spans])
    assert np.allclose(z[:, 0], means, atol=1e-5) and np.allclose(z[:, 1], means, atol=1e-5)
    lin = z.reshape(len(spans), -1) @ AP.flat_direction(probe) + float(probe.v.bias.detach().sum())
    assert np.allclose(lin, AP.scores(probe, b), atol=1e-5)


def test_large_position_weight_attends_to_the_last_token():
    spans, _ = _spans()
    mean, sd = AP.standardiser(spans)
    b = AP.batch(spans, mean, sd)
    probe = AP.AttentionProbe(3, 1)
    with torch.no_grad():
        probe.position_weight.fill_(50.0)
    z = probe.pooled(b.x, b.mask, b.position).detach().numpy()[:, 0]
    assert np.allclose(z, np.stack([(s[-1] - mean) / sd for s in spans]), atol=1e-4)


def test_fit_learns_a_last_token_signal_and_cv_is_grouped():
    spans, y = _spans(60)
    mean, sd = AP.standardiser(spans)
    b = AP.batch(spans, mean, sd)
    probe = AP.fit(b, y, 2, 1e-3, seed=0)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, AP.scores(probe, b)) > 0.95
    wd, acc = AP.choose_weight_decay(b, y, [f"run{i // 4}" for i in range(60)], 1, seed=0)
    assert wd in AP.WD_GRID and set(acc) == set(AP.WD_GRID)
