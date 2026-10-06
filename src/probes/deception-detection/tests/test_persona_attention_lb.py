"""Label-trained multi-head extraction: the Gram shortcut, the mixture, and a planted source."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs as pcs
from deception_detection.scripts import persona_attention_extract as pe
from deception_detection.scripts import persona_attention_lb as lb

SHAPE = "knowledge-denial (4v6)"


def _bank(seed=0, per_persona=4, length=6, hidden=12, k=3, planted=False):
    """A small synthetic bank over the knowledge roles (the 4 v 6 shape), with raw tokens X."""
    rng = np.random.default_rng(seed)
    personas = list(pcs.KNOWLEDGE_DENIAL_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    rows, X, C = [], [], []
    for persona in personas:
        side = 1 if persona in pcs.KNOWLEDGE_DENIAL_PERSONAS else -1
        offset = rng.normal(scale=2.0, size=hidden) * (np.arange(hidden) >= 2)  # persona noise off dims 0-1
        for r in range(per_persona):
            x = rng.normal(scale=0.3, size=(length, hidden)) + offset
            c = rng.normal(scale=0.3, size=(length, k))
            if planted:
                t = rng.integers(length)
                x[t] = rng.normal(scale=0.3, size=hidden)
                x[t, 0] += 3 * side
                c[t, 0] += 4
            rows.append({"persona": persona, "prompt_index": r % 5, "question_index": r, "offset": len(X) * length,
                         "length": length})
            X.append(x)
            C.append(c)
    frame = pd.DataFrame(rows)
    X = np.concatenate(X).astype(np.float32)
    bank = pe.Bank(rows=frame, tok_roll=np.repeat(np.arange(len(frame)), length), keep=np.ones(len(X), bool),
                   coords=np.concatenate(C).astype(np.float32), xb=X,
                   persona_of=frame["persona"].map({p: i for i, p in enumerate(pe.PERSONAS)}).to_numpy())
    return bank, X


def _explicit_scores(bank, X, U, pi, Z):
    """Pool every rollout with each head, average per persona, contrast, mix, score Z."""
    offsets = np.concatenate([[0], np.cumsum(bank.rows["length"].to_numpy())])
    idx = {p: i for i, p in enumerate(pe.PERSONAS)}
    pos, neg = SHAPES = lb.SHAPES[SHAPE]
    total = np.zeros(len(Z))
    for h in range(U.shape[0]):
        a = pe.attention(torch.from_numpy(U[h]), torch.from_numpy(bank.coords.astype(np.float64)),
                         torch.from_numpy(bank.keep), torch.from_numpy(bank.tok_roll.astype(np.int64)), len(offsets) - 1).numpy()
        z = np.stack([a[o0:o1] @ X[o0:o1] for o0, o1 in zip(offsets[:-1], offsets[1:])])
        v = {p: z[bank.persona_of == idx[p]].mean(0) for p in pos + neg}
        d = np.mean([v[p] for p in pos], axis=0) - np.mean([v[p] for p in neg], axis=0)
        total += pi[h] * (Z @ d)
    return total


def test_gram_scores_equal_the_explicit_pipeline_and_mix_linearly():
    bank, X = _bank()
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(15, X.shape[1])).astype(np.float32)
    gram = torch.from_numpy(X @ Z.T)
    U = rng.normal(size=(3, 3))
    rho = rng.normal(size=3)
    m = lb.LBObjective(bank, gram, rng.integers(0, 2, 15), SHAPE, n_heads=3, lam=0.1)
    with torch.no_grad():
        got = m.scores(torch.from_numpy(U), torch.from_numpy(rho)).numpy()
    pi = torch.softmax(torch.from_numpy(rho), 0).numpy()
    assert np.allclose(got, _explicit_scores(bank, X, U, pi, Z), atol=1e-4)
    # zero queries, one head: the mean-extraction contrast
    with torch.no_grad():
        base = m.scores(torch.zeros((1, 3), dtype=torch.float64), torch.zeros(1, dtype=torch.float64)).numpy()
    assert np.allclose(base, _explicit_scores(bank, X, np.zeros((1, 3)), np.ones(1), Z), atol=1e-4)


def test_labels_train_the_head_onto_the_planted_token_and_shuffled_labels_do_not():
    bank, X = _bank(planted=True)
    rng = np.random.default_rng(2)
    n = 120
    y = rng.integers(0, 2, n)
    Z = rng.normal(scale=0.5, size=(n, X.shape[1])).astype(np.float32)
    Z[:, 0] += 2 * (2 * y - 1)  # the label lives along the planted behaviour dimension
    Z[:, 2:] += rng.normal(scale=2.0, size=(n, X.shape[1] - 2))  # loud in the persona-noise dimensions
    gram = torch.from_numpy(X @ Z.T)
    real = lb.LBObjective(bank, gram, y, SHAPE, n_heads=2, lam=1e-3).fit(max_iter=40)
    base = lb.LBObjective(bank, gram, y, SHAPE, n_heads=1, lam=1e6)  # pinned at mean pooling
    everything = np.ones(n, bool)
    assert real.auroc(everything) > base.auroc(everything) + 0.05
    assert max(abs(real.U.detach().numpy()[:, 0])) > np.abs(real.U.detach().numpy()[:, 1:]).max()
    shuffled = lb.LBObjective(bank, gram, rng.permutation(y), SHAPE, n_heads=2, lam=1e-3).fit(max_iter=40)
    with torch.no_grad():
        s = shuffled.scores(shuffled.U, shuffled.rho).numpy()
    assert roc_auc_score(y, s) < real.auroc(everything) - 0.05


def test_several_heads_start_apart_and_a_single_head_starts_at_mean_pooling():
    bank, X = _bank()
    rng = np.random.default_rng(3)
    gram = torch.from_numpy(X @ rng.normal(size=(20, X.shape[1])).astype(np.float32).T)
    y = rng.integers(0, 2, 20)
    one = lb.LBObjective(bank, gram, y, SHAPE, n_heads=1, lam=0.1)
    assert (one.U.detach().numpy() == 0).all()
    three = lb.LBObjective(bank, gram, y, SHAPE, n_heads=3, lam=0.1)
    U = three.U.detach().numpy()
    assert not np.allclose(U[0], U[1]) and not np.allclose(U[1], U[2])
    assert np.abs(U).max() < 0.25
    again = lb.LBObjective(bank, gram, y, SHAPE, n_heads=3, lam=0.1)
    assert np.array_equal(again.U.detach().numpy(), U)  # seeded, so reproducible


def test_two_heads_fit_at_least_as_well_as_one_and_stay_distinct():
    """Two informative tokens per rollout, marked in different persona coordinates. One query can
    attend to both, so the heads need not diverge; but they must not be bit-identical after
    training, and the mixture must not lose to the single head in sample."""
    rng = np.random.default_rng(4)
    bank, X = _bank(planted=True, length=8)
    # plant a second behaviour token per rollout, marked in persona coordinate 1
    C, Xn = bank.coords.copy(), X.copy()
    for i, row in bank.rows.iterrows():
        a = int(row["offset"]); side = 1 if row["persona"] in pcs.KNOWLEDGE_DENIAL_PERSONAS else -1
        t = a + int(rng.integers(8))
        Xn[t] = rng.normal(scale=0.3, size=X.shape[1]); Xn[t, 1] += 3 * side; C[t, 1] += 4
    bank.coords, bank.xb = C, Xn
    n = 150
    y = rng.integers(0, 2, n)
    Z = rng.normal(scale=0.5, size=(n, X.shape[1])).astype(np.float32)
    Z[:, 0] += 2 * (2 * y - 1)
    Z[:, 1] += 2 * (2 * y - 1)
    Z[:, 2:] += rng.normal(scale=2.0, size=(n, X.shape[1] - 2))
    gram = torch.from_numpy(Xn @ Z.T)
    two = lb.LBObjective(bank, gram, y, SHAPE, n_heads=2, lam=1e-3).fit(max_iter=40)
    one = lb.LBObjective(bank, gram, y, SHAPE, n_heads=1, lam=1e-3).fit(max_iter=40)
    U = two.U.detach().numpy()
    assert not np.array_equal(U[0], U[1])
    everything = np.ones(n, bool)
    assert two.auroc(everything) >= one.auroc(everything) - 0.02
    assert two.auroc(everything) > 0.9


def test_source_rows_keep_small_sets_whole_and_subsample_id_stratified():
    y = np.array([0] * 3000 + [1] * 2432)
    keep = lb.source_rows("instruct_dishonesty__plain", len(y), y)
    assert len(keep) == lb.N_ID and abs(y[keep].mean() - y.mean()) < 0.01
    assert np.array_equal(lb.source_rows("convince_me__plain", 267, np.zeros(267)), np.arange(267))
