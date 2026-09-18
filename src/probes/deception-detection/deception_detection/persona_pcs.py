"""Persona vectors and their principal components, for arXiv 2605.09391.

The method builds a low-dimensional "persona space" from role-play rollouts rather than from
any evaluation dataset, then projects dataset activations into it:

    v_p        mean-pooled activation for persona p over its rollouts        (Eq. 1)
    v~_p       = v_p - mean over all personas in P                           (Eq. 2)
    u_1..u_k   PCA of the centred v~                                         (Eq. 3)
    PC_k(z)    = <z, u_k>

Because the basis is fitted on persona vectors alone and never sees a dataset, projecting into
it introduces no train/test leakage -- the same directions are used for every train/test pair.

P here is the 15 deception + honesty personas, not the full 26-persona bank. That restriction is
deliberate: on the 15 the leading component is essentially the deception direction
(|cos(PC1, deception_axis)| = 0.95) and separates the two groups perfectly, while over all 26 the
sycophancy axis blurs it (0.72). See `docs/` note in the module tests.
"""

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

BANK_PATH = REPO_ROOT / "paper16_L20_sharedquestions" / "paper16_L20.pt"

# The bank's own `role_groups` field is unusable: it serialises each group's KEY NAMES
# (["contrast_with", "note", "roles"]) rather than its contents, in both the .pt and the .json.
# The deception group is published separately in deceptive/deceptive_L20.pt; the honesty group was
# recovered from the identity mean(honesty) = mean(deception) - deception_axis, which picks out a
# unique 7-subset of the remaining 17 at residual 2.9e-07 (float32 round-off).
# `recover_honesty_group()` below re-derives it, and the tests assert it still matches.
DECEPTION_PERSONAS = (
    "pc_handler",
    "pc_broker",
    "pc_impression_manager",
    "pc_coverup_operator",
    "pc_gatekeeper",
    "pc_spin_doctor",
    "pc_plausible_denier",
    "pc_narrative_manager",
)
HONESTY_PERSONAS = (
    "pc_archivist",
    "pc_disclosure_counsel",
    "pc_documentarian",
    "pc_fact_checker",
    "pc_ombudsman",
    "pc_transparency_officer",
    "pc_whistleblower",
)


@dataclass
class PersonaBank:
    """The persona vectors at one layer, plus the contrast axes fitted alongside them."""

    personas: np.ndarray  # [n_personas, hidden]
    names: list[str]
    axes: dict[str, np.ndarray]  # "assistant" | "deception" | "sycophancy" | "unified"
    n_rollouts: list[int]
    layer: int
    model: str

    def vectors_for(self, names: tuple[str, ...]) -> np.ndarray:
        missing = [n for n in names if n not in self.names]
        if missing:
            raise KeyError(f"personas not in bank: {missing}")
        return self.personas[[self.names.index(n) for n in names]]

    def axis_unit(self, name: str = "deception") -> np.ndarray:
        """A contrast axis, normalised.

        The bank stores axes as raw `mean(positive) - mean(negative)`, so they are not unit
        vectors (deception has norm 0.890). AUROC is invariant to the rescale, but any magnitude
        or cosine comparison is meaningless without it.
        """
        axis = self.axes[name]
        return axis / np.linalg.norm(axis)


def load_bank(path: Path | str = BANK_PATH) -> PersonaBank:
    raw = torch.load(Path(path), weights_only=False)
    axis_names = list(raw["axis_names"])
    axes = {name: raw["axes"][i].double().numpy() for i, name in enumerate(axis_names)}
    return PersonaBank(
        personas=raw["personas"].double().numpy(),
        names=list(raw["persona_names"]),
        axes=axes,
        n_rollouts=list(raw["persona_n_rollouts"]),
        layer=int(raw["layer"]),
        model=str(raw["subject_model"]),
    )


def recover_honesty_group(bank: PersonaBank, size: int = 7) -> tuple[str, ...]:
    """Re-derive the honesty group from the deception axis.

    `deception_axis = mean(deception) - mean(honesty)`, so `mean(honesty)` is known exactly, and
    exactly one subset of the remaining personas averages to it. Used only to verify the
    hard-coded HONESTY_PERSONAS still matches this bank -- not on any hot path.
    """
    target = bank.vectors_for(DECEPTION_PERSONAS).mean(0) - bank.axes["deception"]
    pool = [n for n in bank.names if n not in DECEPTION_PERSONAS and n != "default"]

    best_err, best = np.inf, ()
    for combo in itertools.combinations(pool, size):
        err = float(np.linalg.norm(bank.vectors_for(combo).mean(0) - target))
        if err < best_err:
            best_err, best = err, combo
    logger.debug(f"recovered honesty group at residual {best_err:.2e}: {best}")
    return tuple(sorted(best))


@dataclass
class PersonaGeometry:
    """A fitted persona subspace: the centred vectors, their PCs, and the contrast axis."""

    components: np.ndarray  # [k, hidden], orthonormal rows
    explained_variance_ratio: np.ndarray  # [k]
    axis: np.ndarray  # [hidden], unit norm
    persona_names: tuple[str, ...]
    group_of: dict[str, str]
    mean: np.ndarray  # [hidden], the centring vector (Eq. 2)

    def project(self, z: np.ndarray, k: int | None = None) -> np.ndarray:
        """PC scores for pooled activations. z is [n, hidden] -> [n, k]."""
        comps = self.components if k is None else self.components[:k]
        return z @ comps.T

    def axis_score(self, z: np.ndarray) -> np.ndarray:
        """Zero-shot contrastive score <z, d_hat>, higher meaning more deceptive."""
        return z @ self.axis

    def alignment(self) -> np.ndarray:
        """|cos| between each PC and the contrast axis. PC1 should dominate."""
        return np.abs(self.components @ self.axis)


def fit_geometry(
    bank: PersonaBank,
    k: int = 3,
    deception: tuple[str, ...] = DECEPTION_PERSONAS,
    honesty: tuple[str, ...] = HONESTY_PERSONAS,
) -> PersonaGeometry:
    """Centre the persona set and fit its top-k principal components (Eq. 2-3).

    P is the union of `deception` and `honesty` -- 15 personas by default, not the full bank.
    A PC's sign is arbitrary, so each is oriented to point along the deception axis; without
    this, "higher score = more deceptive" would hold or not at random per run.
    """
    names = tuple(deception) + tuple(honesty)
    vectors = bank.vectors_for(names)
    mean = vectors.mean(0)
    centred = vectors - mean

    max_k = min(len(names) - 1, vectors.shape[1])
    if k > max_k:
        raise ValueError(f"cannot take {k} components from {len(names)} personas (rank <= {max_k})")

    pca = PCA(n_components=max_k).fit(centred)
    components = pca.components_[:k].copy()
    axis = bank.axis_unit("deception")

    # orient each PC along the deception axis so score direction is reproducible
    for i, comp in enumerate(components):
        if float(comp @ axis) < 0:
            components[i] = -comp

    group_of = {n: "deception" for n in deception} | {n: "honesty" for n in honesty}
    return PersonaGeometry(
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_[:k].copy(),
        axis=axis,
        persona_names=names,
        group_of=group_of,
        mean=mean,
    )
