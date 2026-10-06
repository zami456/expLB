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

# The other contrast in the bank, recovered the same way from the sycophancy axis (residual
# 2.7e-07). Together with deception, honesty and `default` these account for all 26 personas.
SYCOPHANCY_PERSONAS = (
    "pc_cheerleader",
    "pc_enabler",
    "pc_flatterer",
    "pc_people_pleaser",
    "pc_yes_man",
)
ANTI_SYCOPHANCY_PERSONAS = (
    "pc_blunt_advisor",
    "pc_contrarian",
    "pc_critic",
    "pc_skeptic",
    "pc_tough_love_mentor",
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


# One file per persona, each holding `vector[80, 8192]` -- the same run as the single-layer bank
# above, extended to every layer. The layer-20 slice is bit-identical to `paper16_L20.pt`.
VECTORS_DIR = REPO_ROOT / "persona vectors" / "vectors_s3"

# A later, larger extraction that adds ten roles built around *denial of knowledge* rather than
# assertion of falsehood. That distinction is why HP-KR inverts under the original axis: its
# deceptive response is "I don't know" while its honest one is "yes, I know", so an axis fitted
# on personas that all assert (handler, spin_doctor, narrative_manager) points the wrong way.
#
# NOT a superset of VECTORS_DIR -- all 26 original personas were re-extracted here with more
# rollouts (default 100 -> 150, pc_handler 64 -> 106), so vectors differ by up to 0.78 against a
# typical norm of ~4.1. Comparisons across the two folders confound "more rollouts" with "more
# personas"; run the original groups from *this* folder to separate them.
EXTENDED_VECTORS_DIR = REPO_ROOT / "persona including knowlede denial"

KNOWLEDGE_DENIAL_PERSONAS = (
    "pc_amnesiac_witness",
    "pc_deniable_expert",
    "pc_feigned_novice",
    "pc_stonewaller",
)
KNOWLEDGE_CANDOUR_PERSONAS = (
    "pc_calibrated_reporter",
    "pc_candid_witness",
    "pc_forthright_expert",
    "pc_honest_novice",
    "pc_plain_dealer",
    "pc_scrupulous_witness",
)

# These files ship no axes, but all four reconstruct from group means. Verified against the stored
# layer-20 axes to ~1e-7 (float32 round-off); `tests/test_persona_pcs.py` pins the equivalence.
def _axes_from_groups(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def mean_of(names: tuple[str, ...]) -> np.ndarray:
        return np.stack([vectors[n] for n in names]).mean(0)

    others = tuple(n for n in vectors if n != "default")
    axes = {
        "deception": mean_of(DECEPTION_PERSONAS) - mean_of(HONESTY_PERSONAS),
        "sycophancy": mean_of(SYCOPHANCY_PERSONAS) - mean_of(ANTI_SYCOPHANCY_PERSONAS),
        "unified": (
            mean_of(DECEPTION_PERSONAS + SYCOPHANCY_PERSONAS)
            - mean_of(HONESTY_PERSONAS + ANTI_SYCOPHANCY_PERSONAS)
        ),
        "assistant": vectors["default"] - mean_of(others),
    }
    # only present in the extended folder
    if set(KNOWLEDGE_DENIAL_PERSONAS + KNOWLEDGE_CANDOUR_PERSONAS) <= set(vectors):
        axes["denial"] = mean_of(KNOWLEDGE_DENIAL_PERSONAS) - mean_of(KNOWLEDGE_CANDOUR_PERSONAS)
        axes["extended"] = mean_of(DECEPTION_PERSONAS + KNOWLEDGE_DENIAL_PERSONAS) - mean_of(
            HONESTY_PERSONAS + KNOWLEDGE_CANDOUR_PERSONAS
        )
    return axes


def load_bank_at_layer(layer: int, path: Path | str = VECTORS_DIR) -> PersonaBank:
    """Build a bank at any layer from the per-persona all-layer vectors.

    `load_bank()` reads a single file that only exists at layer 20. This reads the 26 per-persona
    files instead, so the same experiment can run at any depth -- which matters because a persona
    direction lives in the residual stream *at one layer*: projecting layer-40 activations onto
    layer-20 directions is not a weaker measurement, it is a meaningless one.
    """
    path = Path(path)
    files = {p.stem: p for p in path.glob("*.pt")}

    expected = set(DECEPTION_PERSONAS + HONESTY_PERSONAS) | set(
        SYCOPHANCY_PERSONAS + ANTI_SYCOPHANCY_PERSONAS
    ) | {"default"}
    missing = sorted(expected - set(files))
    if missing:
        raise FileNotFoundError(f"persona vectors missing from {path}: {missing}")

    names = sorted(files)  # matches the single-file bank's ordering
    vectors: dict[str, np.ndarray] = {}
    n_rollouts: list[int] = []
    for name in names:
        payload = torch.load(files[name], weights_only=False)
        stored = payload["vector"]
        # a file may hold only the layers it was read at, listed in `layers`; the full
        # extractions hold all 80 in order, with or without the list
        layers = [int(x) for x in payload.get("layers", range(stored.shape[0]))]
        if len(layers) != stored.shape[0]:
            raise ValueError(f"{name}: {len(layers)} layers listed for {stored.shape[0]} rows")
        if layer not in layers:
            raise IndexError(f"{name}: layer {layer} outside the {len(layers)} layers it holds "
                             f"({layers[0]}..{layers[-1]})")
        vectors[name] = stored[layers.index(layer)].double().numpy()
        n_rollouts.append(int(payload["n"]))

    return PersonaBank(
        personas=np.stack([vectors[n] for n in names]),
        names=names,
        axes=_axes_from_groups(vectors),
        n_rollouts=n_rollouts,
        layer=layer,
        # not recorded in the per-persona files; same run as the single-layer bank
        model="meta-llama/Llama-3.3-70B-Instruct",
    )


def recover_group(
    bank: PersonaBank,
    positive: tuple[str, ...],
    axis_name: str,
    size: int,
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Re-derive the negative end of a contrast from its axis.

    Each axis is `mean(positive) - mean(negative)`, so `mean(negative)` is known exactly and
    exactly one subset of the remaining personas averages to it. This exists because the bank's
    `role_groups` field is unusable; it verifies the hard-coded lists still match a given bank,
    and is not on any hot path.
    """
    target = bank.vectors_for(positive).mean(0) - bank.axes[axis_name]
    skip = set(positive) | set(exclude) | {"default"}
    pool = [n for n in bank.names if n not in skip]

    best_err, best = np.inf, ()
    for combo in itertools.combinations(pool, size):
        err = float(np.linalg.norm(bank.vectors_for(combo).mean(0) - target))
        if err < best_err:
            best_err, best = err, combo
    logger.debug(f"recovered {axis_name} negative group at residual {best_err:.2e}: {best}")
    return tuple(sorted(best))


def recover_honesty_group(bank: PersonaBank, size: int = 7) -> tuple[str, ...]:
    """The honesty group is the negative end of the deception axis."""
    return recover_group(bank, DECEPTION_PERSONAS, "deception", size)


def recover_anti_sycophancy_group(bank: PersonaBank, size: int = 5) -> tuple[str, ...]:
    """The anti-sycophancy group is the negative end of the sycophancy axis.

    Deception and honesty are excluded from the search pool: they are already accounted for, and
    leaving them in would let the search reach a subset that happens to average correctly without
    being the real group.
    """
    return recover_group(
        bank,
        SYCOPHANCY_PERSONAS,
        "sycophancy",
        size,
        exclude=DECEPTION_PERSONAS + HONESTY_PERSONAS,
    )


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
    axis_name: str = "deception",
) -> PersonaGeometry:
    """Centre the persona set and fit its top-k principal components (Eq. 2-3).

    P is the union of `deception` and `honesty` -- 15 personas by default, not the full bank.
    A PC's sign is arbitrary, so each is oriented to point along `axis_name`; without this,
    "higher score = more of the behaviour" would hold or not at random per run.

    `axis_name` must match the groups being fitted. Fitting the sycophancy personas while leaving
    it at "deception" silently gives the geometry the wrong contrast, and `axis_score` then
    duplicates the deception arm instead of measuring anything new.
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
    axis = bank.axis_unit(axis_name)

    # orient each PC along the contrast axis so score direction is reproducible
    for i, comp in enumerate(components):
        if float(comp @ axis) < 0:
            components[i] = -comp

    group_of = {n: "positive" for n in deception} | {n: "negative" for n in honesty}
    return PersonaGeometry(
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_[:k].copy(),
        axis=axis,
        persona_names=names,
        group_of=group_of,
        mean=mean,
    )


def orthogonalize(components: np.ndarray, against: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Remove a direction from a PC basis, returning an orthonormal basis of what remains.

    Used to ask whether one persona basis helps on another behaviour's data *beyond* the
    component the two share -- the deception and sycophancy axes sit at |cos| = 0.54, so a raw
    comparison between them measures mostly their overlap.

    Rank can drop: if `against` lies inside the span of `components`, the result has one fewer
    row. Callers should report the returned row count rather than assume it equals the input's,
    since a 2-d basis is not a like-for-like comparison against a 3-d one.
    """
    unit = against / np.linalg.norm(against)
    residual = components - np.outer(components @ unit, unit)

    basis: list[np.ndarray] = []
    for row in residual:
        vector = row.copy()
        for kept in basis:  # Gram-Schmidt against what survived so far
            vector = vector - (vector @ kept) * kept
        norm = float(np.linalg.norm(vector))
        if norm > tol:
            basis.append(vector / norm)

    if not basis:
        raise ValueError("nothing left after removing `against` from the basis")
    return np.stack(basis)
