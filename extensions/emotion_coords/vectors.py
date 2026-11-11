"""Emotion vectors from story activations (Anthropic's recipe), the topic basis and the shuffled controls.

    v_e  = mean of emotion e's story activations (positions 50 onward, one story per chosen topic)
    v~_e = v_e - mean over emotions of v_e'
    v^_e = v~_e with the neutral dialogues' top principal directions (together 50% of their variance) removed
    emotion PCs = PCA of the v^_e

The topic basis is the same recipe with the roles swapped: average over all emotions per topic, centre across
topics, denoise, PCA. In the fully crossed design it carries story content with emotion balanced out, and it
shares the generator, the pipeline and the tokens with the emotion basis.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from extensions.emotion_coords import config as C
from extensions.emotion_coords import linalg as la


def _read_info(name: str | None) -> dict:
    """A reader folder's `READ.json` (its rows file and layers), written when it differs from the defaults, as for
    the band reads (`stories_band` holds `stories` rows at other layers)."""
    marker = C.DATA / "acts" / name / "READ.json" if name else None
    info = json.loads(marker.read_text()) if marker is not None and marker.exists() else {}
    return {"rows": info.get("rows", name), "layers": info.get("layers", list(C.load()["layers"]))}


def _layer_index(layer: int, name: str | None = None) -> int:
    return list(_read_info(name)["layers"]).index(layer)


def load_reads(name: str, layer_index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reader output folder: (row ids, content (n, groups, layers, H), points (n, points, layers, H)). With
    `layer_index`, only that layer is kept (the layers axis has length 1): an 8-block story folder is 2 GB whole."""
    folder = C.DATA / "acts" / name
    keep = slice(None) if layer_index is None else slice(layer_index, layer_index + 1)
    ids, content, points = [], [], []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as z:
            ids.append(z["rows"])
            content.append(np.ascontiguousarray(z["content"][:, :, keep]))  # a copy, so the whole file can go
            points.append(np.ascontiguousarray(z["points"][:, :, keep]))
    if not ids:
        raise FileNotFoundError(f"no reads in {folder}")
    order = np.argsort(np.concatenate(ids), kind="stable")
    return np.concatenate(ids)[order], np.concatenate(content)[order], np.concatenate(points)[order]


def row_meta(name: str) -> pd.DataFrame:
    """The row file's metadata (everything but the token ids and spans), indexed by row id."""
    rows = []
    with open(C.DATA / "rows" / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: v for k, v in r.items() if k not in ("ids", "groups", "points")})
    return pd.DataFrame(rows).set_index("row_id")


@dataclass
class Reads:
    """Group-0 span means of one reader file at one layer, with the row metadata, non-finite rows dropped."""

    x: np.ndarray
    meta: pd.DataFrame

    @classmethod
    def of(cls, name: str, layer: int) -> "Reads":
        ids, content, _ = load_reads(name, _layer_index(layer, name))
        x = content[:, 0, 0].astype(np.float64)
        ok = np.isfinite(x).all(1)
        meta = row_meta(_read_info(name)["rows"]).loc[ids[ok]]
        return cls(x[ok], meta.reset_index())


@dataclass
class Basis:
    """Centred, denoised class vectors and their principal directions."""

    name: str
    labels: list[str]
    vectors: np.ndarray  # (classes, H): centred and denoised (the named directions)
    components: np.ndarray  # (r, H) orthonormal PCs
    explained: np.ndarray  # (r,)
    neutral_k: int

    def unit(self, label: str) -> np.ndarray:
        return la.unit(self.vectors[self.labels.index(label)])


def neutral_basis(neutral: Reads, share: float | None = None) -> np.ndarray:
    return la.variance_basis(neutral.x, share if share is not None else C.load()["stories"]["neutral_variance"])


def build(x: np.ndarray, labels: np.ndarray, denoise: np.ndarray, name: str) -> Basis:
    """Mean per label, centre across labels, remove the `denoise` directions, PCA."""
    classes = sorted(set(labels.tolist()))
    means = np.stack([x[labels == c].mean(0) for c in classes])
    centred = means - means.mean(0)
    clean = la.erase_k(centred, denoise)
    components, explained = la.pca_basis(clean, center=False)
    return Basis(name, classes, clean, components, explained, len(denoise))


def emotion_basis(stories: Reads, denoise: np.ndarray, topics: list[int] | None = None,
                  ranks: tuple[int, ...] = (0,), extra: dict[str, tuple[int, ...]] | None = None) -> Basis:
    """Emotion vectors from the rank-0 story of each chosen topic; `extra` lets a named emotion also use more
    ranks (extra stories for its own vector only)."""
    m = stories.meta
    keep = m["rank"].isin(ranks)
    if topics is not None:
        keep &= m["topic_idx"].isin(topics)
    if extra:
        for emotion, more in extra.items():
            keep |= (m["emotion"] == emotion) & m["rank"].isin(more) & (topics is None or m["topic_idx"].isin(topics))
    return build(stories.x[keep.to_numpy()], m.loc[keep, "emotion"].to_numpy(), denoise, "emotion")


def topic_basis(stories: Reads, denoise: np.ndarray) -> Basis:
    """Per topic over every emotion (rank-0 stories only, so the design stays balanced)."""
    keep = (stories.meta["rank"] == 0).to_numpy()
    labels = stories.meta.loc[keep, "topic_idx"].astype(str).to_numpy()
    return build(stories.x[keep], labels, denoise, "topic")


def shuffled_bases(stories: Reads, denoise: np.ndarray, n: int = 5, seed: int = 0) -> list[Basis]:
    """Emotion labels permuted within each topic, then the same recipe."""
    keep = (stories.meta["rank"] == 0).to_numpy()
    x, meta = stories.x[keep], stories.meta[keep].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n):
        labels = meta["emotion"].to_numpy().copy()
        for _, idx in meta.groupby("topic_idx").groups.items():
            idx = np.asarray(idx)
            labels[idx] = labels[rng.permutation(idx)]
        out.append(build(x, labels, denoise, f"shuffled_{p}"))
    return out


def split_half(stories: Reads, denoise: np.ndarray, seed: int = 0,
               extra: dict[str, tuple[int, ...]] | None = None) -> pd.DataFrame:
    """Each emotion's vector from two disjoint halves of the topics (same centring and denoising within each
    half), and the cosine between them. `extra` adds an emotion's further ranks within each half."""
    topics = sorted(stories.meta["topic_idx"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(topics)
    halves = [sorted(order[: len(order) // 2]), sorted(order[len(order) // 2:])]
    a, b = (emotion_basis(stories, denoise, topics=[int(t) for t in h], extra=extra) for h in halves)
    return pd.DataFrame({"emotion": a.labels,
                         "cos": [float(la.unit(a.vectors[i]) @ la.unit(b.vectors[i])) for i in range(len(a.labels))]})


# --- sanity checks -----------------------------------------------------------------------------


def vad_table() -> pd.DataFrame:
    return pd.read_csv(C.results_dir("phase0") / "vad_coverage.csv").dropna(subset=["valence"]).set_index("emotion")


def geometry(basis: Basis) -> dict:
    """PC1/PC2 variance and their correlations with valence and arousal over the lexicon-matched emotions."""
    vad = vad_table()
    scores = basis.vectors @ basis.components[:2].T
    matched = [i for i, e in enumerate(basis.labels) if e in vad.index]
    v = vad.loc[[basis.labels[i] for i in matched]]
    r_val = stats.pearsonr(scores[matched, 0], v["valence"])[0]
    r_aro = stats.pearsonr(scores[matched, 1], v["arousal"])[0]
    top = basis.vectors[matched] @ basis.components[:5].T
    best = {f"PC{k + 1}": float(stats.pearsonr(top[:, k], v["valence"])[0]) for k in range(top.shape[1])}
    return {"pc1_var": float(basis.explained[0]), "pc2_var": float(basis.explained[1]),
            "r_pc1_valence": float(r_val), "r_pc2_arousal": float(r_aro), "n_matched": len(matched),
            "r_valence_by_pc": best}


def neighbours(basis: Basis, queries: list[str], top: int = 5) -> dict[str, list[tuple[str, float]]]:
    unit = la.unit(basis.vectors)
    sims = unit @ unit.T
    out = {}
    for q in queries:
        if q not in basis.labels:
            continue
        i = basis.labels.index(q)
        order = [j for j in np.argsort(-sims[i]) if j != i][:top]
        out[q] = [(basis.labels[j], round(float(sims[i, j]), 3)) for j in order]
    return out


def dose_response(basis: Basis, layer: int, emotions=("afraid", "calm"), reads: str = "tylenol") -> pd.DataFrame:
    """The Tylenol prompts' pre-response activations projected on unit emotion vectors, by dose."""
    ids, _, points = load_reads(reads, _layer_index(layer, reads))
    x = points[:, 0, 0].astype(np.float64)
    meta = row_meta(_read_info(reads)["rows"]).loc[ids]
    table = pd.DataFrame({"dose_mg": meta["dose_mg"].to_numpy()})
    for e in emotions:
        table[e] = x @ basis.unit(e)
    return table.sort_values("dose_mg").reset_index(drop=True)
