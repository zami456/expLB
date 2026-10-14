"""The thesis's Llama persona experiments with a random persona bank as the control.

The thesis builds D5-D9 from roles designed around deception (persona_pcs groups). A
collaborator built a control bank by the same pipeline (same 5 prompt variants, same 30 shared
questions, same judge filters): 35 roles drawn uniformly at random (seed 20261010) from the 274
Assistant Axis roles, plus the `default` anchor (Drive `random_persona_bank_2026-10-10`, copied to
`RANDOM_ROOT`). `code/rpb/splits.json`, fixed on 2026-10-10 before any rollout, partitions the
35 roles 100 times into six groups of the thesis groups' sizes (A 8, B 7, C 5, D 5, E 4, F 6);
R5-R9 are built from them exactly as D5-D9 are (R5 = A v B, R6 = C v D, R7 = A+C v B+D,
R8 = E v F, R9 = A+E v B+F; equal weight per role). Split 0 is the primary random bank;
splits 1-99 are the null. This scores D and R side by side on Llama's pooled activations (CPU
only): zero-shot AUROC, train-one-test-all transfer, with K, HP-KR's wording slices,
Gender-Secret through the fine-tune, AISI's held-out rollouts, across depth.

Pre-registered (2026-10-14, before any random-bank direction was scored).
Banks: primary = random vectors_s3 v the thesis bank (persona_pcs.EXTENDED_VECTORS_DIR);
secondary = random vectors_s23 v Llama's s23 bank (reported, not decided). Null: splits 1-99
(n = 99); split 0 is shown beside D and in the paired intervals, not part of the null.
Orientation: every R direction is scored with the sign giving AUROC >= 0.5 on RepE (not a
Liars' Bench set); D keeps its a-priori sign (checked to give RepE >= 0.5). Transfer learns its
own sign. K = the thesis K (recall_denial.Inputs.k); X + K = recall_denial.standardised_sum(X, K,
Alpaca). Layer 40 only. Test: one-sided p = (1 + #{s in 1..99: R_s >= D}) / 100; pass iff
p <= 0.025.
    T1 transfer   D9's train-one-test-all transfer (off-diagonal mean AUROC over CG, HP-C, HP-KR,
                  IT, ID, RepE) exceeds the R9 null.
    T2 zero-shot  D9's zero-shot AUROC averaged over CG, HP-C, HP-KR, IT, ID exceeds the R9 null.
    T3 with K     D9 + K's transfer exceeds the R9 + K null.
    T4 HP-KR      D9's zero-shot AUROC on HP-KR (all rows) exceeds the R9 null.
    T5 roles      D9 - D5 in T2's metric exceeds the null of R9 - R5 from the same split.
    T6 held out   D9's AUROC on AISI's held-out rollouts exceeds the R9 null.
Holm-adjusted p over T1-T6 from 9,999 further splits (the same seed rule, i = 100..10098) is
reported beside them. Everything else (other layers, contrasts and metrics, s23, balanced
accuracy, HP-KR slices, Gender-Secret, depth, the paired intervals) is descriptive.

    python deception_detection/scripts/random_bank.py check --bank s3
    python deception_detection/scripts/random_bank.py roleplay
    python deception_detection/scripts/random_bank.py score --bank s3
    python deception_detection/scripts/random_bank.py depth --bank s3
    python deception_detection/scripts/random_bank.py verdict

Outputs go to `results/xmodel/llama/random_bank/`.
"""

import json
import random
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import lb_models as LB

RANDOM_ROOT = LB.XMODEL_ROOT / "banks" / "llama-70b-3.3" / "random"
SPLITS_JSON = RANDOM_ROOT / "code" / "rpb" / "splits.json"
OUT = LB.XMODEL_RESULTS / "llama" / "random_bank"
LAYERS = (20, 36, 40, 44)
CRITERIA_LAYER = 40
N_NULL = 99
N_EXT = 9999
N_DRAWS = 1000
SEED = 0
ALPHA = 0.025
CONTRASTS = ("5", "6", "7", "8", "9")
# contrast -> (positive groups, negative groups); groups pooled role by role, A before E as
# persona_pcs._axes_from_groups concatenates them
GROUP_SPEC = {"5": (("A",), ("B",)), "6": (("C",), ("D",)), "7": (("A", "C"), ("B", "D")),
              "8": (("E",), ("F",)), "9": (("A", "E"), ("B", "F"))}
AXIS_OF = {"5": "deception", "6": "sycophancy", "7": "unified", "8": "denial", "9": "extended"}
GROUP_SIZES = {"A": 8, "B": 7, "C": 5, "D": 5, "E": 4, "F": 6}
WITH_K = ("5", "9")
LB5 = ("CG", "HP-C", "HP-KR", "IT", "ID")


def vectors_dir(bank: str) -> Path:
    return RANDOM_ROOT / "persona" / f"vectors_{bank}"


def thesis_dir(bank: str) -> Path:
    from deception_detection import persona_pcs as pcs

    return pcs.EXTENDED_VECTORS_DIR if bank == "s3" else LB.BANK_DIRS[("llama", bank)]


# --- roles, splits and contrasts --------------------------------------------------------------

def load_roles(folder: Path, layers=None) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Every role file in `folder`: names (sorted), vectors [roles, layers, hidden] float64 at
    `layers` (all stored layers by default) and kept rollouts per role. Parses payloads as
    `persona_pcs.load_bank_at_layer` does, without its published-name rule."""
    import torch

    paths = sorted(Path(folder).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no role files in {folder}")
    names, rows, counts = [], [], {}
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=True)
        vec = payload["vector"].double().numpy()
        stored = list(payload.get("layers", range(vec.shape[0])))
        if len(stored) != vec.shape[0]:
            raise ValueError(f"{p.name}: {vec.shape[0]} rows for {len(stored)} layers")
        if payload.get("role", p.stem) != p.stem:
            raise ValueError(f"{p.name}: payload role {payload.get('role')!r}")
        want = stored if layers is None else [int(x) for x in layers]
        try:
            vec = vec[[stored.index(L) for L in want]]
        except ValueError as error:
            raise IndexError(f"{p.name}: a requested layer is not stored") from error
        if not np.isfinite(vec).all():
            raise ValueError(f"{p.name}: non-finite values")
        names.append(p.stem)
        rows.append(vec)
        counts[p.stem] = int(payload.get("n", 0))
    return names, np.stack(rows), counts


def read_splits(path: Path = SPLITS_JSON) -> tuple[list[dict], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [s["groups"] for s in data["splits"]], data


def check_split(groups: dict, roles: list[str]) -> None:
    seen = []
    for g, size in GROUP_SIZES.items():
        if len(groups[g]) != size:
            raise ValueError(f"group {g}: {len(groups[g])} roles, not {size}")
        seen += list(groups[g])
    if len(set(seen)) != len(seen) or set(seen) != set(roles):
        raise ValueError("the groups are not a partition of the drawn roles")


def regenerate_split(roles: list[str], i: int) -> dict:
    """The seed rule: random.Random(1000000 + i).sample(sorted(roles), 35), dealt A..F in order."""
    draw = random.Random(1000000 + i).sample(sorted(roles), len(roles))
    out, k = {}, 0
    for g, size in GROUP_SIZES.items():
        out[g] = draw[k:k + size]
        k += size
    return out


def weights(names: list[str], groups: dict, contrast: str) -> np.ndarray:
    """Role weights of a contrast: mean(positive roles) - mean(negative roles); `default` 0."""
    pos = [r for g in GROUP_SPEC[contrast][0] for r in groups[g]]
    neg = [r for g in GROUP_SPEC[contrast][1] for r in groups[g]]
    w = np.zeros(len(names))
    index = {n: i for i, n in enumerate(names)}
    for r in pos:
        w[index[r]] += 1 / len(pos)
    for r in neg:
        w[index[r]] -= 1 / len(neg)
    return w


def contrast(vectors: dict[str, np.ndarray], groups: dict, c: str) -> np.ndarray:
    """The raw contrast as `persona_pcs._axes_from_groups` computes it (stack, then mean)."""
    pos = [r for g in GROUP_SPEC[c][0] for r in groups[g]]
    neg = [r for g in GROUP_SPEC[c][1] for r in groups[g]]
    return np.stack([vectors[r] for r in pos]).mean(0) - np.stack([vectors[r] for r in neg]).mean(0)


def thesis_groups() -> dict:
    from deception_detection import persona_pcs as pcs

    return {"A": pcs.DECEPTION_PERSONAS, "B": pcs.HONESTY_PERSONAS, "C": pcs.SYCOPHANCY_PERSONAS,
            "D": pcs.ANTI_SYCOPHANCY_PERSONAS, "E": pcs.KNOWLEDGE_DENIAL_PERSONAS,
            "F": pcs.KNOWLEDGE_CANDOUR_PERSONAS}


# --- statistics ---------------------------------------------------------------------------------

def auroc_columns(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of S (ties averaged), as roc_auc_score."""
    from scipy.stats import rankdata

    S = S.reshape(len(y), -1)
    r = rankdata(S, axis=0)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return (r[pos].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fitted_sign(x: np.ndarray, y: np.ndarray) -> int:
    """The sign `persona_ood.transfer_matrix`'s probe learns on a 1-d score (0 if it stays at 0)."""
    from deception_detection.scripts import persona_ood as P

    probe = P.make_probe()
    probe.fit(x[:, None], y)
    return int(np.sign(probe[-1].coef_.ravel()[0]))


def shortcut_sign(x: np.ndarray, y: np.ndarray) -> int:
    """The same sign without a fit: the gradient at 0 of the standardised, intercept-free loss is
    -(n1/n) times the lies' standardised mean; lbfgs stays at 0 when that is tiny."""
    z = (x - x.mean()) / x.std()
    g = (y == 1).mean() * z[y == 1].mean()
    return 0 if abs(g) <= 1e-4 else int(np.sign(g))


def transfer_1d(aurocs: dict[str, float], signs: dict[str, int], order: list[str]) -> float:
    """Off-diagonal mean of a fixed direction's transfer matrix: training set a gives its sign,
    and the cell on test set b is A_b, 1 - A_b or 0.5."""
    cells = []
    for a in order:
        for b in order:
            if a != b:
                s = signs[a]
                cells.append(0.5 if s == 0 else aurocs[b] if s > 0 else 1 - aurocs[b])
    return float(np.mean(cells))


def permutation_p(d: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= d)) / (1 + len(null)))


# --- one layer's data -----------------------------------------------------------------------------

class Layer:
    """Everything scored at one layer: each set's labels and activations, K and the control."""

    def __init__(self, layer: int, gs: bool = True, aisi: bool = True):
        from deception_detection.scripts import k_final as kf
        from deception_detection.scripts import k_transfer as ktr
        from deception_detection.scripts import recall_denial as rd

        self.layer = layer
        inp = rd.Inputs()
        pooled = inp.pooled(layer)
        self.order = [rd.SHORT[n] for n in ktr.ORDER]  # CG, HP-C, HP-KR, IT, ID, RepE
        self.sets = {rd.SHORT[n]: (pooled[n].y.astype(int), pooled[n].z) for n in ktr.ORDER}
        self.control = pooled[rd.CONTROL].z
        self.k = inp.k(layer)
        frame, slices = inp.hpkr()
        y_kr, z_kr = self.sets["HP-KR"]
        if not (frame["lie"].to_numpy() == y_kr).all():
            raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
        self.extra = {f"HP-KR {s}": (y_kr[slices[s]], z_kr[slices[s]]) for s in ("identical text", "denials")}
        if gs and layer in (20, 36, 40, 44):
            g = rd._gs_reads(layer)["adapted"]
            self.extra["Gender-Secret"] = ((g["labels"].astype(str) == "DECEPTIVE").astype(int),
                                           g["pooled"].astype(np.float64))
        if aisi and layer in kf.HELD_OUT_LAYERS:
            z, y = kf._held_out(layer)
            self.extra["AISI held out"] = (y, z)
        self.probe = inp.probe(layer)
        del pooled

    def all_sets(self):
        return {**self.sets, **self.extra}


def _scores_of(layer: Layer, W: np.ndarray) -> dict[str, np.ndarray]:
    """Scores of the unit directions in the columns of W [hidden, m] on every set and on Alpaca."""
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    out = {name: z @ W for name, (_, z) in layer.all_sets().items()}
    out["Alpaca"] = layer.control @ W
    return out


def _with_k(scores: dict[str, np.ndarray], k_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """X + K in score space: x / sd(x on Alpaca) + k / sd(k on Alpaca) (= standardised_sum)."""
    sx = scores["Alpaca"].std(0)
    sk = k_scores["Alpaca"].std()
    return {n: s / sx + k_scores[n][:, None] / sk for n, s in scores.items()}


def metrics(layer: Layer, scores: dict[str, np.ndarray], signs_fn=fitted_sign) -> pd.DataFrame:
    """Every metric of every column: AUROC per set, the LB-5 mean, balanced accuracy at 1% of
    Alpaca (LB-5 mean) and transfer."""
    from deception_detection.scripts import persona_ood as P

    m = next(iter(scores.values())).shape[1]
    au = {n: auroc_columns(y, scores[n]) for n, (y, _) in layer.all_sets().items()}
    out = {f"auroc {n}": v for n, v in au.items()}
    out["LB-5 mean"] = np.mean([au[n] for n in LB5], axis=0)
    thr = np.quantile(scores["Alpaca"], 0.99, axis=0)
    out["balacc LB-5 (1% Alpaca)"] = np.mean(
        [[P.balanced_accuracy(layer.sets[n][0], scores[n][:, j], thr[j]) for j in range(m)] for n in LB5], axis=0)
    signs = {a: [signs_fn(scores[a][:, j], layer.sets[a][0]) for j in range(m)] for a in layer.order}
    out["transfer"] = np.array([transfer_1d({b: au[b][j] for b in layer.order}, {a: signs[a][j] for a in layer.order},
                                            layer.order) for j in range(m)])
    return pd.DataFrame(out)


def _orient(scores: dict[str, np.ndarray], repe_y: np.ndarray) -> np.ndarray:
    """+1 / -1 per column: the sign giving AUROC >= 0.5 on RepE."""
    return np.where(auroc_columns(repe_y, scores["RepE"]) >= 0.5, 1.0, -1.0)


def random_directions(names, X_L, splits, contrasts=CONTRASTS) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Raw R directions [hidden, splits * contrasts] at one layer, and their (split, contrast)."""
    cols, keys = [], []
    for i, groups in enumerate(splits):
        for c in contrasts:
            cols.append(weights(names, groups, c) @ X_L)
            keys.append((i, c))
    return np.stack(cols, axis=1), keys


def thesis_directions(bank: str, layer: int) -> dict[str, np.ndarray]:
    from deception_detection import persona_pcs as pcs

    b = pcs.load_bank_at_layer(layer, thesis_dir(bank))
    return {c: b.axis_unit(AXIS_OF[c]) for c in CONTRASTS}


def score_layer(layer: Layer, bank: str, names, X_L, splits) -> pd.DataFrame:
    """Every metric for D5-D9, D5 + K, D9 + K and, per split, R5-R9, R5 + K and R9 + K."""
    D = thesis_directions(bank, layer.layer)
    k_scores = {n: s[:, 0] for n, s in _scores_of(layer, layer.k[:, None]).items()}
    Wd = np.stack([D[c] for c in CONTRASTS], axis=1)
    sd = _scores_of(layer, Wd)
    if (auroc_columns(layer.sets["RepE"][0], sd["RepE"]) < 0.5).any():
        raise RuntimeError(f"layer {layer.layer}: a thesis contrast scores RepE below 0.5")
    sdk = _with_k({n: s[:, [CONTRASTS.index(c) for c in WITH_K]] for n, s in sd.items()}, k_scores)
    rows = []
    for j, (frame, labels) in enumerate(((metrics(layer, sd), [f"D{c}" for c in CONTRASTS]),
                                         (metrics(layer, sdk), [f"D{c} + K" for c in WITH_K]))):
        for i, label in enumerate(labels):
            rows.append({"direction": label, "split": -1, **frame.iloc[i].to_dict()})
    Wr, keys = random_directions(names, X_L, splits)
    sr = _scores_of(layer, Wr)
    o = _orient(sr, layer.sets["RepE"][0])
    sr = {n: s * o for n, s in sr.items()}
    mr = metrics(layer, sr)
    kcols = [j for j, (_, c) in enumerate(keys) if c in WITH_K]
    mrk = metrics(layer, _with_k({n: s[:, kcols] for n, s in sr.items()}, k_scores))
    for j, (i, c) in enumerate(keys):
        rows.append({"direction": f"R{c}", "split": i, "orientation": o[j], **mr.iloc[j].to_dict()})
    for jj, j in enumerate(kcols):
        i, c = keys[j]
        rows.append({"direction": f"R{c} + K", "split": i, **mrk.iloc[jj].to_dict()})
    t = pd.DataFrame(rows)
    t.insert(0, "layer", layer.layer)
    t.insert(0, "bank", bank)
    return t


# --- commands -------------------------------------------------------------------------------------

def check(bank: str = "s3") -> None:
    """Integrity gates; any failure raises. Writes `check/{bank}.json`."""
    import hashlib

    import torch

    from deception_detection import persona_pcs as pcs
    from deception_detection.scripts.layer_sweep import ALL_LAYERS

    report = {}
    names, X, counts = load_roles(vectors_dir(bank))
    drawn = [n for n in names if n != "default"]
    if len(names) != 36 or "default" not in names or X.shape[1:] != (80, 8192):
        raise RuntimeError(f"{bank}: {len(names)} roles, shape {X.shape}")
    report["roles"] = len(names)
    draw = json.loads((RANDOM_ROOT / "code" / "rpb" / "draw.json").read_text(encoding="utf-8"))
    if sorted(draw["drawn"]) != sorted(drawn):
        raise RuntimeError("the role files are not draw.json's roles")
    splits, data = read_splits()
    for groups in splits:
        check_split(groups, drawn)
    for c, spec in GROUP_SPEC.items():
        defined = data["contrasts"][f"R{c}"]
        if (tuple(defined["positive"]), tuple(defined["negative"])) != spec:
            raise RuntimeError(f"R{c} in splits.json is not {spec}")
    regen = sum(regenerate_split(drawn, i) == {g: list(v) for g, v in s.items()} or
                {g: sorted(v) for g, v in regenerate_split(drawn, i).items()} == {g: sorted(v) for g, v in s.items()}
                for i, s in enumerate(splits))
    report["splits regenerated from the seed rule"] = f"{regen} of {len(splits)}"
    md5 = hashlib.md5((SPLITS_JSON).read_bytes()).hexdigest()
    # the collaborator's split-0 directions and their recorded inputs
    worst = 1.0
    vec = {n: X[i] for i, n in enumerate(names)}
    for c in CONTRASTS:
        meta = json.loads((RANDOM_ROOT / f"bank_{bank}" / f"r{c}" / f"R{c}_{bank}.json").read_text(encoding="utf-8"))
        if meta["splits_md5"] != md5:
            raise RuntimeError(f"R{c}: splits.json md5 differs")
        for role, h in meta["input_md5"].items():
            if hashlib.md5((vectors_dir(bank) / f"{role}.pt").read_bytes()).hexdigest() != h:
                raise RuntimeError(f"R{c}: {role}.pt differs from the file it was built from")
        theirs = torch.load(RANDOM_ROOT / f"bank_{bank}" / f"r{c}" / f"R{c}_{bank}.pt", weights_only=True)
        mine = contrast(vec, splits[0], c)  # [80, hidden]
        raw = theirs["raw"].double().numpy()
        cos = np.sum(mine * raw, 1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(raw, axis=1)
        worst = min(worst, float(cos.min()))
        if sorted(meta["positive"]) != sorted(r for g in GROUP_SPEC[c][0] for r in splits[0][g]):
            raise RuntimeError(f"R{c}: the positive roles differ")
    if worst < 0.9999:
        raise RuntimeError(f"split 0 does not reproduce the collaborator's R5-R9: cosine {worst}")
    report["split 0 v collaborator, smallest cosine over 5 contrasts x 80 layers"] = worst
    null = pd.read_csv(RANDOM_ROOT / f"bank_{bank}" / "r_null" / f"R_null_{bank}.csv")
    gaps = []
    for (i, c), part in null.groupby(["split", "contrast"]):
        mine = np.linalg.norm(contrast(vec, splits[int(i)], c[1:]), axis=1)
        gaps.append(np.abs(mine[part["layer"].to_numpy()] - part["norm"].to_numpy()).max())
    report["r_null norms, largest gap"] = float(max(gaps))
    if max(gaps) > 2e-5:
        raise RuntimeError(f"r_null norms do not reproduce: {max(gaps)}")
    # the builder on the thesis bank reproduces persona_pcs' axes
    tn, TX, _ = load_roles(thesis_dir(bank), ALL_LAYERS)
    tv = {n: TX[i] for i, n in enumerate(tn)}
    tg = thesis_groups()
    for li, L in enumerate(ALL_LAYERS):
        axes = pcs.load_bank_at_layer(L, thesis_dir(bank)).axes
        for c in CONTRASTS:
            mine = contrast({n: v[li] for n, v in tv.items()}, tg, c)
            if not np.allclose(mine, axes[AXIS_OF[c]], rtol=0, atol=1e-12):
                raise RuntimeError(f"layer {L}: the builder does not reproduce {AXIS_OF[c]}")
    report["builder reproduces persona_pcs D5-D9 at the 16 sweep layers"] = True
    report["kept rollouts per role (random bank)"] = counts
    (OUT / "check").mkdir(parents=True, exist_ok=True)
    (OUT / "check" / f"{bank}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{bank}: every check passed: {json.dumps({k: v for k, v in report.items() if k != 'kept rollouts per role (random bank)'})}")


def roleplay() -> None:
    """Judge scores of the random bank's rollouts: counts of 0-3 and the share fully in role."""
    rows = []
    for p in sorted((RANDOM_ROOT / "persona" / "scores").glob("*.json")):
        s = list(json.loads(p.read_text(encoding="utf-8")).values())
        rows.append({"role": p.stem, "judged": len(s), **{f"score {k}": sum(v == k for v in s) for k in range(4)}})
    t = pd.DataFrame(rows)
    t["share fully in role"] = t["score 3"] / t["judged"]
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "roleplay.csv", index=False)
    logger.info(f"random bank: {t['score 3'].sum()} of {t['judged'].sum()} fully in role "
                f"({t['score 3'].sum() / t['judged'].sum():.3f}); the thesis bank's Llama share is 0.686")


def _reproduce(t: pd.DataFrame, bank: str, layer: int) -> list[dict]:
    """D's numbers against the stored thesis results."""
    folder = (LB.REPO_ROOT / "results" / f"persona_ood_ext_L{layer}") if bank == "s3" else \
        (LB.XMODEL_RESULTS / "llama" / f"persona_ood_s23_L{layer}")
    zs = pd.read_csv(folder / "zero_shot.csv").set_index("dataset")
    summary = pd.read_csv(folder / "summary.csv").set_index("features")
    col = {"5": "auroc_axis", "6": "auroc_syc_axis", "7": "auroc_uni_axis", "8": "auroc_denial_axis", "9": "auroc_ext_axis"}
    feat = {"5": "axis1", "6": "syc_axis1", "7": "uni_axis1", "8": "denial_axis1", "9": "ext_axis1"}
    from deception_detection.scripts import recall_denial as rd

    out = []
    d = t[t["split"] == -1].set_index("direction")
    for c in CONTRASTS:
        for name in zs.index:
            short = rd.SHORT.get(name)
            if short in LB5 or short == "RepE":
                out.append({"bank": bank, "layer": layer, "what": f"D{c} zero-shot {short}",
                            "ours": d.loc[f"D{c}", f"auroc {short}"], "stored": zs.loc[name, col[c]]})
        out.append({"bank": bank, "layer": layer, "what": f"D{c} transfer", "ours": d.loc[f"D{c}", "transfer"],
                    "stored": summary.loc[feat[c], "off_diagonal_mean"]})
    if bank == "s3":
        ninek = pd.read_csv(LB.REPO_ROOT / "results" / "recall_denial" / "transfer" / "depth_ninek.csv").set_index("layer")
        if layer in ninek.index:
            out.append({"bank": bank, "layer": layer, "what": "D9 + K transfer", "ours": d.loc["D9 + K", "transfer"],
                        "stored": ninek.loc[layer, "D9 + K: transfer"]})
    for r in out:
        r["gap"] = abs(r["ours"] - r["stored"])
    return out


def _summarise(t: pd.DataFrame) -> pd.DataFrame:
    """D, split 0 and the null (splits 1-99) for every direction and metric."""
    rows = []
    metric_cols = [c for c in t.columns if c not in ("bank", "layer", "direction", "split", "orientation")]
    for (bank, layer), part in t.groupby(["bank", "layer"]):
        d = part[part["split"] == -1].set_index("direction")
        r = part[part["split"] >= 0]
        for name in d.index:
            rname = name.replace("D", "R", 1)
            rr = r[r["direction"] == rname].set_index("split")
            for m in metric_cols:
                if m not in rr or rr[m].isna().all():
                    continue
                null = rr.loc[rr.index >= 1, m].to_numpy()
                dv = d.loc[name, m]
                rows.append({"bank": bank, "layer": layer, "direction": name, "metric": m, "D": dv,
                             "R split 0": rr.loc[0, m], "null median": np.median(null),
                             "null 2.5%": np.quantile(null, 0.025), "null 97.5%": np.quantile(null, 0.975),
                             "null max": null.max(), "D percentile": 100 * float(np.mean(null < dv)),
                             "p": permutation_p(dv, null)})
        # difference in differences, split by split
        for m in ("LB-5 mean", "transfer", "auroc HP-KR"):
            dd = d.loc["D9", m] - d.loc["D5", m]
            r9 = r[r["direction"] == "R9"].set_index("split")[m]
            r5 = r[r["direction"] == "R5"].set_index("split")[m]
            null = (r9 - r5).loc[lambda s: s.index >= 1].to_numpy()
            rows.append({"bank": bank, "layer": layer, "direction": "D9 - D5", "metric": m, "D": dd,
                         "R split 0": r9.loc[0] - r5.loc[0], "null median": np.median(null),
                         "null 2.5%": np.quantile(null, 0.025), "null 97.5%": np.quantile(null, 0.975),
                         "null max": null.max(), "D percentile": 100 * float(np.mean(null < dd)),
                         "p": permutation_p(dd, null)})
    return pd.DataFrame(rows)


def _paired(layer: Layer, bank: str, names, X_L, split0: dict, n_draws: int, rng) -> list[dict]:
    """Paired 95% intervals of D - R(split 0): stratified row resamples shared by every scorer;
    transfer with each probe's fitted signs held fixed, as `transfer_ci.py`."""
    D = thesis_directions(bank, layer.layer)
    cols, labels = [], []
    for c in CONTRASTS:
        cols += [D[c], weights(names, split0, c) @ X_L]
        labels += [f"D{c}", f"R{c}"]
    sc = _scores_of(layer, np.stack(cols, axis=1))
    o = _orient(sc, layer.sets["RepE"][0])
    o[0::2] = 1.0
    sc = {n: s * o for n, s in sc.items()}
    k_scores = {n: s[:, 0] for n, s in _scores_of(layer, layer.k[:, None]).items()}
    pick = [labels.index(x) for x in ("D5", "R5", "D9", "R9")]
    sk = _with_k({n: s[:, pick] for n, s in sc.items()}, k_scores)
    sc = {n: np.concatenate([sc[n], sk[n]], axis=1) for n in sc}
    labels += ["D5 + K", "R5 + K", "D9 + K", "R9 + K"]
    m = len(labels)
    signs = {a: np.array([fitted_sign(sc[a][:, j], layer.sets[a][0]) for j in range(m)]) for a in layer.order}
    sets = layer.all_sets()
    strata = {n: (np.flatnonzero(y == 1), np.flatnonzero(y == 0)) for n, (y, _) in sets.items()}

    def summary(idx):
        au = {n: auroc_columns(sets[n][0][idx[n]], sc[n][idx[n]]) for n in sets}
        out = {f"auroc {n}": au[n] for n in sets}
        out["LB-5 mean"] = np.mean([au[n] for n in LB5], axis=0)
        cells = []
        for a in layer.order:
            for b in layer.order:
                if a != b:
                    s = signs[a]
                    cells.append(np.where(s == 0, 0.5, np.where(s > 0, au[b], 1 - au[b])))
        out["transfer"] = np.mean(cells, axis=0)
        return out

    point = summary({n: np.arange(len(sets[n][0])) for n in sets})
    draws = {k: np.empty((n_draws, m)) for k in point}
    for b in range(n_draws):
        idx = {n: np.concatenate([rng.choice(p, len(p)), rng.choice(q, len(q))]) for n, (p, q) in strata.items()}
        for k, v in summary(idx).items():
            draws[k][b] = v
    rows = []
    for a, r in (("D5", "R5"), ("D6", "R6"), ("D7", "R7"), ("D8", "R8"), ("D9", "R9"), ("D5 + K", "R5 + K"),
                 ("D9 + K", "R9 + K")):
        ia, ir = labels.index(a), labels.index(r)
        for k in point:
            diff = draws[k][:, ia] - draws[k][:, ir]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            rows.append({"bank": bank, "layer": layer.layer, "metric": k, "a": a, "b": r + " (split 0)",
                         "a_value": point[k][ia], "b_value": point[k][ir], "delta": point[k][ia] - point[k][ir],
                         "lo": float(lo), "hi": float(hi)})
    return rows


def score(bank: str = "s3", layers=LAYERS, n_draws: int = N_DRAWS, seed: int = SEED) -> None:
    """Every metric for D and for R at splits 0-99, the null summaries and the paired intervals,
    at the four thesis layers. Writes `{bank}/values.csv`, `null_summary.csv`, `paired_ci.csv`
    and `reproduction.csv`."""
    names, X, _ = load_roles(vectors_dir(bank), layers)
    splits, _ = read_splits()
    rng = np.random.default_rng(seed)
    out = OUT / bank
    out.mkdir(parents=True, exist_ok=True)
    tables, repro, paired = [], [], []
    for li, L in enumerate([int(x) for x in layers]):
        started = time.time()
        layer = Layer(L)
        t = score_layer(layer, bank, names, X[:, li], splits)
        repro += _reproduce(t, bank, L)
        paired += _paired(layer, bank, names, X[:, li], splits[0], n_draws, rng)
        tables.append(t)
        logger.info(f"{bank} layer {L} scored in {time.time() - started:.0f} s")
        del layer
    t = pd.concat(tables, ignore_index=True)
    t.round(6).to_csv(out / "values.csv", index=False)
    rep = pd.DataFrame(repro)
    rep.to_csv(out / "reproduction.csv", index=False)
    worst = rep.groupby(rep["what"].str.contains("transfer"))["gap"].max()
    logger.info(f"reproduction of the stored D numbers: largest gap zero-shot {worst.get(False, 0):.2e}, "
                f"transfer {worst.get(True, 0):.2e}")
    if worst.get(False, 0) > 1e-6 or worst.get(True, 0) > 1e-3:
        raise AssertionError(f"the D numbers do not reproduce the stored thesis results:\n{rep.sort_values('gap').tail()}")
    s = _summarise(t)
    s.round(6).to_csv(out / "null_summary.csv", index=False)
    pd.DataFrame(paired).round(6).to_csv(out / "paired_ci.csv", index=False)
    pd.set_option("display.width", 250)
    head = s[(s["layer"] == CRITERIA_LAYER) & s["metric"].isin(["LB-5 mean", "transfer", "auroc HP-KR", "auroc AISI held out"])]
    logger.info(f"{bank} at layer {CRITERIA_LAYER}:\n{head.round(3).to_string(index=False)}")


def extension(bank: str = "s3", n_ext: int = N_EXT, layer: int = CRITERIA_LAYER) -> None:
    """T1-T6's metrics at `layer` over 9,999 further splits (seed rule, i = 100..), for Holm.
    Transfer uses `shortcut_sign`, checked against the fitted signs on splits 0-99."""
    names, X, _ = load_roles(vectors_dir(bank), [layer])
    drawn = [n for n in names if n != "default"]
    splits = [regenerate_split(drawn, 100 + i) for i in range(n_ext)]
    L = Layer(layer, gs=False)
    k_scores = {n: s[:, 0] for n, s in _scores_of(L, L.k[:, None]).items()}
    base, _ = read_splits()
    rows = []
    for chunk in range(0, n_ext, 500):
        part = splits[chunk:chunk + 500]
        W, keys = random_directions(names, X[:, 0], part, contrasts=("5", "9"))
        sr = _scores_of(L, W)
        o = _orient(sr, L.sets["RepE"][0])
        sr = {n: s * o for n, s in sr.items()}
        m = metrics(L, sr, signs_fn=shortcut_sign)
        k9 = [j for j, (_, c) in enumerate(keys) if c == "9"]
        mk = metrics(L, _with_k({n: s[:, k9] for n, s in sr.items()}, k_scores), signs_fn=shortcut_sign)
        for j, (i, c) in enumerate(keys):
            rows.append({"split": 100 + chunk + i, "direction": f"R{c}", **m.iloc[j][["LB-5 mean", "transfer",
                         "auroc HP-KR", "auroc AISI held out"]].to_dict()})
        for jj, j in enumerate(k9):
            rows.append({"split": 100 + chunk + keys[j][0], "direction": "R9 + K", "transfer": mk.iloc[jj]["transfer"]})
        logger.info(f"extension: {min(chunk + 500, n_ext)} of {n_ext} splits")
    # the shortcut against the fitted signs on the pre-registered splits
    Wb, kb = random_directions(names, X[:, 0], base, contrasts=("9",))
    sb = _scores_of(L, Wb)
    sb = {n: s * _orient(sb, L.sets["RepE"][0]) for n, s in sb.items()}
    a, b = metrics(L, sb)["transfer"], metrics(L, sb, signs_fn=shortcut_sign)["transfer"]
    gap = float(np.abs(a - b).max())
    logger.info(f"shortcut v fitted signs on splits 0-99: largest transfer gap {gap:.2e}")
    if gap > 1e-9:
        raise AssertionError("the shortcut sign does not reproduce the fitted transfer")
    pd.DataFrame(rows).round(6).to_csv(OUT / bank / f"extension_L{layer}.csv", index=False)


def depth(bank: str = "s3") -> None:
    """D9 and D9 + K against the R9 and R9 + K nulls (splits 1-99) at all 16 sweep layers:
    the LB-5 mean and transfer. Writes `{bank}/depth.csv` and `depth.png`."""
    from deception_detection.scripts.layer_sweep import ALL_LAYERS

    names, X, _ = load_roles(vectors_dir(bank), ALL_LAYERS)
    splits, _ = read_splits()
    rows = []
    for li, L in enumerate(ALL_LAYERS):
        layer = Layer(L, gs=False, aisi=False)
        D = thesis_directions(bank, L)
        k_scores = {n: s[:, 0] for n, s in _scores_of(layer, layer.k[:, None]).items()}
        sd = _scores_of(layer, D["9"][:, None])
        md = metrics(layer, sd)
        mdk = metrics(layer, _with_k(sd, k_scores))
        W, _ = random_directions(names, X[:, li], splits, contrasts=("9",))
        sr = _scores_of(layer, W)
        sr = {n: s * _orient(sr, layer.sets["RepE"][0]) for n, s in sr.items()}
        mr, mrk = metrics(layer, sr), metrics(layer, _with_k(sr, k_scores))
        for name, dm, rm in (("D9", md, mr), ("D9 + K", mdk, mrk)):
            for m in ("LB-5 mean", "transfer"):
                null = rm[m].to_numpy()[1:]
                rows.append({"bank": bank, "layer": L, "direction": name, "metric": m, "D": dm[m].iloc[0],
                             "R split 0": rm[m].iloc[0], "null median": np.median(null),
                             "null 2.5%": np.quantile(null, 0.025), "null 97.5%": np.quantile(null, 0.975),
                             "p": permutation_p(dm[m].iloc[0], null)})
        logger.info(f"{bank} depth: layer {L}")
        del layer
    t = pd.DataFrame(rows)
    t.round(6).to_csv(OUT / bank / "depth.csv", index=False)
    _depth_figure(t, OUT / bank / "depth.png")


def _depth_figure(t: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, m in zip(axes, ("LB-5 mean", "transfer")):
        for name, colour in (("D9", "#2a78d6"), ("D9 + K", "#3f3e3b")):
            p = t[(t["direction"] == name) & (t["metric"] == m)].sort_values("layer")
            ax.fill_between(p["layer"], p["null 2.5%"], p["null 97.5%"], color=colour, alpha=0.15,
                            label=f"R{name[1:]} null, 95%")
            ax.plot(p["layer"], p["D"], color=colour, lw=1.6, label=name)
        ax.axhline(0.5, color="#999", lw=0.6, ls=":")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("layer", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("AUROC", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verdict(bank: str = "s3") -> None:
    """T1-T6 at layer 40 from the 99-split null, Holm from the extension."""
    s = pd.read_csv(OUT / bank / "null_summary.csv")
    s = s[s["layer"] == CRITERIA_LAYER]
    ext_path = OUT / bank / f"extension_L{CRITERIA_LAYER}.csv"
    ext = pd.read_csv(ext_path) if ext_path.exists() else None
    tests = (("T1 transfer", "D9", "transfer", "R9"), ("T2 zero-shot", "D9", "LB-5 mean", "R9"),
             ("T3 with K", "D9 + K", "transfer", "R9 + K"), ("T4 HP-KR", "D9", "auroc HP-KR", "R9"),
             ("T5 roles", "D9 - D5", "LB-5 mean", None), ("T6 held out", "D9", "auroc AISI held out", "R9"))
    rows = []
    for label, direction, metric, rname in tests:
        r = s[(s["direction"] == direction) & (s["metric"] == metric)].iloc[0]
        p_ext = np.nan
        if ext is not None:
            if rname is None:
                e = ext.pivot_table(index="split", columns="direction", values=metric)
                null = (e["R9"] - e["R5"]).to_numpy()
            else:
                null = ext.loc[ext["direction"] == rname, metric].dropna().to_numpy()
            p_ext = permutation_p(r["D"], null)
        rows.append({"test": label, "D": r["D"], "R split 0": r["R split 0"], "null median": r["null median"],
                     "null 97.5%": r["null 97.5%"], "p (99 splits)": r["p"], "p (9,999 splits)": p_ext,
                     "verdict": "pass" if r["p"] <= ALPHA else "fail"})
    t = pd.DataFrame(rows)
    if t["p (9,999 splits)"].notna().all():
        order = np.argsort(t["p (9,999 splits)"].to_numpy())
        adj = np.empty(len(t))
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(t) - rank) * t["p (9,999 splits)"].iloc[i]))
            adj[i] = running
        t["Holm p (9,999 splits)"] = adj
    t.round(6).to_csv(OUT / bank / "verdict.csv", index=False)
    lines = [f"{r['test']}: D {r['D']:.3f}, null median {r['null median']:.3f}, 97.5% {r['null 97.5%']:.3f}, "
             f"p {r['p (99 splits)']:.3f}: {r['verdict']}" for _, r in t.iterrows()]
    (OUT / bank / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"{bank} verdicts at layer {CRITERIA_LAYER}:\n" + "\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"check": check, "roleplay": roleplay, "score": score, "extension": extension, "depth": depth,
               "verdict": verdict})
