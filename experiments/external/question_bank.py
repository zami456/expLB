"""The question extension of the persona bank, Llama side: banks from the per-question store, scored on
the thesis's development data (Liars' Bench) with the random-bank machinery; the pre-registered tests;
the spectrum analyses with questions as domains; the single confirmation read.

Pre-registration: experiments/external/questions_150/PREREGISTRATION.md. Plan:
C:/Users/zamig/.claude/plans/recursive-munching-glade.md.

The store: <run>/<store>/<role>/q###.pt = {"vector": (K,H), "layers", "n", ...} (or q###.empty), one mean
vector per (role, question) at a few layers, written by 2_activations_ndif.py --per_question. Every bank
variant is a weighted average over questions: plain = sum_q n_rq v_rq / sum_q n_rq (what the pipeline's
one-vector-per-role mean is), balanced = equal weight per question with n_rq >= 1.

    python experiments/external/question_bank.py phase0                      # free dry run (base.npz)
    python experiments/external/question_bank.py store --name vectors_pq_s3 --tag s3q30
    python experiments/external/question_bank.py gate0 --tag s3q30            # GATE-0 + PRED-0
    python experiments/external/question_bank.py store                        # the 150-question store
    python experiments/external/question_bank.py score
    python experiments/external/question_bank.py null
    python experiments/external/question_bank.py paired
    python experiments/external/question_bank.py spectrum
    python experiments/external/question_bank.py verdict
    python experiments/external/question_bank.py materialise --variant B150
    python experiments/external/question_bank.py confirm --bank <selected>   # once

Development commands build `random_bank.Layer(L, gs=False, aisi=False)`: AISI held-out and Gender-Secret
are never loaded. `confirm` is the only command that loads them, and it refuses to run twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from experiments.external import random_bank as RB
from experiments.knowledge import recall_denial as rd
from deception_detection.utils import REPO_ROOT

QDIR = REPO_ROOT / "experiments" / "external" / "questions_150"
QFILE = QDIR / "kd_questions_shared_v2.jsonl"
RUN = Path(r"E:\qbank\banks\Llama-3.3-70B-Instruct\persona")
STORE = "vectors_pq_s3_q150"
OUT = REPO_ROOT / "results" / "question_bank" / "llama"
FREEZE = Path(r"E:\qbank\logs\freeze.txt")
LAYERS = (20, 36, 40, 44)
CRITERIA_LAYER = 40
N_NULL = 200
N_DRAWS = 1000
SEED = 0
AXES = {"D5": "deception", "D6": "sycophancy", "D7": "unified", "D8": "denial", "D9": "extended"}
WITH_K = ("D5", "D9")
HOLM_TESTS = ("T1", "T2", "T3", "T4")
EQUIV = 0.03   # H2 equivalence margin on the LB-5 mean
TIE = 0.005    # bank-selection tie margin (larger bank wins)
GROUPS = {"deception": pcs.DECEPTION_PERSONAS, "honesty": pcs.HONESTY_PERSONAS,
          "denial": pcs.KNOWLEDGE_DENIAL_PERSONAS, "candour": pcs.KNOWLEDGE_CANDOUR_PERSONAS}


# --- questions -----------------------------------------------------------------------------------

def questions() -> list[dict]:
    rows = [json.loads(l) for l in QFILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if [r["id"] for r in rows] != list(range(len(rows))):
        raise ValueError("question ids are not the line order")
    return rows


def families(rows: list[dict]) -> dict[str, list[int]]:
    """The ten question families: three original, six grid types, the monotone set."""
    fam: dict[str, list[int]] = {}
    for r in rows:
        if r["set"] == "orig":
            fam.setdefault(r["domain"], []).append(r["id"])
        elif r["set"] == "mono":
            fam.setdefault("mono", []).append(r["id"])
        else:
            fam.setdefault("imt_" + r["grid"]["imt"], []).append(r["id"])
    return fam


def margins(rows: list[dict], key: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        if r["set"] == "div":
            out.setdefault(f"{key}_{r['grid'][key]}", []).append(r["id"])
    return out


def variants(rows: list[dict]) -> dict[str, tuple[list[int], str]]:
    """name -> (question ids, weighting). The pre-registered banks, the families and the grid margins."""
    orig = [r["id"] for r in rows if r["set"] == "orig"]
    mono = [r["id"] for r in rows if r["set"] == "mono"]
    div = [r["id"] for r in rows if r["set"] == "div"]
    every = orig + mono + div
    out = {"B30": (orig, "plain"), "B30-bal": (orig, "balanced"), "B30orig": (orig, "plain"),
           "B30mono": (mono, "plain"), "B30+30mono": (orig + mono, "plain"), "B30+90div": (orig + div, "plain"),
           "B90div": (div, "plain"), "B120new": (mono + div, "plain"), "B150": (every, "plain"),
           "B150-bal": (every, "balanced")}
    for name, q in families(rows).items():
        out[f"fam_{name}"] = (q, "plain")
    for key in ("deceiver", "domain", "format"):
        for name, q in margins(rows, key).items():
            out[name] = (q, "plain")
    return out


def qfile_sha() -> str:
    return hashlib.sha256(QFILE.read_bytes()).hexdigest()


def freeze_check() -> None:
    sha = qfile_sha()
    if not FREEZE.exists() or sha.lower() not in FREEZE.read_text(encoding="utf-8").lower():
        raise SystemExit(f"{QFILE.name} sha256 {sha} is not recorded in {FREEZE}; freeze before scoring")


# --- the store -----------------------------------------------------------------------------------

def store(run: str = str(RUN), name: str = STORE, tag: str = "s3", layers=LAYERS, out: Path = OUT) -> None:
    """Read <run>/<name>/<role>/q###.pt|.empty into one npz per layer:
    vectors [roles, questions, hidden] float32, n [roles, questions] int, roles, question_ids."""
    import torch

    out = Path(out)
    root = Path(run) / name
    roles = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(roles) != 36:
        raise FileNotFoundError(f"{root}: {len(roles)} role folders, expected 36")
    qids = sorted({int(f.stem[1:]) for r in roles for f in (root / r).glob("q*.pt")}
                  | {int(f.stem[1:]) for r in roles for f in (root / r).glob("q*.empty")})
    first = next(f for r in roles for f in sorted((root / r).glob("q*.pt")))
    d0 = torch.load(first, weights_only=True)
    stored_layers = [int(x) for x in d0["layers"]]
    want = [int(L) for L in layers]
    missing = [L for L in want if L not in stored_layers]
    if missing:
        raise IndexError(f"{root}: layers {missing} not in the store ({stored_layers})")
    hidden = int(d0["vector"].shape[1])
    out.mkdir(parents=True, exist_ok=True)
    n = np.zeros((len(roles), len(qids)), dtype=np.int64)
    vectors = {L: np.zeros((len(roles), len(qids), hidden), dtype=np.float32) for L in want}
    empties = 0
    for i, role in enumerate(roles):
        for j, q in enumerate(qids):
            p = root / role / f"q{q:03d}.pt"
            if p.exists():
                d = torch.load(p, weights_only=True)
                sl = [int(x) for x in d["layers"]]
                v = d["vector"]
                for L in want:
                    vectors[L][i, j] = v[sl.index(L)].float().numpy()
                n[i, j] = int(d["n"])
            elif (root / role / f"q{q:03d}.empty").exists():
                empties += 1
            else:
                raise FileNotFoundError(f"{root / role}: question {q} has neither a vector nor an .empty marker")
    for L in want:
        np.savez(out / f"store_{tag}_L{L}.npz", vectors=vectors[L], n=n, roles=np.array(roles),
                 question_ids=np.array(qids), layer=L)
    logger.info(f"store {name}: {len(roles)} roles x {len(qids)} questions, {empties} empty cells, "
                f"layers {want} -> {out}/store_{tag}_L*.npz")
    _keep_rates(Path(run), roles, qids, n, tag, out)


def _keep_rates(run: Path, roles: list[str], qids: list[int], n: np.ndarray, tag: str, out: Path = OUT) -> None:
    """Judged rows and kept share per role x family, from the responses and scores beside the store."""
    rows_q = questions() if len(qids) > 30 else None
    fam = families(rows_q) if rows_q else {"orig_disclosure": list(range(0, 10)),
                                            "orig_sycophancy": list(range(10, 20)),
                                            "orig_knowledge": list(range(20, 30))}
    pos = {q: j for j, q in enumerate(qids)}
    out_rows = []
    for i, role in enumerate(roles):
        scores = {}
        p = run / "scores" / f"{role}.json"
        if p.exists():
            scores = json.loads(p.read_text(encoding="utf-8"))
        for name, qs in fam.items():
            qs = [q for q in qs if q in pos]
            judged = sum(1 for k in scores if int(k.split("_q")[1]) in set(qs))
            kept = int(n[i, [pos[q] for q in qs]].sum())
            zero = int((n[i, [pos[q] for q in qs]] == 0).sum())
            out_rows.append({"role": role, "family": name, "questions": len(qs), "judged": judged, "kept": kept,
                             "kept_share": kept / judged if judged else np.nan, "questions_with_no_kept_row": zero,
                             "flag_n_lt_10": kept < 10})
    pd.DataFrame(out_rows).to_csv(Path(out) / f"keep_rates_{tag}.csv", index=False)


def load_store(tag: str, layer: int, out: Path = OUT):
    st = np.load(Path(out) / f"store_{tag}_L{layer}.npz")
    return st["vectors"], st["n"], [str(r) for r in st["roles"]], [int(q) for q in st["question_ids"]]


def bank_vectors(V: np.ndarray, n: np.ndarray, qpos: list[int], weighting: str) -> np.ndarray:
    """[roles, hidden] float64: the plain (n-weighted) or question-balanced mean over the questions at
    positions `qpos`. Raises if a role has no kept row in the subset."""
    sub = V[:, qpos].astype(np.float64)
    nn = n[:, qpos].astype(np.float64)
    if (nn.sum(1) == 0).any():
        raise ValueError(f"roles with no kept row in the subset: {np.flatnonzero(nn.sum(1) == 0).tolist()}")
    if weighting == "plain":
        w = nn / nn.sum(1, keepdims=True)
    elif weighting == "balanced":
        has = (nn > 0).astype(np.float64)
        w = has / has.sum(1, keepdims=True)
    else:
        raise ValueError(weighting)
    return np.einsum("rq,rqh->rh", w, sub)


def directions(role_vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """D5-D9 as unit directions, by `persona_pcs._axes_from_groups` (the thesis's definition)."""
    axes = pcs._axes_from_groups(role_vectors)
    return {d: rd._unit(axes[a]) for d, a in AXES.items()}


def banks_at(layer: int, tag: str = "s3", names=None, out: Path = OUT,
             thesis_dir: Path = pcs.EXTENDED_VECTORS_DIR) -> dict[str, dict[str, np.ndarray]]:
    """Every variant's role vectors at one layer, plus the reference all-layer bank as `B30-thesis`."""
    rows = questions()
    V, n, roles, qids = load_store(tag, layer, out)
    qpos = {q: i for i, q in enumerate(qids)}
    out = {}
    for name, (qs, w) in variants(rows).items():
        if names is not None and name not in names:
            continue
        if any(q not in qpos for q in qs):
            continue  # a 30-question store has no new questions
        try:
            vec = bank_vectors(V, n, [qpos[q] for q in qs], w)
        except ValueError as error:
            logger.warning(f"{name} at layer {layer}: {error}")
            continue
        out[name] = {r: vec[i] for i, r in enumerate(roles)}
    if names is None or "B30-thesis" in names:
        tb = pcs.load_bank_at_layer(layer, thesis_dir)
        out["B30-thesis"] = {nm: tb.personas[i] for i, nm in enumerate(tb.names)}
    return out


def materialise(variant: str, tag: str = "s3", layers=LAYERS, run: str = str(RUN), out: Path = OUT) -> Path:
    """Write <run>/banks_cpu/<variant>/<role>.pt so `persona_pcs.load_bank_at_layer` reads the bank."""
    import torch

    rows = questions()
    qs, w = variants(rows)[variant]
    out_dir = Path(run) / "banks_cpu" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    per_layer = {}
    for L in layers:
        V, n, roles, qids = load_store(tag, L, out)
        qpos = {q: i for i, q in enumerate(qids)}
        per_layer[L] = (bank_vectors(V, n, [qpos[q] for q in qs], w), roles, n[:, [qpos[q] for q in qs]].sum(1))
    roles = per_layer[layers[0]][1]
    for i, role in enumerate(roles):
        vec = torch.tensor(np.stack([per_layer[L][0][i] for L in layers]), dtype=torch.float32)
        torch.save({"vector": vec, "layers": [int(L) for L in layers], "type": "mean" if role == "default" else "pos_3",
                    "role": role, "n": int(per_layer[layers[0]][2][i]), "question_ids": [int(q) for q in qs],
                    "balance": w, "variant": variant}, out_dir / f"{role}.pt")
    logger.info(f"{variant}: {len(roles)} role files -> {out_dir}")
    return out_dir


# --- scoring -------------------------------------------------------------------------------------

def score_banks(layer: RB.Layer, banks: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Every `random_bank.metrics` column for D5-D9 and D5 + K, D9 + K of every bank, a-priori signs.
    A direction scoring RepE below 0.5 is logged, never flipped."""
    k_scores = {n: s[:, 0] for n, s in RB._scores_of(layer, layer.k[:, None]).items()}
    cols, labels = [], []
    for bname, rv in banks.items():
        D = directions(rv)
        for d in AXES:
            cols.append(D[d])
            labels.append((bname, d))
    sc = RB._scores_of(layer, np.stack(cols, axis=1))
    m = RB.metrics(layer, sc)
    kidx = [i for i, (_, d) in enumerate(labels) if d in WITH_K]
    mk = RB.metrics(layer, RB._with_k({n: s[:, kidx] for n, s in sc.items()}, k_scores))
    repe = RB.auroc_columns(layer.sets["RepE"][0], sc["RepE"])
    rows = []
    for i, (b, d) in enumerate(labels):
        rows.append({"bank": b, "layer": layer.layer, "direction": d, "repe_ok": bool(repe[i] >= 0.5), **m.iloc[i].to_dict()})
        if repe[i] < 0.5:
            logger.warning(f"layer {layer.layer}: {b} {d} scores RepE at {repe[i]:.3f} (a-priori sign kept)")
    for j, i in enumerate(kidx):
        b, d = labels[i]
        rows.append({"bank": b, "layer": layer.layer, "direction": d + " + K", "repe_ok": bool(repe[i] >= 0.5),
                     **mk.iloc[j].to_dict()})
    return pd.DataFrame(rows)


def score(layers=LAYERS, tag: str = "s3") -> None:
    """values.csv (bank x layer x direction x metric) and matrix_L{L}.csv (family x set, D9 and D9 + K)."""
    freeze_check()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    for L in layers:
        layer = RB.Layer(L, gs=False, aisi=False)
        t = score_banks(layer, banks_at(L, tag))
        tables.append(t)
        fam = t[t["bank"].str.startswith("fam_") | t["bank"].isin(["B150", "B30", "B30-thesis"])]
        fam = fam[fam["direction"].isin(["D9", "D9 + K"])]
        cols = [c for c in fam.columns if c.startswith("auroc ")] + ["LB-5 mean", "transfer"]
        fam.pivot_table(index="bank", columns="direction", values=cols).round(4).to_csv(OUT / f"matrix_L{L}.csv")
        logger.info(f"layer {L}: {len(t)} rows scored")
    pd.concat(tables).round(6).to_csv(OUT / f"values_{tag}.csv" if tag != "s3" else OUT / "values.csv", index=False)


def null(n: int = N_NULL, layer: int = CRITERIA_LAYER, tag: str = "s3", seed: int = SEED) -> None:
    """Question-subsampling nulls at one layer, D9 and D9 + K: 200 random 30-of-150; the originals plus a random
    30 of the 90 diverse; a random 30 of the 90 alone."""
    freeze_check()
    rows = questions()
    V, nn, roles, qids = load_store(tag, layer)
    qpos = {q: i for i, q in enumerate(qids)}
    orig = [r["id"] for r in rows if r["set"] == "orig"]
    div = [r["id"] for r in rows if r["set"] == "div"]
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        draws.append(("rand30of150", i, sorted(rng.choice(qids, 30, replace=False).tolist())))
    for i in range(n):
        draws.append(("orig+rand30div", i, orig + sorted(rng.choice(div, 30, replace=False).tolist())))
    for i in range(n):
        draws.append(("rand30div", i, sorted(rng.choice(div, 30, replace=False).tolist())))
    lay = RB.Layer(layer, gs=False, aisi=False)
    k_scores = {nm: s[:, 0] for nm, s in RB._scores_of(lay, lay.k[:, None]).items()}
    cols, keys = [], []
    for kind, i, qs in draws:
        vec = bank_vectors(V, nn, [qpos[q] for q in qs], "plain")
        cols.append(directions({r: vec[j] for j, r in enumerate(roles)})["D9"])
        keys.append((kind, i))
    sc = RB._scores_of(lay, np.stack(cols, axis=1))
    m9 = RB.metrics(lay, sc)
    m9k = RB.metrics(lay, RB._with_k(sc, k_scores))
    out = []
    for j, (kind, i) in enumerate(keys):
        out.append({"kind": kind, "draw": i, "direction": "D9", **m9.iloc[j].to_dict()})
        out.append({"kind": kind, "draw": i, "direction": "D9 + K", **m9k.iloc[j].to_dict()})
    pd.DataFrame(out).round(6).to_csv(OUT / f"null_L{layer}.csv", index=False)
    logger.info(f"null: {len(draws)} draws scored at layer {layer}")


def _best_family(layer: int = CRITERIA_LAYER, direction: str = "D9") -> str:
    v = pd.read_csv(OUT / "values.csv")
    f = v[(v["layer"] == layer) & (v["direction"] == direction) & v["bank"].str.startswith("fam_")]
    return str(f.sort_values("LB-5 mean", ascending=False).iloc[0]["bank"])


def paired(layer: int = CRITERIA_LAYER, tag: str = "s3", n_draws: int = N_DRAWS, seed: int = SEED,
           pairs=(("B30+90div", "B30+30mono"), ("B150", "B30"), ("B30mono", "B30orig"), ("B150", "best_family"),
                  ("B150-bal", "B150"), ("B30-bal", "B30"))) -> None:
    """Paired 95% intervals of bank differences (D9 and D9 + K): stratified row resamples shared by every scorer,
    transfer with fitted signs held fixed, as `random_bank._paired`."""
    freeze_check()
    best = _best_family(layer)
    pairs = [(a, best if b == "best_family" else b) for a, b in pairs]
    names = sorted({x for p in pairs for x in p})
    lay = RB.Layer(layer, gs=False, aisi=False)
    banks = banks_at(layer, tag, names=names)
    k_scores = {nm: s[:, 0] for nm, s in RB._scores_of(lay, lay.k[:, None]).items()}
    cols, labels = [], []
    for b in names:
        D = directions(banks[b])
        cols.append(D["D9"]); labels.append((b, "D9"))
    sc = RB._scores_of(lay, np.stack(cols, axis=1))
    sck = RB._with_k(sc, k_scores)
    sc = {nm: np.concatenate([sc[nm], sck[nm]], axis=1) for nm in sc}
    labels = labels + [(b, "D9 + K") for b in names]
    m = len(labels)
    signs = {a: np.array([RB.fitted_sign(sc[a][:, j], lay.sets[a][0]) for j in range(m)]) for a in lay.order}
    sets = lay.all_sets()
    strata = {nm: (np.flatnonzero(y == 1), np.flatnonzero(y == 0)) for nm, (y, _) in sets.items()}

    def summary(idx):
        au = {nm: RB.auroc_columns(sets[nm][0][idx[nm]], sc[nm][idx[nm]]) for nm in sets}
        out = {f"auroc {nm}": au[nm] for nm in sets}
        out["LB-5 mean"] = np.mean([au[nm] for nm in RB.LB5], axis=0)
        cells = []
        for a in lay.order:
            for b in lay.order:
                if a != b:
                    s = signs[a]
                    cells.append(np.where(s == 0, 0.5, np.where(s > 0, au[b], 1 - au[b])))
        out["transfer"] = np.mean(cells, axis=0)
        return out

    rng = np.random.default_rng(seed)
    point = summary({nm: np.arange(len(sets[nm][0])) for nm in sets})
    draws = {k: np.empty((n_draws, m)) for k in point}
    for b in range(n_draws):
        idx = {nm: np.concatenate([rng.choice(p, len(p)), rng.choice(q, len(q))]) for nm, (p, q) in strata.items()}
        for k, v in summary(idx).items():
            draws[k][b] = v
    rows = []
    for a, b in pairs:
        for d in ("D9", "D9 + K"):
            ia, ib = labels.index((a, d)), labels.index((b, d))
            for k in point:
                diff = draws[k][:, ia] - draws[k][:, ib]
                lo, hi = np.quantile(diff, [0.025, 0.975])
                rows.append({"layer": layer, "direction": d, "metric": k, "a": a, "b": b, "a_value": point[k][ia],
                             "b_value": point[k][ib], "delta": point[k][ia] - point[k][ib], "lo": float(lo),
                             "hi": float(hi), "p_boot_le0": float(np.mean(diff <= 0))})
    pd.DataFrame(rows).round(6).to_csv(OUT / f"paired_ci_L{layer}.csv", index=False)
    logger.info(f"paired: {len(pairs)} pairs x 2 directions at layer {layer}, {n_draws} draws")


# --- the spectrum with questions as domains ------------------------------------------------------

def _dom_direction(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return z[y == 1].mean(0) - z[y == 0].mean(0)


def spectrum(layer: int = CRITERIA_LAYER, tag: str = "s3", n_boot: int = N_DRAWS, seed: int = SEED,
             families_spec: str = "all", prefix: str = "spectrum") -> None:
    """Per (family, set): the family D9's AUROC and its alignment with the set's own probe (plain cosine,
    covariance-weighted cosine with Sigma = the set, and with Sigma = Alpaca), for the LR own probe
    (`transfer_alignment.own_probe`) and a difference-of-means probe; the line-fit R^2 and Spearman rho^2 of
    each predictor; a family-resampling bootstrap of Delta R^2; leave-one-family-out D9; pooled vs family.
    `families_spec` "orig" restricts to the three original families (PRED-0 on a 30-question store)."""
    from scipy.stats import spearmanr
    from experiments.probes import transfer_alignment as ta

    freeze_check()
    rows = questions()
    fam = families(rows)
    if families_spec == "orig":
        fam = {k: v for k, v in fam.items() if k.startswith("orig_")}
    V, nn, roles, qids = load_store(tag, layer)
    qpos = {q: i for i, q in enumerate(qids)}
    lay = RB.Layer(layer, gs=False, aisi=False)

    def d9_of(qs):
        vec = bank_vectors(V, nn, [qpos[q] for q in qs], "plain")
        return directions({r: vec[j] for j, r in enumerate(roles)})["D9"]

    fam_d9 = {f: d9_of(qs) for f, qs in fam.items()}
    pooled = d9_of(qids)
    own = {}
    for s, (y, z) in lay.sets.items():
        o = ta.own_probe(z, y)
        own[s] = {"LR": o["own_dir"], "DoM": _dom_direction(z, y)}
    cells = []
    for f, w in fam_d9.items():
        for s, (y, z) in lay.sets.items():
            au = float(RB.auroc_columns(y, z @ w)[0])
            for probe, od in own[s].items():
                cells.append({"family": f, "set": s, "probe": probe, "auroc": au,
                              "cos_plain": ta.cosine(w, od),
                              "cos_sigma_target": ta.mahalanobis_cosine(w, od, z),
                              "cos_sigma_alpaca": ta.mahalanobis_cosine(w, od, lay.control)})
    c = pd.DataFrame(cells)
    c.round(6).to_csv(OUT / f"{prefix}_L{layer}.csv", index=False)

    preds = ("cos_plain", "cos_sigma_target", "cos_sigma_alpaca")

    def fits(frame: pd.DataFrame) -> dict:
        out = {}
        for p in preds:
            x, yv = frame[p].to_numpy(), frame["auroc"].to_numpy()
            out[f"r2_{p}"] = ta._r2_linear(x, yv)
            out[f"rho2_{p}"] = float(spearmanr(x, yv).correlation ** 2)
        out["delta_r2_target_minus_plain"] = out["r2_cos_sigma_target"] - out["r2_cos_plain"]
        return out

    frows = []
    rng = np.random.default_rng(seed)
    fnames = list(fam)
    for probe, part in c.groupby("probe"):
        base = fits(part)
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(fnames, len(fnames), replace=True)
            frame = pd.concat([part[part["family"] == f] for f in pick])
            boot.append(fits(frame)["delta_r2_target_minus_plain"])
        boot = np.array(boot)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        frows.append({"layer": layer, "probe": probe, "cells": len(part), **base, "delta_r2_lo": float(lo),
                      "delta_r2_hi": float(hi), "p_boot_le0": float(np.mean(boot <= 0))})
    pd.DataFrame(frows).round(6).to_csv(OUT / f"{prefix}_fits_L{layer}.csv", index=False)

    # leave-one-family-out and pooled vs family on the LB-5 mean
    lb5 = lambda w: float(np.mean([RB.auroc_columns(lay.sets[s][0], lay.sets[s][1] @ w)[0] for s in RB.LB5]))  # noqa: E731
    pooled_lb5 = lb5(pooled)
    lodo = []
    for f in fam:
        rest = [q for g, qs in fam.items() if g != f for q in qs]
        lodo.append({"left_out": f, "n_questions_left": len(rest), "LB-5 mean": lb5(d9_of(rest)),
                     "delta_vs_pooled": lb5(d9_of(rest)) - pooled_lb5})
    pd.DataFrame(lodo).round(6).to_csv(OUT / f"{prefix}_lodo_L{layer}.csv", index=False)
    pvf = [{"family": f, "n_questions": len(fam[f]), "LB-5 mean": lb5(w)} for f, w in fam_d9.items()]
    pvf.append({"family": "pooled (all)", "n_questions": len(qids), "LB-5 mean": pooled_lb5})
    pd.DataFrame(pvf).round(6).to_csv(OUT / f"{prefix}_pooled_vs_family_L{layer}.csv", index=False)
    logger.info(f"{prefix} at layer {layer}: {len(c)} cells; pooled LB-5 {pooled_lb5:.4f}; "
                f"best family {max(pvf[:-1], key=lambda r: r['LB-5 mean'])}\n"
                + pd.DataFrame(frows).round(4).to_string())


# --- verdict -------------------------------------------------------------------------------------

def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def verdict(layer: int = CRITERIA_LAYER) -> None:
    """H1-H5 at the criteria layer from values.csv, null_L40.csv, paired_ci_L40.csv and spectrum_fits_L40.csv;
    Holm over T1-T4; the decision rule for the confirmation bank."""
    v = pd.read_csv(OUT / "values.csv")
    v = v[v["layer"] == layer].set_index(["bank", "direction"])
    nul = pd.read_csv(OUT / f"null_L{layer}.csv")
    pc = pd.read_csv(OUT / f"paired_ci_L{layer}.csv")
    sf = pd.read_csv(OUT / f"spectrum_fits_L{layer}.csv").set_index("probe")
    rows = []

    def null_p(kind, direction, value):
        draws = nul[(nul["kind"] == kind) & (nul["direction"] == direction)]["LB-5 mean"].to_numpy()
        return float((1 + np.sum(draws <= value)) / (1 + len(draws))), draws

    for test, direction in (("T1", "D9"), ("T2", "D9 + K")):
        mono = float(v.loc[("B30+30mono", direction), "LB-5 mean"])
        p, draws = null_p("orig+rand30div", direction, mono)
        rows.append({"test": test, "hypothesis": "H1 diverse > monotone at equal count", "direction": direction,
                     "statistic": f"B30+30mono LB-5 {mono:.4f} vs 200 draws of B30+rand30div (median {np.median(draws):.4f})",
                     "p": p, "pass": p <= 0.05})
    # H2
    mono30 = float(v.loc[("B30mono", "D9"), "LB-5 mean"])
    _, d150 = null_p("rand30of150", "D9", mono30)
    lo, hi = np.quantile(d150, [0.025, 0.975])
    r = pc[(pc["a"] == "B30mono") & (pc["b"] == "B30orig") & (pc["direction"] == "D9") & (pc["metric"] == "LB-5 mean")].iloc[0]
    h2 = (lo <= mono30 <= hi) and (r["lo"] >= -EQUIV) and (r["hi"] <= EQUIV)
    rows.append({"test": "H2", "hypothesis": "H2 monotone ~ original", "direction": "D9",
                 "statistic": f"B30mono {mono30:.4f} in [{lo:.4f}, {hi:.4f}] of rand30of150; paired CI vs B30orig "
                              f"[{r['lo']:.4f}, {r['hi']:.4f}] within +-{EQUIV}", "p": np.nan, "pass": bool(h2)})
    # T3
    best = _best_family(layer)
    r = pc[(pc["a"] == "B150") & (pc["b"] == best) & (pc["direction"] == "D9") & (pc["metric"] == "LB-5 mean")].iloc[0]
    rows.append({"test": "T3", "hypothesis": "H3 pooled >= best single family", "direction": "D9",
                 "statistic": f"B150 {r['a_value']:.4f} - {best} {r['b_value']:.4f} = {r['delta']:.4f}, CI [{r['lo']:.4f}, {r['hi']:.4f}]",
                 "p": float(r["p_boot_le0"]), "pass": bool(r["delta"] >= 0 and r["lo"] > -0.01)})
    # T4
    s = sf.loc["LR"]
    rows.append({"test": "T4", "hypothesis": "H4 Sigma-cosine predicts transfer better than plain cosine", "direction": "D9",
                 "statistic": f"R2 sigma {s['r2_cos_sigma_target']:.3f} vs plain {s['r2_cos_plain']:.3f}; delta "
                              f"{s['delta_r2_target_minus_plain']:.3f} CI [{s['delta_r2_lo']:.3f}, {s['delta_r2_hi']:.3f}]",
                 "p": float(s["p_boot_le0"]), "pass": bool(s["delta_r2_target_minus_plain"] > 0 and s["delta_r2_lo"] > 0)})
    # H5 exploratory
    col = "auroc HP-KR identical text"
    r = pc[(pc["a"] == "B150") & (pc["b"] == "B30") & (pc["direction"] == "D9") & (pc["metric"] == col)]
    if len(r):
        r = r.iloc[0]
        rows.append({"test": "H5", "hypothesis": "H5 HP-KR identical-text slice, B150 vs B30 (exploratory)", "direction": "D9",
                     "statistic": f"{r['a_value']:.4f} vs {r['b_value']:.4f}, CI [{r['lo']:.4f}, {r['hi']:.4f}]",
                     "p": np.nan, "pass": bool(r["lo"] > 0)})
    t = pd.DataFrame(rows)
    mask = t["test"].isin(HOLM_TESTS)
    t.loc[mask, "holm_p"] = holm(t.loc[mask, "p"].to_numpy())
    # decision rule
    cands = ["B30", "B30+30mono", "B30+90div", "B150"]
    vals = {c: float(v.loc[(c, "D9 + K"), "LB-5 mean"]) for c in cands}
    top = max(vals.values())
    chosen = [c for c in cands if vals[c] >= top - TIE][-1]  # ties within TIE -> the larger bank (listed last)
    t.to_csv(OUT / "verdict.csv", index=False)
    lines = [f"{r['test']}: {r['hypothesis']} [{r['direction']}] {r['statistic']} p={r['p']:.4f} holm={r.get('holm_p', np.nan):.4f} "
             f"-> {'pass' if r['pass'] else 'fail'}" for _, r in t.iterrows()]
    lines.append(f"decision (D9 + K LB-5 at layer {layer}): {vals} -> confirmation bank {chosen}")
    (OUT / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("\n".join(lines))


# --- GATE-0 and PRED-0 on the 30 originals ------------------------------------------------------

def gate0(tag: str = "s3q30", layers=LAYERS, tol_cos: float = 0.9995, tol_auroc: float = 0.005) -> None:
    """The store of the 30 originals must recombine to the thesis bank (per-role cosine, equal n) and reproduce
    its D5/D8/D9 zero-shot AUROCs; then PRED-0: pooled vs original-family D9 and the alignment fits."""
    import torch

    fails = []
    report = []
    for L in layers:
        V, n, roles, qids = load_store(tag, L)
        vec = bank_vectors(V, n, list(range(len(qids))), "plain")
        for i, role in enumerate(roles):
            p = pcs.EXTENDED_VECTORS_DIR / f"{role}.pt"
            d = torch.load(p, weights_only=True)
            sl = [int(x) for x in d.get("layers", range(d["vector"].shape[0]))]
            ref = d["vector"][sl.index(L)].double().numpy()
            cos = float(vec[i] @ ref / (np.linalg.norm(vec[i]) * np.linalg.norm(ref)))
            n_ref, n_store = int(d["n"]), int(n[i].sum())
            report.append({"layer": L, "role": role, "cos": cos, "n_thesis": n_ref, "n_store": n_store})
            if cos < tol_cos:
                fails.append(f"L{L} {role}: cos {cos:.5f} < {tol_cos}")
            if n_ref != n_store:
                fails.append(f"L{L} {role}: n {n_store} != thesis {n_ref}")
    rep = pd.DataFrame(report)
    rep.round(6).to_csv(OUT / "gate0_parity.csv", index=False)
    logger.info(f"GATE-0 parity: min cos {rep['cos'].min():.5f}; n mismatches "
                f"{int((rep['n_thesis'] != rep['n_store']).sum())}")
    # zero-shot reproduction at layer 40 against the stored thesis numbers
    zs = pd.read_csv(REPO_ROOT / "results" / "persona_ood_ext_L40" / "zero_shot.csv")
    zs = zs[zs["layer"] == 40].set_index("dataset")
    V, n, roles, qids = load_store(tag, 40)
    vec = bank_vectors(V, n, list(range(len(qids))), "plain")
    D = directions({r: vec[i] for i, r in enumerate(roles)})
    lay = RB.Layer(40, gs=False, aisi=False)
    inv = {v: k for k, v in rd.SHORT.items()}
    for short, (y, z) in lay.sets.items():
        name = inv.get(short, short)
        if name not in zs.index:
            continue
        for d, col in (("D5", "auroc_axis"), ("D8", "auroc_denial_axis"), ("D9", "auroc_ext_axis")):
            got = float(RB.auroc_columns(y, z @ D[d])[0])
            ref = float(zs.loc[name, col])
            if abs(got - ref) > tol_auroc:
                fails.append(f"{short} {d}: store {got:.4f} vs thesis {ref:.4f}")
    # PRED-0
    fam = {"orig_disclosure": list(range(0, 10)), "orig_sycophancy": list(range(10, 20)), "orig_knowledge": list(range(20, 30))}
    qpos = {q: i for i, q in enumerate(qids)}
    pred = []
    for f, qs in list(fam.items()) + [("pooled", qids)]:
        try:
            fv = bank_vectors(V, n, [qpos[q] for q in qs], "plain")
        except ValueError as error:
            logger.warning(f"PRED-0 {f}: {error}"); continue
        w = directions({r: fv[i] for i, r in enumerate(roles)})["D9"]
        au = {s: float(RB.auroc_columns(y, z @ w)[0]) for s, (y, z) in lay.sets.items()}
        pred.append({"family": f, **{f"auroc {s}": a for s, a in au.items()}, "LB-5 mean": np.mean([au[s] for s in RB.LB5])})
    pr = pd.DataFrame(pred)
    pr.round(6).to_csv(OUT / "pred0_families_L40.csv", index=False)
    (OUT / "gate0.md").write_text(
        f"GATE-0 {'PASSED' if not fails else 'FAILED'}\n" + "\n".join(fails) + "\n\nPRED-0 (layer 40):\n" + pr.round(4).to_string() + "\n",
        encoding="utf-8")
    if fails:
        logger.error("GATE-0 FAILED:\n" + "\n".join(fails))
        raise SystemExit(1)
    logger.info("GATE-0 passed\n" + pr.round(4).to_string())


# --- the free dry run on the stored per-rollout reads -------------------------------------------

def phase0(layers=LAYERS) -> None:
    """D5, D8, D9 per original question family from the per-rollout reads of `persona_finetuned/base.npz`
    (25 contrast personas, one rollout per question, layers 20/40/44; `_L36` for 36). A dry run of the
    family-vs-pooled comparison; no `default`, no sycophancy roles, so D6/D7 are unavailable."""
    acts = REPO_ROOT / "activations"
    fam = {"orig_disclosure": range(0, 10), "orig_sycophancy": range(10, 20), "orig_knowledge": range(20, 30),
           "pooled": range(0, 30)}
    out = []
    for L in layers:
        if L == 36:
            data = np.load(acts / "persona_finetuned_L36" / "base.npz", allow_pickle=True)
        else:
            data = np.load(acts / "persona_finetuned" / "base.npz", allow_pickle=True)
        stored = [int(x) for x in data["layers"]] if "layers" in data.files else ([36] if L == 36 else [20, 40, 44])
        X = data["pooled"][:, stored.index(L)].astype(np.float64)
        persona = np.array([str(p) for p in data["persona"]])
        qi = data["question_index"].astype(int)
        lay = RB.Layer(L, gs=False, aisi=False)
        k_scores = {nm: s[:, 0] for nm, s in RB._scores_of(lay, lay.k[:, None]).items()}
        for f, qs in fam.items():
            mask = np.isin(qi, list(qs))
            means = {p: X[mask & (persona == p)].mean(0) for p in np.unique(persona) if (mask & (persona == p)).any()}
            g = {k: [p for p in v if p in means] for k, v in GROUPS.items()}
            m = lambda names: np.stack([means[p] for p in names]).mean(0)  # noqa: E731
            D = {"D5": rd._unit(m(g["deception"]) - m(g["honesty"])),
                 "D8": rd._unit(m(g["denial"]) - m(g["candour"])),
                 "D9": rd._unit(m(g["deception"] + g["denial"]) - m(g["honesty"] + g["candour"]))}
            sc = RB._scores_of(lay, np.stack([D[d] for d in D], axis=1))
            met = RB.metrics(lay, sc)
            metk = RB.metrics(lay, RB._with_k({nm: s[:, [2]] for nm, s in sc.items()}, k_scores))
            for j, d in enumerate(D):
                out.append({"layer": L, "family": f, "direction": d, "personas": len(means),
                            "rollouts": int(mask.sum()), **met.iloc[j].to_dict()})
            out.append({"layer": L, "family": f, "direction": "D9 + K", "personas": len(means),
                        "rollouts": int(mask.sum()), **metk.iloc[0].to_dict()})
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(out)
    t.round(6).to_csv(OUT / "phase0_base_npz.csv", index=False)
    show = t[t["direction"].isin(["D9", "D9 + K"])].pivot_table(index=["layer", "direction"], columns="family",
                                                                  values="LB-5 mean")
    logger.info("phase0 LB-5 mean by family:\n" + show.round(4).to_string())


# --- the one confirmation read -------------------------------------------------------------------

def confirm(bank: str, tag: str = "s3", layers=(20, 36, 40, 44), n_draws: int = N_DRAWS, seed: int = SEED) -> None:
    """The selected bank beside B30 on AISI held-out (layers 20/40/44) and Gender-Secret (20/36/40/44), D9 and
    D9 + K, with paired bootstrap intervals. Runs once: refuses if confirm.csv exists."""
    out = OUT / "confirm.csv"
    if out.exists():
        raise SystemExit(f"{out} exists: the confirmation read has been done; it is not repeated")
    freeze_check()
    rows = []
    for L in layers:
        lay = RB.Layer(L, gs=True, aisi=True)
        banks = banks_at(L, tag, names=["B30", bank])
        k_scores = {nm: s[:, 0] for nm, s in RB._scores_of(lay, lay.k[:, None]).items()}
        labels = [("B30", "D9"), (bank, "D9")]
        sc = RB._scores_of(lay, np.stack([directions(banks[b])["D9"] for b, _ in labels], axis=1))
        sck = RB._with_k(sc, k_scores)
        sc = {nm: np.concatenate([sc[nm], sck[nm]], axis=1) for nm in sc}
        labels += [("B30", "D9 + K"), (bank, "D9 + K")]
        sets = {nm: v for nm, v in lay.all_sets().items() if nm in ("AISI held out", "Gender-Secret")}
        if not sets:
            continue
        rng = np.random.default_rng(seed)
        for nm, (y, _) in sets.items():
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            point = RB.auroc_columns(y, sc[nm])
            draws = np.empty((n_draws, len(labels)))
            for b in range(n_draws):
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                draws[b] = RB.auroc_columns(y[idx], sc[nm][idx])
            for d in ("D9", "D9 + K"):
                ia, ib = labels.index((bank, d)), labels.index(("B30", d))
                diff = draws[:, ia] - draws[:, ib]
                lo, hi = np.quantile(diff, [0.025, 0.975])
                rows.append({"layer": L, "set": nm, "direction": d, "bank": bank, "auroc": point[ia],
                             "B30 auroc": point[ib], "delta": point[ia] - point[ib], "lo": float(lo), "hi": float(hi)})
    t = pd.DataFrame(rows)
    t.round(6).to_csv(out, index=False)
    logger.info("confirmation read:\n" + t.round(4).to_string())


if __name__ == "__main__":
    fire.Fire({"store": store, "materialise": materialise, "score": score, "null": null, "paired": paired,
               "spectrum": spectrum, "verdict": verdict, "gate0": gate0, "phase0": phase0, "confirm": confirm})
