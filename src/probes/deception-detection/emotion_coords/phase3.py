"""Phase 3 of the emotion-coordinate study: every feature set, both probe families, protocols A, B and C.

    python -m emotion_coords.phase3 run --early         # the arms that need no story reads, while NDIF reads
    python -m emotion_coords.phase3 run [--workers 4]   # blocks: 20/28/40/52 at pooling (a); 20 at (b) and (c)
    python -m emotion_coords.phase3 merge               # parts -> results/emotion_coords/phase3/probes.csv + scores
    python -m emotion_coords.phase3 summary             # protocol-B tables and P1-P6 (ANALYSIS_PLAN.md)

Rows are the spec's caps (`phase2 subset`); Alpaca (2,000) sets thresholds; the instructed pairs (550) train
protocol A. Each (block, probe family, feature set) is one unit, saved to data/emotion_coords/phase3/parts/
when it finishes, so a stopped run resumes where it left off. Units run in parallel worker processes; the
erase arms (protocol B, pooling a) run after the raw arm, whose fold-chosen C they reuse.
"""

import os
import pickle
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C
from emotion_coords import lbdata as LB
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords.pilot import ZERO_SHOT, erase_sets, persona_basis

BLOCKS = ((20, "a"), (28, "a"), (40, "a"), (52, "a"), (20, "b"), (20, "c"))
HEAVY = ("raw", "trainpca_pc")


PARTS = "parts"  # a smoke test writes elsewhere (`--parts smoke`)


def _parts() -> Path:
    return C.data_dir("phase3", PARTS)


def _unit_name(layer: int, pooling: str, family: str, fs: P.FeatureSet) -> str:
    return f"L{layer}{pooling}_{family}_{fs.name}_s{fs.seed}"


def _write(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def independent_sets(layer: int, data: dict, hidden: int, n_random: int = 20,
                     persona: tuple[np.ndarray, np.ndarray] | None = None) -> list[P.FeatureSet]:
    """The feature sets that need no story reads (raw, training-set PCs, random, Alpaca PCs, persona PCs), so
    `run --early` can fit them while NDIF is still reading the stories. `persona`: (extended, published) bases of
    another model (`xmodel`); Llama's own by default."""
    control = C.load()["control"]
    extended, published = persona if persona is not None else (persona_basis(layer), persona_basis(layer, "published"))
    return [P.FeatureSet("raw", "raw"),
            P.FeatureSet("trainpca_pc", "trainpca_pc", None, P.KS, per_fold=lambda x: la.top_pcs(x, 20)),
            *[P.FeatureSet("random", "random", la.random_subspace(hidden, 20, seed=10_000 + s),
                           tuple(sorted(set(P.KS) | {6})), seed=s) for s in range(n_random)],
            P.FeatureSet("alpaca_pc", "alpaca_pc", la.top_pcs(data[control].x, 20), P.KS),
            P.FeatureSet("persona_pc", "persona_pc", extended, P.KS),
            # the published persona space (user decision 2026-10-30, after layers 20-40 were seen; ANALYSIS_PLAN.md)
            P.FeatureSet("persona_pub_pc", "persona_pub_pc", published, P.KS)]


def feature_sets(layer: int, data: dict, bases: dict, hidden: int, n_random: int = 20,
                 persona: tuple[np.ndarray, np.ndarray] | None = None) -> list[P.FeatureSet]:
    e, t = bases["E_components"], bases["T_components"]
    sets = independent_sets(layer, data, hidden, n_random, persona)
    persona = next(fs.basis for fs in sets if fs.name == "persona_pc")
    persona_pub = next(fs.basis for fs in sets if fs.name == "persona_pub_pc")
    sets += [P.FeatureSet("emotion_pc", "emotion_pc", e, P.KS),
             P.FeatureSet("topic_pc", "topic_pc", t, P.KS),
             *[P.FeatureSet("shuffled_pc", "shuffled_pc", bases[f"S{p}_components"], P.KS, seed=p) for p in range(5)]]
    for k in (3, 5, 10):
        union = la.orthonormal(np.vstack([persona[:k], e[:k]]))
        sets.append(P.FeatureSet("persona_emotion", "persona_emotion", union, (len(union),), seed=k))
        union = la.orthonormal(np.vstack([persona_pub[:k], e[:k]]))
        sets.append(P.FeatureSet("persona_pub_emotion", "persona_pub_emotion", union, (len(union),), seed=k))
    labels = list(bases["E_labels"])
    named = [n for n in C.load()["emotion_names"]["core"] if n in labels]
    sets.append(P.FeatureSet("named", "named", la.unit(bases["E_vectors"][[labels.index(n) for n in named]]),
                             (len(named),)))
    return sets


def exploratory_sets(data: dict, bases: dict, hidden: int, n_random: int = 5) -> list[P.FeatureSet]:
    """Exploratory (not preregistered): the whole emotion subspace (all 170 PCs of the 171 centred vectors) at
    fixed k, against same-dimension controls. Kept out of the emotion family so P1's k choice is untouched."""
    control = C.load()["control"]
    e = bases["E_components"]
    r = len(e)
    return [P.FeatureSet("emotion_all", "exploratory_emotion_all", e, (r,)),
            *[P.FeatureSet("shuffled_all", "exploratory_shuffled_all", bases[f"S{p}_components"][:r], (r,), seed=p)
              for p in range(5)],
            *[P.FeatureSet("random_all", "exploratory_random_all", la.random_subspace(hidden, r, seed=30_000 + s),
                           (r,), seed=s) for s in range(n_random)],
            P.FeatureSet("alpaca_all", "exploratory_alpaca_all", la.top_pcs(data[control].x, r), (r,)),
            P.FeatureSet("trainpca_all", "exploratory_trainpca_all", None, (r,), per_fold=lambda x: la.top_pcs(x, r))]


def _unit(family: str, fs: P.FeatureSet, data: dict, layer: int, pooling: str, path: str,
          fixed_c: dict | None = None, protocols: str = "ABC") -> tuple[str, float]:
    s = C.load()
    lb5, control, train = list(s["datasets"]), s["control"], s["train_pairs"]
    store, t0 = P.Store(), time.time()
    if fs.family.startswith("erase") or protocols == "B":
        chosen = P.protocol_b(family, fs, data, lb5, control, layer, pooling, store, fixed_c=fixed_c)
    else:
        P.protocol_a(family, fs, data, train, lb5, control, layer, pooling, store)
        chosen = P.protocol_b(family, fs, data, lb5, control, layer, pooling, store)
        P.protocol_c(family, fs, data, lb5, control, layer, pooling, store)
    seconds = time.time() - t0
    _write(Path(path), {"rows": store.rows, "scores": store.scores, "chosen_c": chosen, "seconds": seconds})
    return Path(path).name, seconds


def run(workers: int = 4, threads: int = 3, blocks: str | None = None, n_random: int = 20, variant: str = "full",
        bases: str = "full", parts: str = "parts", early: bool = False) -> None:
    """`variant`, `bases` and `parts` exist for a small smoke test: pilot rows and bases, a separate parts folder.
    `early` fits only `independent_sets`, before the bases exist; a later full run skips their saved parts."""
    from joblib import Parallel, delayed

    global PARTS
    PARTS = parts

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)  # inherited by the worker processes
    from deception_detection.utils import REPO_ROOT
    from deception_detection.scripts.persona_ood import run_probe

    s = C.load()
    lb5, control, train = LB.lb5(), s["control"], s["train_pairs"]
    subset = LB.subset(variant)
    todo_blocks = BLOCKS if blocks is None else tuple((int(b[:-1]), b[-1]) for b in str(blocks).split(","))
    tmp = str(C.data_dir("phase3", "joblib_tmp"))
    for layer, pooling in todo_blocks:
        t_block = time.time()
        data = LB.load(layer, pooling, subset, bc_variant=variant)
        hidden = data[control].x.shape[1]
        if early:
            base, sets = None, independent_sets(layer, data, hidden, n_random)
        else:
            with np.load(C.data_dir("bases") / f"{bases}_L{layer}.npz") as z:
                base = {k: z[k] for k in z.files}
            sets = feature_sets(layer, data, base, hidden, n_random)
        units = [(fam, fs) for fs in sorted(sets, key=lambda f: f.name not in HEAVY) for fam in ("lr", "dom")]
        todo = [(fam, fs) for fam, fs in units if not (_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl").exists()]
        logger.info(f"L{layer}{pooling}: {len(units)} units, {len(todo)} to run on {workers} workers")
        done = Parallel(n_jobs=workers, backend="loky", temp_folder=tmp, max_nbytes="1M")(
            delayed(_unit)(fam, fs, data, layer, pooling, str(_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl"))
            for fam, fs in todo)
        for name, seconds in done:
            logger.info(f"  {name}: {seconds:.0f} s")
        if early:
            logger.info(f"L{layer}{pooling} (early) done in {(time.time() - t_block) / 60:.1f} min")
            continue
        if pooling == "a":  # erase arms, protocol B: lr reuses the raw arm's fold-chosen C
            with open(_parts() / f"{_unit_name(layer, pooling, 'lr', P.FeatureSet('raw', 'raw'))}.pkl", "rb") as f:
                raw_c = pickle.load(f)["chosen_c"]
            erase = [(fam, fs) for fs in erase_sets(base, hidden) for fam in ("lr", "dom")]
            erase = [(fam, fs) for fam, fs in erase
                     if not (_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl").exists()]
            Parallel(n_jobs=workers, backend="loky", temp_folder=tmp, max_nbytes="1M")(
                delayed(_unit)(fam, fs, data, layer, pooling,
                               str(_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl"),
                               raw_c if fam == "lr" else None) for fam, fs in erase)
            # exploratory: the whole emotion subspace at fixed k, protocol B only
            explore = [(fam, fs) for fs in exploratory_sets(data, base, hidden) for fam in ("lr", "dom")]
            explore = [(fam, fs) for fam, fs in explore
                       if not (_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl").exists()]
            Parallel(n_jobs=workers, backend="loky", temp_folder=tmp, max_nbytes="1M")(
                delayed(_unit)(fam, fs, data, layer, pooling,
                               str(_parts() / f"{_unit_name(layer, pooling, fam, fs)}.pkl"), None, "B")
                for fam, fs in explore)
        extra = _parts() / f"L{layer}{pooling}_zero_shot_and_published.pkl"
        if not extra.exists():
            store = P.Store()
            labels = list(base["E_labels"])
            P.zero_shot({n: (base["E_vectors"][labels.index(n)], sign) for n, sign in ZERO_SHOT.items()
                         if n in labels}, data, lb5, control, layer, pooling, store)
            run_dir = REPO_ROOT / "results" / ("XXXX-9-llama-all" if layer == 20 else f"XXXX-9-llama-all-L{layer}")
            if pooling == "a" and run_dir.exists():  # the published probes exist every fourth block only
                score = run_probe(run_dir)
                ctrl = score(data[control].x)
                for t in lb5:
                    store.add({"protocol": "A-published", "feature_set": "raw", "fs_family": "raw", "seed": 0,
                               "layer": layer, "pooling": "a", "train": train, "probe": "lb_mean_probe",
                               "k": hidden, "test": t}, data[t].y, score(data[t].x), ctrl, C=0.1,
                              inner_auroc=np.nan, chosen=True)
            _write(extra, {"rows": store.rows, "scores": store.scores, "chosen_c": None, "seconds": 0.0})
        logger.info(f"L{layer}{pooling} done in {(time.time() - t_block) / 60:.1f} min")


def run_mlp(workers: int = 4, threads: int = 3, layers: str | None = None, n_random: int = 5,
            variant: str = "full", bases: str = "full", parts: str = "parts") -> None:
    """Exploratory MLP probes on every projected feature set (not raw), protocol B, pooling (a): k and weight
    decay by inner leave-one-set-out CV. Runs after `run`, so the preregistered fits come first."""
    from joblib import Parallel, delayed

    global PARTS
    PARTS = parts
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    s = C.load()
    subset = LB.subset(variant)
    tmp = str(C.data_dir("phase3", "joblib_tmp"))
    for layer in (s["layers"] if layers is None else C.int_list(layers)):
        t_block = time.time()
        data = LB.load(layer, "a", subset)
        with np.load(C.data_dir("bases") / f"{bases}_L{layer}.npz") as z:
            base = {k: z[k] for k in z.files}
        hidden = data[s["control"]].x.shape[1]
        sets = [fs for fs in feature_sets(layer, data, base, hidden, n_random) if fs.name != "raw"]
        sets += exploratory_sets(data, base, hidden)
        todo = [fs for fs in sets if not (_parts() / f"{_unit_name(layer, 'a', 'mlp', fs)}.pkl").exists()]
        logger.info(f"MLP L{layer}a: {len(sets)} units, {len(todo)} to run on {workers} workers")
        Parallel(n_jobs=workers, backend="loky", temp_folder=tmp, max_nbytes="1M")(
            delayed(_unit)("mlp", fs, data, layer, "a", str(_parts() / f"{_unit_name(layer, 'a', 'mlp', fs)}.pkl"),
                           None, "B") for fs in todo)
        logger.info(f"MLP L{layer}a done in {(time.time() - t_block) / 60:.1f} min")


def merge(parts: str = "parts", out: str = "phase3") -> None:
    global PARTS
    PARTS = parts
    rows, scores, seconds = [], {}, []
    for path in sorted(_parts().glob("*.pkl")):
        with open(path, "rb") as f:
            part = pickle.load(f)
        rows.extend(part["rows"])
        scores.update(part["scores"])
        seconds.append({"unit": path.stem, "seconds": part["seconds"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(C.results_dir(out) / "probes.csv", index=False)
    pd.DataFrame(seconds).to_csv(C.results_dir(out) / "unit_seconds.csv", index=False)
    with open(C.data_dir(out) / "scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    logger.info(f"merged {len(seconds)} units: {len(frame)} rows, {sum(s['seconds'] for s in seconds) / 3600:.1f} h of fitting")


# --- P1-P6 (ANALYSIS_PLAN.md) ------------------------------------------------------------------------


def _key(row) -> tuple:
    fields = ("protocol", "feature_set", "fs_family", "seed", "layer", "pooling", "train", "probe", "k", "test")
    return tuple(sorted((f, row[f]) for f in fields))


class Arms:
    """Held-out scores of protocol B at one layer, pooling and probe family, and their bootstrap AUROCs."""

    def __init__(self, frame: pd.DataFrame, scores: dict, ys: dict, layer: int, pooling: str, probe: str,
                 n_boot: int):
        self.b = frame[(frame["protocol"] == "B") & (frame["layer"] == layer) & (frame["pooling"] == pooling)
                       & (frame["probe"] == probe)]
        self.scores, self.ys = scores, ys
        self.boots = {t: P.Resamples.draw(y, n_boot) for t, y in ys.items()}

    def rows(self, test: str, feature_set: str, k: int | None = None, chosen: bool = False) -> pd.DataFrame:
        r = self.b[(self.b["test"] == test) & (self.b["feature_set"] == feature_set)]
        if k is not None:
            r = r[r["k"] == k]
        return r[r["chosen"]] if chosen else r

    def chosen_k(self, test: str, family: str) -> int:
        return int(self.b[(self.b["test"] == test) & (self.b["fs_family"] == family) & self.b["chosen"]]["k"].iloc[0])

    def boot(self, test: str, rows: pd.DataFrame) -> tuple[float, np.ndarray]:
        """Point AUROC and bootstrap AUROCs of `rows` on `test`, averaged over the rows (random seeds)."""
        if rows.empty:
            return np.nan, np.full(len(self.boots[test].pos), np.nan)
        y = self.ys[test]
        point = float(np.mean([P.auroc(y, self.scores[_key(r)]) for _, r in rows.iterrows()]))
        draws = np.mean([self.boots[test].aurocs(y, self.scores[_key(r)]) for _, r in rows.iterrows()], axis=0)
        return point, draws

    def best_by_inner(self, test: str, feature_set: str) -> pd.DataFrame:
        r = self.rows(test, feature_set)
        return r.loc[[r["inner_auroc"].idxmax()]]


def _delta(a: dict, b: dict, tests: list[str]) -> dict:
    """Mean over `tests` of a - b, point and paired-bootstrap CI."""
    point = float(np.mean([a[t][0] - b[t][0] for t in tests]))
    draws = np.mean([a[t][1] - b[t][1] for t in tests], axis=0)
    lo, hi = P.ci(draws[np.isfinite(draws)]) if np.isfinite(draws).any() else (np.nan, np.nan)
    return {"delta": point, "ci_lo": lo, "ci_hi": hi}


def summary(n_boot: int = 1000, probe: str = "lr", out: str = "phase3", variant: str = "full",
            layers: str | None = None) -> None:
    s = C.load()
    frame = pd.read_csv(C.results_dir(out) / "probes.csv")
    with open(C.data_dir(out) / "scores.pkl", "rb") as f:
        scores = pickle.load(f)
    lb5, subset = LB.lb5(), LB.subset(variant)
    hpc, hpkr, it, idd = lb5[1], lb5[2], lb5[3], lb5[4]
    verdicts, tables, exploratory = [], [], []
    for layer in (s["layers"] if layers is None else C.int_list(layers)):
        data = LB.load(layer, "a", subset)
        ys = {t: data[t].y for t in lb5}
        arms = Arms(frame, scores, ys, layer, "a", probe, n_boot)
        hidden = data[lb5[0]].x.shape[1]
        k_e = {t: arms.chosen_k(t, "emotion_pc") for t in lb5}
        # persona arm: the published persona space (user decision 2026-10-30); the extended thesis D4 basis is
        # reported beside it as a sensitivity check (ANALYSIS_PLAN.md addendum)
        k_p = {t: arms.chosen_k(t, "persona_pub_pc") for t in lb5}
        k_x = {t: arms.chosen_k(t, "persona_pc") for t in lb5}
        emo = {t: arms.boot(t, arms.rows(t, "emotion_pc", k_e[t])) for t in lb5}
        rnd = {t: arms.boot(t, arms.rows(t, "random", k_e[t])) for t in lb5}
        top = {t: arms.boot(t, arms.rows(t, "topic_pc", k_e[t])) for t in lb5}
        per = {t: arms.boot(t, arms.rows(t, "persona_pub_pc", k_p[t])) for t in lb5}
        pe = {t: arms.boot(t, arms.best_by_inner(t, "persona_pub_emotion")) for t in lb5}
        per_x = {t: arms.boot(t, arms.rows(t, "persona_pc", k_x[t])) for t in lb5}
        pe_x = {t: arms.boot(t, arms.best_by_inner(t, "persona_emotion")) for t in lb5}
        raw = {t: arms.boot(t, arms.rows(t, "raw", hidden)) for t in lb5}
        tables.append(pd.DataFrame({"test": lb5, "k_emotion": [k_e[t] for t in lb5],
                                    "k_persona_pub": [k_p[t] for t in lb5], "k_persona_ext": [k_x[t] for t in lb5],
                                    **{name: [d[t][0] for t in lb5] for name, d in (
                                        ("emotion_pc", emo), ("random_mean", rnd), ("topic_pc", top),
                                        ("persona_pub_pc", per), ("persona_pub_emotion", pe),
                                        ("persona_pc_extended", per_x), ("persona_emotion_extended", pe_x),
                                        ("raw", raw))}}
                                   ).assign(layer=layer))
        d_rand, d_topic = _delta(emo, rnd, lb5), _delta(emo, top, lb5)
        p1 = all(d["delta"] >= 0.03 and d["ci_lo"] > 0 for d in (d_rand, d_topic))
        d_it, d_it_x = _delta(emo, per, [it]), _delta(emo, per_x, [it])
        p2, p2_x = d_it["ci_lo"] > 0, d_it_x["ci_lo"] > 0
        p3_parts = {t: emo[t][0] - rnd[t][0] for t in (hpc, hpkr)}
        p3 = all(v < 0.02 for v in p3_parts.values())
        d_pe, d_pe_x = _delta(pe, per, lb5), _delta(pe_x, per_x, lb5)
        p4, p4_x = d_pe["delta"] >= 0.02, d_pe_x["delta"] >= 0.02
        zs = frame[(frame["protocol"] == "zero-shot") & (frame["layer"] == layer) & (frame["pooling"] == "a")]
        zs_auc = {(r["feature_set"], r["test"]): r["auroc"] for _, r in zs.iterrows()}
        p5_parts = {e: (zs_auc.get((f"zs_{e}", it), np.nan), zs_auc.get((f"zs_{e}", idd), np.nan))
                    for e in ("guilty", "afraid", "desperate")}
        p5 = any(a > 0.6 and b < 0.55 for a, b in p5_parts.values())
        p6_parts = {}
        for variant, seed in (("erase_emotion_5", 5), ("erase_emotion_20", 20), ("leace", None)):
            em = arms.b[(arms.b["feature_set"] == variant)]
            rank = int(em["seed"].iloc[0]) if seed is None else seed
            er = {t: arms.boot(t, arms.rows(t, variant)) for t in lb5}
            rr = {t: arms.boot(t, arms.rows(t, f"erase_random_{rank}")) for t in lb5}
            d = _delta(er, rr, lb5)
            gain = float(np.mean([er[t][0] - raw[t][0] for t in lb5]))
            p6_parts[variant] = d | {"rank": rank, "gain_over_raw": gain, "holds": d["ci_lo"] > 0 and gain > 0}
        p6 = sum(v["holds"] for v in p6_parts.values()) >= 2
        expl = {n: {t: arms.boot(t, arms.rows(t, n)) for t in lb5}
                for n in ("emotion_all", "shuffled_all", "random_all", "alpaca_all", "trainpca_all")}
        if np.isfinite([expl["emotion_all"][t][0] for t in lb5]).all():
            row = {"layer": layer, "k": int(arms.rows(lb5[0], "emotion_all")["k"].iloc[0])}
            row |= {f"mean_{n}": float(np.mean([d[t][0] for t in lb5])) for n, d in expl.items()}
            row["mean_raw"] = float(np.mean([raw[t][0] for t in lb5]))
            for n in ("shuffled_all", "random_all", "alpaca_all", "trainpca_all"):
                d = _delta(expl["emotion_all"], expl[n], lb5)
                row |= {f"emotion_all_minus_{n}": d["delta"], f"ci_lo_{n}": d["ci_lo"], f"ci_hi_{n}": d["ci_hi"]}
            exploratory.append(row)
        verdicts.append({"layer": layer, "P1": p1, "P1_vs_random": d_rand, "P1_vs_topic": d_topic,
                         "P2": p2, "P2_IT_emotion_minus_persona": d_it, "P3": p3, "P3_emotion_minus_random": p3_parts,
                         "P4": p4, "P4_pe_minus_persona": d_pe, "P5": p5, "P5_IT_ID": p5_parts, "P6": p6,
                         "P6_variants": p6_parts,
                         "persona_basis": "published (arXiv 2605.09391: 8 deception + 7 honesty personas)",
                         "P2_extended": p2_x, "P2_extended_IT_emotion_minus_persona": d_it_x,
                         "P4_extended": p4_x, "P4_extended_pe_minus_persona": d_pe_x,
                         "mean_auroc": {n: float(np.mean([d[t][0] for t in lb5])) for n, d in (
                             ("emotion_pc", emo), ("random_mean", rnd), ("topic_pc", top),
                             ("persona_pub_pc", per), ("persona_pub_emotion", pe), ("persona_pc_extended", per_x),
                             ("persona_emotion_extended", pe_x), ("raw", raw))}})
        logger.info(f"L{layer}: P1 {p1} P2 {p2} P3 {p3} P4 {p4} P5 {p5} P6 {p6} | extended persona: P2 {p2_x} P4 {p4_x}")
    pd.concat(tables).to_csv(C.results_dir(out) / f"protocol_b_per_set_{probe}.csv", index=False)
    if exploratory:
        pd.DataFrame(exploratory).to_csv(C.results_dir(out) / f"exploratory_emotion_all_{probe}.csv", index=False)
    import json

    held = {p: sum(v[p] for v in verdicts) for p in ("P1", "P2", "P3", "P4", "P5", "P6")}
    held_x = {p: sum(v[p] for v in verdicts) for p in ("P2_extended", "P4_extended")}
    (C.results_dir(out) / f"predictions_{probe}.json").write_text(
        json.dumps({"per_layer": verdicts, "layers_holding": held,
                    "held_overall": {p: n >= 3 for p, n in held.items()},
                    "extended_persona_layers_holding": held_x,
                    "extended_persona_held_overall": {p: n >= 3 for p, n in held_x.items()}},
                   indent=2, default=float), encoding="utf-8")
    logger.info(f"layers holding: {held}")


def summary_mlp(n_boot: int = 1000, out: str = "phase3", variant: str = "full", layers: str | None = None) -> None:
    """Exploratory MLP probes, protocol B: mean held-out AUROC per projected arm and paired differences of the
    emotion arms against their controls (the same matching as P1: controls at the emotion basis's k)."""
    s = C.load()
    frame = pd.read_csv(C.results_dir(out) / "probes.csv")
    with open(C.data_dir(out) / "scores.pkl", "rb") as f:
        scores = pickle.load(f)
    lb5, subset = LB.lb5(), LB.subset(variant)
    rows = []
    for layer in (s["layers"] if layers is None else C.int_list(layers)):
        data = LB.load(layer, "a", subset)
        arms = Arms(frame, scores, {t: data[t].y for t in lb5}, layer, "a", "mlp", n_boot)
        if arms.b.empty:
            continue
        k_e = {t: arms.chosen_k(t, "emotion_pc") for t in lb5}
        own = lambda fam: {t: arms.boot(t, arms.b[(arms.b["test"] == t) & (arms.b["fs_family"] == fam)  # noqa: E731
                                                  & arms.b["chosen"]]) for t in lb5}
        at_k = lambda name: {t: arms.boot(t, arms.rows(t, name, k_e[t])) for t in lb5}  # noqa: E731
        arm = {"emotion_pc": own("emotion_pc"), "random (k_e)": at_k("random"), "topic_pc (k_e)": at_k("topic_pc"),
               "shuffled_pc (k_e)": at_k("shuffled_pc"), "persona_pub_pc": own("persona_pub_pc"),
               "persona_pub_emotion": {t: arms.boot(t, arms.best_by_inner(t, "persona_pub_emotion")) for t in lb5},
               "persona_pc_extended": own("persona_pc"),
               "persona_emotion_extended": {t: arms.boot(t, arms.best_by_inner(t, "persona_emotion")) for t in lb5},
               "named": own("named"), "alpaca_pc": own("alpaca_pc"), "trainpca_pc": own("trainpca_pc")}
        for n in ("emotion_all", "shuffled_all", "random_all", "alpaca_all", "trainpca_all"):
            arm[n] = {t: arms.boot(t, arms.rows(t, n)) for t in lb5}
        row = {"layer": layer} | {f"mean_{n}": float(np.nanmean([d[t][0] for t in lb5])) for n, d in arm.items()}
        for a, b in (("emotion_pc", "random (k_e)"), ("emotion_pc", "topic_pc (k_e)"), ("emotion_pc", "shuffled_pc (k_e)"),
                     ("emotion_pc", "persona_pub_pc"), ("emotion_pc", "persona_pc_extended"),
                     ("emotion_all", "shuffled_all"), ("emotion_all", "random_all"),
                     ("emotion_all", "alpaca_all"), ("emotion_all", "trainpca_all")):
            d = _delta(arm[a], arm[b], lb5)
            row |= {f"{a} - {b}": d["delta"], f"{a} - {b} lo": d["ci_lo"], f"{a} - {b} hi": d["ci_hi"]}
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.results_dir(out) / "exploratory_mlp.csv", index=False)
    logger.info(f"MLP summary for {len(rows)} layers written")


if __name__ == "__main__":
    fire.Fire({"run": run, "run_mlp": run_mlp, "merge": merge, "summary": summary, "summary_mlp": summary_mlp})
