"""Arm A on their released activations (CPU): nulls and controls for their probe, through their aggregators.

- `random_null` (A2): 1,000 random unit directions per layer in their standardised space (their
  scaler, fitted on their training rows, seed 0), scored with Peak Suspicion, Deception Split and the
  Asymmetry Probe (refitted per direction on the same run labels), over their leave-domain-out Core
  folds and their zero-shot Transfer and Stego groups. Each draw's value is the mean over their layers,
  as their reported numbers are. Activation Alignment uses no direction, so it has no direction null;
  Sorted Concatenation (a PCA per fold and draw) is left out of the null for cost.
- `pseudo_null` (A3): their probe recipe trained on Core *control* runs, where no coalition exists,
  with a random pair of each run labelled colluder; zero-shot to Transfer and Stego as theirs is.
- `sensitivity` (A4): their pipeline with a converged probe (max_iter 1000) and with v1's C = 1.0.
- `stance` (A6): their leave-domain-out probe scores per Core agent, by role x stance (the agent's
  vote on the colluders' target option), from the metadata of their own runs.
- `stego_corrected`: their probe on Stego with each agent's hands 1-7 averaged, instead of the
  final-hand rows whose replies are empty in their extraction (gate G1).
"""

import json
from pathlib import Path

import fire
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from extensions.narcbench import rose as RO
from extensions.narcbench import spec as S

DIRECTED = ("peak_suspicion", "deception_split", "asymmetry_probe")


def auroc_cols(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC of each column of `scores` (n, k) against binary `y` (n,), ties averaged."""
    ranks = rankdata(scores, axis=0)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def their_scaler(meta, npz, layer, exclude_domain=None, seed=0) -> StandardScaler:
    """The scaler their probe is trained with (same rows, same subsample)."""
    acts = RO.get_layer(npz, len(meta), layer)
    col, hon = RO.training_rows(meta, "discussion", exclude_domain)
    nc = min(len(col), len(hon))
    rng = np.random.RandomState(seed)
    hon = list(rng.choice(hon, nc, replace=False))
    return StandardScaler().fit(np.vstack([acts[hon], acts[col[:nc]]]))


def _projections(groups, rids, scaler, W) -> np.ndarray:
    """(runs, 4, draws) projections of each run's agents (sorted by name, as theirs) on each direction."""
    vecs = np.stack([np.stack([groups[r]["agents"][n] for n in sorted(groups[r]["agents"])]) for r in rids])
    z = scaler.transform(vecs.reshape(-1, vecs.shape[-1]))
    return (z @ W).reshape(len(rids), 4, -1)


def _features(P: np.ndarray) -> dict:
    s = -np.sort(-P, axis=1)  # descending per run and draw
    split = s[:, :2].mean(1) - s[:, 2:].mean(1)
    asym = np.stack([P.var(1), s[:, 0] - s[:, 3], split, P.mean(1)], axis=1)  # (runs, 4, draws)
    return {"peak_suspicion": s[:, 0], "deception_split": split, "asymmetry_probe": asym}


def _asym_auroc(f_tr, y_tr, f_te, y_te, hp=RO.HP()) -> np.ndarray:
    out = np.empty(f_tr.shape[-1])
    for j in range(f_tr.shape[-1]):
        fs = StandardScaler().fit(f_tr[:, :, j])
        clf = LogisticRegression(max_iter=hp.clf_iter, C=hp.c_clf, penalty=hp.penalty, solver=hp.solver,
                                 random_state=42).fit(fs.transform(f_tr[:, :, j]), y_tr)
        p = clf.predict_proba(fs.transform(f_te[:, :, j]))[:, 1]
        out[j] = auroc_cols(y_te, p[:, None])[0]
    return out


def _directed(f_tr, y_tr, f_te, y_te) -> dict:
    return {"peak_suspicion": auroc_cols(y_te, f_te["peak_suspicion"]),
            "deception_split": auroc_cols(y_te, f_te["deception_split"]),
            "asymmetry_probe": _asym_auroc(f_tr["asymmetry_probe"], y_tr, f_te["asymmetry_probe"], y_te)}


def random_null(model: str = "llama_3_1_70b", draws: int = 1000, seed: int = 0) -> None:
    data = RO.load_released(model)
    layers = RO.RELEASED_LAYERS[model]
    (cm, cn), (tm, tn), (sm, sn) = data["core"], data["transfer"], data["stego"]
    per_layer = []
    for layer in layers:
        dim = RO.get_layer(cn, 1, layer).shape[1]
        rng = np.random.default_rng([seed, layer])
        W = rng.standard_normal((dim, draws))
        W /= np.linalg.norm(W, axis=0)
        out = {}
        groups = RO.build_groups(cm, cn, layer, "discussion", True)
        rids = sorted(groups)
        domains = sorted({groups[r]["domain"] for r in rids if groups[r]["domain"]})
        folds = {m: [] for m in DIRECTED}
        for d in domains:
            tr = [r for r in rids if groups[r]["domain"] != d]
            te = [r for r in rids if groups[r]["domain"] == d]
            y_tr, y_te = RO._run_labels(groups, tr), RO._run_labels(groups, te)
            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue
            sc = their_scaler(cm, cn, layer, exclude_domain=d)
            res = _directed(_features(_projections(groups, tr, sc, W)), y_tr,
                            _features(_projections(groups, te, sc, W)), y_te)
            for m in DIRECTED:
                folds[m].append(res[m])
        out["Held-out"] = {m: np.mean(folds[m], axis=0) for m in DIRECTED}
        sc = their_scaler(cm, cn, layer)
        y_core = RO._run_labels(groups, rids)
        f_core = _features(_projections(groups, rids, sc, W))
        tg = RO.build_groups(tm, tn, layer, "final", False)
        trids = sorted(tg)
        fams = sorted({tg[r]["scenario"] for r in trids})
        for fam in fams:
            sub = [r for r in trids if tg[r]["scenario"] == fam]
            out[fam] = _directed(f_core, y_core, _features(_projections(tg, sub, sc, W)), RO._run_labels(tg, sub))
        for name, phase in (("blackjack", "final"), ("blackjack_corrected", "discussion")):
            sg = RO.build_groups(sm, sn, layer, phase, phase == "discussion")
            srids = sorted(sg)
            out[name] = _directed(f_core, y_core, _features(_projections(sg, srids, sc, W)),
                                  RO._run_labels(sg, srids))
        per_layer.append(out)
        print(f"layer {layer} done", flush=True)
    groups_out = list(per_layer[0])
    null = {g: {m: np.mean([pl[g][m] for pl in per_layer], axis=0) for m in DIRECTED} for g in groups_out}
    summary = {g: {m: {"band97.5": float(np.percentile(v, 97.5)), "median": float(np.median(v)),
                       "max": float(v.max())} for m, v in d.items()} for g, d in null.items()}
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(S.RESULTS / f"armA_random_null_{model}.npz",
                        **{f"{g}|{m}": v for g, d in null.items() for m, v in d.items()})
    (S.RESULTS / f"armA_random_null_{model}.json").write_text(
        json.dumps({"model": model, "draws": draws, "layers": list(layers), "summary": summary}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=1))


def pseudo_direction(draw: int, hp=RO.HP()):
    """Their probe recipe on Core control runs with a random pair per run labelled colluder."""
    def fn(meta, npz, layer, exclude_domain=None, seed=0):
        acts = RO.get_layer(npz, len(meta), layer)
        runs = sorted({m["run_id"] for m in meta if m["mode"] == "control"})
        rng = np.random.RandomState(10_000 + draw)
        pairs = {}
        for rid in runs:
            agents = sorted({m["agent_name"] for m in meta if m["run_id"] == rid})
            pairs[rid] = set(rng.choice(agents, 2, replace=False))
        idx, y = [], []
        for i, m in enumerate(meta):
            if m["mode"] != "control" or m["channel"] != "public" or m["phase"] != "discussion":
                continue
            _, sid = RO._parse_run_id(m["run_id"])
            if exclude_domain and RO._domain_from_scenario_id(sid) == exclude_domain:
                continue
            idx.append(i)
            y.append(int(m["agent_name"] in pairs[m["run_id"]]))
        scaler = StandardScaler()
        X = scaler.fit_transform(acts[idx])
        probe = LogisticRegression(max_iter=hp.max_iter, C=hp.c_probe, penalty=hp.penalty, solver=hp.solver,
                                   random_state=seed).fit(X, np.array(y))
        w = probe.coef_[0]
        return scaler, w / np.linalg.norm(w)
    return fn


def pseudo_null(model: str = "llama_3_1_70b", draws: int = 50) -> None:
    data = RO.load_released(model)
    layers = list(RO.RELEASED_LAYERS[model])
    methods = ["peak_suspicion", "deception_split", "activation_alignment", "asymmetry_probe"]
    vals = []
    for d in range(draws):
        groups, res = RO.compute_all_results(data, layers, seed=0, direction_fn=pseudo_direction(d),
                                             methods=methods, core_cv=False)
        vals.append({g: {m: float(np.mean(res[g][m])) for m in methods} for g in groups})
        print(f"draw {d} done", flush=True)
    keys = list(vals[0])
    summary = {g: {m: {"band97.5": float(np.percentile([v[g][m] for v in vals], 97.5)),
                       "median": float(np.median([v[g][m] for v in vals]))} for m in methods} for g in keys}
    (S.RESULTS / f"armA_pseudo_null_{model}.json").write_text(
        json.dumps({"model": model, "draws": draws, "values": vals, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def sensitivity(model: str = "llama_3_1_70b", seeds: int = 3) -> None:
    data = RO.load_released(model)
    layers = list(RO.RELEASED_LAYERS[model])
    out = {}
    for name, hp in (("theirs", RO.HP()), ("converged", RO.HP(max_iter=1000)), ("v1_C1", RO.HP(c_probe=1.0))):
        raws, groups = [], None
        for seed in range(seeds):
            groups, raw = RO.compute_all_results(data, layers, seed=seed, hp=hp)
            raws.append(raw)
        means, overall = RO.summarise(raws, groups)
        out[name] = {"means": means, "overall": overall}
        print(name, f"{overall:.4f}", flush=True)
    (S.RESULTS / f"armA_sensitivity_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def stego_corrected(model: str = "llama_3_1_70b", seeds: int = 10) -> None:
    data = RO.load_released(model)
    layers = list(RO.RELEASED_LAYERS[model])
    raws, groups = [], None
    for seed in range(seeds):
        groups, raw = RO.compute_all_results(data, layers, seed=seed, stego_phase="discussion", core_cv=False)
        raws.append(raw)
    means, _ = RO.summarise(raws, groups)
    (S.RESULTS / f"armA_stego_corrected_{model}.json").write_text(
        json.dumps({"model": model, "blackjack_hands_1_7": means["blackjack"]}, indent=1), encoding="utf-8")
    print(json.dumps(means["blackjack"], indent=1))


def stance(model: str = "llama_3_1_70b", seed: int = 0) -> None:
    """Role x stance of Core agents under their leave-domain-out probe (mean projection over their layers)."""
    meta, npz = RO.load_released(model)["core"]
    layers = RO.RELEASED_LAYERS[model]
    score = {}
    for layer in layers:
        groups = RO.build_groups(meta, npz, layer, "discussion", True)
        for d in sorted({g["domain"] for g in groups.values()}):
            scaler, w = RO.train_deception_probe(meta, npz, layer, exclude_domain=d, seed=seed)
            for rid, g in groups.items():
                if g["domain"] != d:
                    continue
                for agent, v in g["agents"].items():
                    score.setdefault((rid, agent), []).append(float(w @ scaler.transform(v[None])[0]))
    info = {}
    for m in meta:
        info[(m["run_id"], m["agent_name"])] = (m["mode"], m["role"], m.get("agent_vote"), m.get("target_option"),
                                               m.get("fair_option"))
    cells = {}
    for key, vals in score.items():
        mode, role, vote, target, fair = info[key]
        stance_ = "target" if vote == target else "fair" if vote == fair else "other"
        cells.setdefault(f"{mode}/{role}/{stance_}", []).append(float(np.mean(vals)))
    summary = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(cells.items())}
    a = cells.get("collusion/honest/target", [])
    b = cells.get("collusion/colluder/fair", [])
    if a and b:
        y = np.array([1] * len(a) + [0] * len(b))
        summary["AUROC honest-voting-target over colluder-voting-fair"] = float(
            auroc_cols(y, np.array(a + b)[:, None])[0])
    (S.RESULTS / f"armA_stance_{model}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    fire.Fire({"random_null": random_null, "pseudo_null": pseudo_null, "sensitivity": sensitivity,
               "stego_corrected": stego_corrected, "stance": stance})
