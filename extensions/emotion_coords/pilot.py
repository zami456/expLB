"""The pilot: every step end to end at small size, then emotion_coords/PILOT_REPORT.md.

Run after the pilot reader chain (`reader/pilot.jobs`) has finished:

    python -m extensions.emotion_coords.pilot checks    # parity, positions, fidelity, D2 formatting, generation, NDIF counts
    python -m extensions.emotion_coords.pilot vectors   # emotion, topic and shuffled bases at layers 20 and 52; sanity checks
    python -m extensions.emotion_coords.pilot probes    # protocols A, B, C and zero-shot on 50 rows per class per set
    python -m extensions.emotion_coords.pilot hpkr      # D1 on every HP-KR row; D2 and D3 on the pilot's 20 questions
    python -m extensions.emotion_coords.pilot summary   # protocol B: mean held-out AUROC, paired bootstraps vs controls

Everything lands in results/emotion_coords/pilot/; emotion_coords/PILOT_REPORT.md is written from it by hand.
Pilot numbers exercise the pipeline and size the full run; they are not results.
"""

import json
import pickle
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import config as C
from extensions.emotion_coords import hpkr as H
from extensions.emotion_coords import lbdata as LB
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import probes as P
from extensions.emotion_coords import vectors as V

OUT = "pilot"
HPKR = "harm_refusal_knowledge_report__plain"
ZERO_SHOT = {"guilty": 1, "ashamed": 1, "remorseful": 1, "nervous": 1, "afraid": 1, "desperate": 1, "calm": -1,
             "compassionate": 1, "empathetic": 1, "sympathetic": 1, "kind": 1, "loving": 1, "protective": 1}


def _out(name: str) -> Path:
    return C.results_dir(OUT) / name


def _layers() -> list[int]:
    return list(C.load()["pilot"]["layers"])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


# --- checks ------------------------------------------------------------------------------------------


def _logprob_reads(name: str) -> tuple[np.ndarray, np.ndarray]:
    ids, lps = [], []
    for path in sorted((C.DATA / "acts" / name).glob("*.npz")):
        with np.load(path) as z:
            ids.append(z["rows"])
            lps.append(z["logprobs"])
    width = max(x.shape[1] for x in lps)
    lps = [np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=np.nan) for x in lps]
    return np.concatenate(ids), np.concatenate(lps)


def checks() -> dict:
    s = C.load()
    out: dict = {}

    # parity: server-side span means of 50 Liars' Bench rows against the stored pooled (a), every layer
    ids, content, points = V.load_reads("lb_parity")
    meta = V.row_meta("lb_parity").loc[ids].reset_index()
    parity = []
    for j, layer in enumerate(s["layers"]):
        stored = {n: np.load(C.pooled_dir(layer) / f"{n}.npz")["pooled"] for n in meta["dataset"].unique()}
        for i, m in meta.iterrows():
            parity.append({"layer": layer, "dataset": m["dataset"], "pooled_row": int(m["pooled_row"]),
                           "cos": _cos(content[i, 0, j], stored[m["dataset"]][int(m["pooled_row"])])})
    parity = pd.DataFrame(parity)
    parity.to_csv(_out("parity_a.csv"), index=False)
    out["parity_a_min_cos_by_layer"] = parity.groupby("layer")["cos"].min().to_dict()

    # positions: the server's pre-response (c) and last-token (b) vectors against the layer-20 cache
    wanted = {n: g["pooled_row"].to_numpy() for n, g in meta.groupby("dataset")}
    cached = LB.cache_vectors(wanted)
    j20 = s["layers"].index(20)
    pts = []
    for i, m in meta.iterrows():
        cv = cached[m["dataset"]]
        k = list(cv["rows"]).index(int(m["pooled_row"]))
        pts.append({"dataset": m["dataset"], "pooled_row": int(m["pooled_row"]),
                    "cos_pre_c": _cos(points[i, 0, j20], cv["c"][k]), "cos_last_b": _cos(points[i, 1, j20], cv["b"][k])})
    pts = pd.DataFrame(pts)
    pts.to_csv(_out("parity_bc_L20.csv"), index=False)
    out["parity_bc_L20_min_cos"] = {"pre_c": float(pts["cos_pre_c"].min()), "last_b": float(pts["cos_last_b"].min())}

    # fidelity: teacher-forced log-probs of the recorded replies, formatting A (cached) and B (no template date)
    fids, lps = _logprob_reads("fidelity")
    fmeta = V.row_meta("fidelity").loc[fids].reset_index()
    rows = []
    for i, m in fmeta.iterrows():
        lp = lps[i][np.isfinite(lps[i])]
        rows.append({"dataset": m["dataset"], "pooled_row": int(m["pooled_row"]), "variant": m["variant"],
                     "label": m["label"], "n_tokens": len(lp) - 1, "mean_logprob": float(lp[:-1].mean()),
                     "eot_logprob": float(lp[-1])})
    fid = pd.DataFrame(rows)
    fid.to_csv(_out("fidelity.csv"), index=False)
    wide = fid.pivot_table(index=["dataset", "pooled_row"], columns="variant", values="mean_logprob")
    out["fidelity"] = {
        "mean_logprob_by_dataset": fid.pivot_table(index="dataset", columns="variant", values="mean_logprob").round(3)
        .to_dict(),
        "eot_logprob_by_dataset": fid.pivot_table(index="dataset", columns="variant", values="eot_logprob").round(3)
        .to_dict(),
        "B_minus_A_mean": float((wide["B"] - wide["A"]).mean()),
        "share_rows_B_higher": float((wide["B"] > wide["A"]).mean()),
    }

    # D2 formatting: a D2 prompt identical to an HP-KR row's prompt must give that row's stored pre-answer vector
    pre_checks = {}
    for layer in (20, 40):
        table = H.preanswer_check(layer)
        if len(table):
            table.to_csv(_out(f"d2_preanswer_check_L{layer}.csv"), index=False)
            pre_checks[layer] = {"pairs": len(table), "min_cos": float(table["cos"].min()),
                                 "median_cos": float(table["cos"].median())}
    out["d2_vs_hpkr_preanswer"] = pre_checks

    # generation, recheck, NDIF requests and time per file
    gen_path = C.DATA / "acts" / "gen_pilot" / "generations.jsonl"
    out["generation"] = [json.loads(line) for line in gen_path.open(encoding="utf-8")] if gen_path.exists() else []
    recheck = C.DATA / "logs" / "recheck_lb_parity.json"
    out["recheck"] = json.loads(recheck.read_text()) if recheck.exists() else None
    ndif = {}
    for name in ("lb_parity", "stories", "neutral", "tylenol", "d2d3", "fidelity"):
        done = C.DATA / "acts" / name / "DONE"
        manifest = json.loads((C.DATA / "rows" / f"{name}.manifest.json").read_text())
        stats = json.loads(done.read_text()) if done.exists() else {}
        ndif[name] = {"rows": manifest["rows"], "tokens": manifest["tokens"], **stats}
        if stats.get("seconds_this_run"):
            ndif[name]["tokens_per_s"] = round(manifest["tokens"] / stats["seconds_this_run"], 1)
    out["ndif"] = ndif
    _json(_out("checks.json"), out)
    logger.info(f"checks: {json.dumps({k: v for k, v in out.items() if k != 'generation'}, default=str)[:1500]}")
    return out


# --- vectors -----------------------------------------------------------------------------------------


def vectors() -> dict:
    out = {}
    seed = C.load()["seeds"]["stories"]
    for layer in _layers():
        stories, neutral = V.Reads.of("stories", layer), V.Reads.of("neutral", layer)
        denoise = V.neutral_basis(neutral)
        e, t = V.emotion_basis(stories, denoise), V.topic_basis(stories, denoise)
        shuffled = V.shuffled_bases(stories, denoise, n=5, seed=seed)
        keep = (stories.meta["rank"] == 0).to_numpy()
        eraser = la.fit_leace(stories.x[keep], stories.meta.loc[keep, "emotion"].to_numpy())
        split = V.split_half(stories, denoise, seed=seed)
        split.to_csv(_out(f"split_half_L{layer}.csv"), index=False)
        dose = V.dose_response(e, layer)
        dose.to_csv(_out(f"dose_response_L{layer}.csv"), index=False)
        arrays = {"E_vectors": e.vectors, "E_components": e.components, "E_explained": e.explained,
                  "E_labels": np.array(e.labels), "T_vectors": t.vectors, "T_components": t.components,
                  "T_labels": np.array(t.labels), "denoise": denoise, "leace_a": eraser.a, "leace_b": eraser.b,
                  "leace_mean": eraser.mean}
        for p, b in enumerate(shuffled):
            arrays[f"S{p}_components"] = b.components
        np.savez(C.data_dir("bases") / f"pilot_L{layer}.npz", **arrays)
        out[layer] = {
            "stories": int(keep.sum()), "neutral_rows": len(neutral.x), "neutral_pcs_removed": len(denoise),
            "emotions": len(e.labels), "topics": len(t.labels), "geometry": V.geometry(e),
            "topic_pc1_var": float(t.explained[0]),
            "split_half": {"median": float(split["cos"].median()), "min": float(split["cos"].min()),
                           "below_0.7": split.loc[split["cos"] < 0.7, "emotion"].tolist()},
            "neighbours": V.neighbours(e, ["afraid", "calm"]),
            "dose_spearman": {em: float(pd.Series(dose[em]).corr(dose["dose_mg"], method="spearman"))
                              for em in ("afraid", "calm")},
            "leace": {"rank": eraser.rank, "shrinkage": eraser.shrinkage},
        }
    _json(_out("vectors.json"), out)
    logger.info(f"vectors: {json.dumps(out, default=str)[:2000]}")
    return out


# --- probes ------------------------------------------------------------------------------------------


def persona_basis(layer: int, bank: str = "extended", model: str | None = None) -> np.ndarray:
    """`extended`: the thesis D4 basis, 25 personas of the 36-role bank (24 PCs). `published`: the persona space of
    arXiv 2605.09391, the 8 deception + 7 honesty personas of the original 26-role bank (14 PCs). `model` (an
    `lb_models` key: qwen, mistral, gemma): the same personas from that model's primary bank."""
    from deception_detection import persona_pcs as pcs

    own = None
    if model is not None:
        from experiments.knowledge import recall_denial as RD

        own = RD.Inputs(model).bank_at(layer)
    if bank == "published":
        b = own if own is not None else pcs.load_bank_at_layer(layer, pcs.VECTORS_DIR)
        n = len(pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS)
        return pcs.fit_geometry(b, k=n - 1).components
    b = own if own is not None else pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    names = pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS + pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS
    geom = pcs.fit_geometry(b, k=len(names) - 1, deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                            honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
    return geom.components


def feature_sets(layer: int, data: dict, bases, hidden: int, n_random: int = 20) -> list[P.FeatureSet]:
    control = C.load()["control"]
    e, t = bases["E_components"], bases["T_components"]
    persona = persona_basis(layer)
    random_ks = tuple(sorted(set(P.KS) | {6}))
    sets = [P.FeatureSet("raw", "raw"),
            P.FeatureSet("emotion_pc", "emotion_pc", e, P.KS),
            P.FeatureSet("topic_pc", "topic_pc", t, P.KS),
            *[P.FeatureSet("shuffled_pc", "shuffled_pc", bases[f"S{p}_components"], P.KS, seed=p) for p in range(5)],
            *[P.FeatureSet("random", "random", la.random_subspace(hidden, 20, seed=10_000 + s), random_ks, seed=s)
              for s in range(n_random)],
            P.FeatureSet("alpaca_pc", "alpaca_pc", la.pca_basis(data[control].x)[0][:20], P.KS),
            P.FeatureSet("trainpca_pc", "trainpca_pc", None, P.KS, per_fold=lambda x: la.pca_basis(x)[0][:20]),
            P.FeatureSet("persona_pc", "persona_pc", persona, P.KS)]
    for k in (3, 5, 10):
        union = la.orthonormal(np.vstack([persona[:k], e[:k]]))
        sets.append(P.FeatureSet("persona_emotion", "persona_emotion", union, (len(union),), seed=k))
    labels = list(bases["E_labels"])
    named = [n for n in C.load()["emotion_names"]["core"] if n in labels]
    sets.append(P.FeatureSet("named", "named", la.unit(bases["E_vectors"][[labels.index(n) for n in named]]),
                             (len(named),)))
    return sets


def erase_sets(bases, hidden: int, n_seeds: int = 10) -> list[P.FeatureSet]:
    e = bases["E_components"]
    top = min(20, len(e))
    eraser = la.Leace(bases["leace_mean"], bases["leace_a"], bases["leace_b"], float("nan"))
    sets = [P.FeatureSet(f"erase_emotion_{k}", "erase_emotion", transform=lambda x, k=k: la.erase_k(x, e, k), seed=k)
            for k in (5, top)]
    sets.append(P.FeatureSet("leace", "erase_emotion", transform=eraser, seed=eraser.rank))
    for rank in sorted({5, top, eraser.rank}):
        for s in range(n_seeds):
            r = la.random_subspace(hidden, rank, seed=20_000 + 100 * rank + s)
            sets.append(P.FeatureSet(f"erase_random_{rank}", "erase_random",
                                     transform=lambda x, r=r: la.erase_k(x, r), seed=s))
    return sets


def probes(n_random: int = 20) -> None:
    from deception_detection.utils import REPO_ROOT
    from experiments.persona.persona_ood import run_probe

    s = C.load()
    lb5, control, train = LB.lb5(), s["control"], s["train_pairs"]
    subset = LB.pilot_subset()
    store, timings = P.Store(), []
    for layer, pooling in [(L, "a") for L in _layers()] + [(20, "b"), (20, "c")]:
        data = LB.load(layer, pooling, subset)
        hidden = data[control].x.shape[1]
        bases = np.load(C.data_dir("bases") / f"pilot_L{layer}.npz")
        sets = feature_sets(layer, data, bases, hidden, n_random)
        for family in ("lr", "dom"):
            raw_c = None
            for fs in sets:
                t0 = time.time()
                P.protocol_a(family, fs, data, train, lb5, control, layer, pooling, store)
                chosen = P.protocol_b(family, fs, data, lb5, control, layer, pooling, store)
                if fs.name == "raw":
                    raw_c = chosen
                P.protocol_c(family, fs, data, lb5, control, layer, pooling, store)
                timings.append({"layer": layer, "pooling": pooling, "probe": family, "feature_set": fs.name,
                                "seed": fs.seed, "seconds": round(time.time() - t0, 2), "protocols": "A,B,C"})
            if pooling == "a":
                for fs in erase_sets(bases, hidden):
                    t0 = time.time()
                    P.protocol_b(family, fs, data, lb5, control, layer, pooling, store,
                                 fixed_c=raw_c if family == "lr" else None)
                    timings.append({"layer": layer, "pooling": pooling, "probe": family, "feature_set": fs.name,
                                    "seed": fs.seed, "seconds": round(time.time() - t0, 2), "protocols": "B"})
            logger.info(f"L{layer}{pooling} {family}: {len(store.rows)} rows so far")
        labels = list(bases["E_labels"])
        P.zero_shot({n: (bases["E_vectors"][labels.index(n)], sign) for n, sign in ZERO_SHOT.items() if n in labels},
                    data, lb5, control, layer, pooling, store)
        if pooling == "a":  # the published per-token RepE probe, scored on pooled (a) (exact, by linearity)
            run = REPO_ROOT / "results" / ("XXXX-9-llama-all" if layer == 20 else f"XXXX-9-llama-all-L{layer}")
            score = run_probe(run)
            ctrl = score(data[control].x)
            for t in lb5:
                store.add({"protocol": "A-published", "feature_set": "raw", "fs_family": "raw", "seed": 0,
                           "layer": layer, "pooling": "a", "train": train, "probe": "lb_mean_probe", "k": hidden,
                           "test": t}, data[t].y, score(data[t].x), ctrl, C=0.1, inner_auroc=np.nan, chosen=True)
    frame = store.frame()
    frame.to_csv(_out("probes.csv"), index=False)
    with open(C.data_dir("pilot") / "scores.pkl", "wb") as f:
        pickle.dump(store.scores, f)
    pd.DataFrame(timings).to_csv(_out("probe_timings.csv"), index=False)
    logger.info(f"probes: {len(frame)} rows, {sum(t['seconds'] for t in timings):.0f} s of fitting")


# --- HP-KR diagnostics -------------------------------------------------------------------------------


def hpkr() -> None:
    from deception_detection.utils import REPO_ROOT
    from experiments.persona.persona_ood import run_probe

    s = C.load()
    lb5 = LB.lb5()
    rows = H.hpkr_rows()
    frame = pd.read_csv(_out("probes.csv"))
    subset = LB.pilot_subset()
    d1 = []
    for layer in _layers():
        full = LB.load(layer, "a")
        x_hpkr = full[HPKR].x
        if len(x_hpkr) != len(rows):
            raise RuntimeError(f"L{layer}: {len(x_hpkr)} HP-KR rows, expected {len(rows)}")
        bases = np.load(C.data_dir("bases") / f"pilot_L{layer}.npz")
        run = REPO_ROOT / "results" / ("XXXX-9-llama-all" if layer == 20 else f"XXXX-9-llama-all-L{layer}")
        scorers = {"A published raw probe": run_probe(run)(x_hpkr)}
        data = LB.load(layer, "a", subset)
        train = [n for n in lb5 if n != HPKR]
        x = np.concatenate([data[n].x for n in train])
        y = np.concatenate([data[n].y for n in train])
        for name, basis in (("raw", None), ("persona_pc", persona_basis(layer)), ("emotion_pc", bases["E_components"])):
            pick = frame[(frame["protocol"] == "B") & (frame["feature_set"] == name) & (frame["test"] == HPKR)
                         & (frame["probe"] == "lr") & (frame["layer"] == layer) & (frame["pooling"] == "a")
                         & frame["chosen"]]
            k, c = int(pick["k"].iloc[0]), float(pick["C"].iloc[0])
            f = (lambda z: z) if basis is None else (lambda z, b=basis[:k]: z @ b.T)
            scorers[f"B {name} (k={k})"] = P.fit("lr", f(x), y, c).score(f(x_hpkr))
        labels = list(bases["E_labels"])
        for n, sign in ZERO_SHOT.items():
            if n in labels:
                scorers[f"zero-shot {n}"] = sign * (x_hpkr @ la.unit(bases["E_vectors"][labels.index(n)]))
        for name, sc in scorers.items():
            for lies in ("all", "explicit"):
                d1.append({"layer": layer} | H.describe(name, sc, rows, lies))
    pd.DataFrame(d1).to_csv(_out("d1.csv"), index=False)

    for layer in _layers():
        bases = np.load(C.data_dir("bases") / f"pilot_L{layer}.npz")
        labels = list(bases["E_labels"])
        emotions = {n: bases["E_vectors"][i] for i, n in enumerate(labels)}
        pre, reply, meta = H.d2d3_acts(layer)
        design = H.Design.of(meta)
        for name, acts in (("d2", pre), ("d3", reply)):
            H.did(acts, design, emotions, bases["T_vectors"]).to_csv(_out(f"{name}_L{layer}.csv"), index=False)
    logger.info("hpkr: d1, d2, d3 written")


def summary() -> None:
    """Protocol B, mean held-out AUROC per feature set and paired bootstrap differences (`compare.py`)."""
    from extensions.emotion_coords import compare

    frame = pd.read_csv(_out("probes.csv"))
    with open(C.data_dir("pilot") / "scores.pkl", "rb") as f:
        scores = pickle.load(f)
    subset = LB.pilot_subset()
    per_set, summaries = [], []
    for layer, pooling in [(L, "a") for L in _layers()] + [(20, "b"), (20, "c")]:
        data = LB.load(layer, pooling, subset)
        ys = {t: data[t].y for t in LB.lb5()}
        for probe in ("lr", "dom"):
            table, summ = compare.protocol_b_table(frame, scores, ys, layer, pooling, probe)
            per_set.append(table.assign(layer=layer, pooling=pooling, probe=probe))
            summaries.append(summ)
    pd.concat(per_set).to_csv(_out("protocol_b_per_set.csv"), index=False)
    pd.concat(summaries).to_csv(_out("protocol_b_summary.csv"), index=False)
    logger.info("summary written")


if __name__ == "__main__":
    fire.Fire({"checks": checks, "vectors": vectors, "probes": probes, "hpkr": hpkr, "summary": summary})
