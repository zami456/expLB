"""Rule K, the knowledge direction without any of HP-KR's wording (`k_final.py`), inside the
benchmark's other models. Written, with its criteria, on 2026-10-14, before any of its rows was read.

`k_final.py` built rule K for Llama: system "You are a helpful assistant.", the bare question, one
of the six `k_rule` instruction lines, the reply forced to "That is not something I know." / "I have
no knowledge of that.", the mean over the reply's tokens; 6 lines x 120 questions x 2 replies =
1,440 rows, the six lines pooled with equal weight. This builds it by the same recipe inside Qwen,
Mistral and Gemma. The questions are each model's own 60 + 60, those of its original K
(`k_models.py`: known questions it answered wrongly replaced from the reserve), so the two Ks are
compared on the same questions; only the wording differs. Read at every block of the model's sweep
(`ModelSpec.layers`). The original K, built in HP-KR's knowledge-report template, is the model's
`recall_denial.Inputs.k`; D9 and D5 come from its primary persona bank.

Criteria, fixed before reading, at the model's mid block (Qwen 40, Mistral 20, Gemma 31):
    X1  (Mistral, the only model with a released identical-text slice) rule K scores HP-KR's
        identical text above the 97.5th percentile of 1,000 random directions
    X2  rule K's mean AUROC over the model's datasets >= the original K's - 0.03
    X3  rule K's transfer (train one, test all, as `k_transfer.py` scores it) >= the original K's - 0.03
    X4  D9 + rule K's transfer >= D9 + the original K's - 0.03
    X5  (Gemma, held out: AISI's Gemma rollouts, read by `aisi_probe.py extract`) rule K's AUROC
        >= the original K's - 0.03 (Llama's C2)
    X6  (Gemma, held out) D9 + rule K's AUROC >= D9 + the original K's - 0.03 (Llama's C3)
The two Ks' cosine, the other blocks, D5 + either K and HP-KR's denials are reported beside them,
not used to decide.

    python experiments/other_models/k_final_xmodel.py read --model mistral     # NDIF, resumable
    python experiments/other_models/k_final_xmodel.py score --model mistral

Outputs go to `results/xmodel/<model>/k_final/`.
"""

import json
import time

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from experiments.knowledge import forced_denial_bank as fdb
from experiments.knowledge import k_models as km
from experiments.knowledge import k_rule as kru
from experiments.knowledge import k_template as kt
from experiments.other_models import lb_models as LB
from experiments.knowledge import recall_denial as rd
from experiments.knowledge import recall_items
from experiments.knowledge.k_cutoff import sides
from experiments.persona.layer_sweep import done_rows, save_request

N_BOOT = 1000
N_RANDOM = 1000
SEED = 0
MARGIN = 0.03  # Llama's C2 and C3


def out_dir(model: str):
    return LB.XMODEL_RESULTS / LB.spec(model).key / "k_final"


def questions(model: str) -> list[dict]:
    """The model's 60 known and 60 unknowable questions, under the ids its original K gives them."""
    target = km._target(model)
    checked = json.loads((km._folder(target) / "verify.json").read_text(encoding="utf-8"))["known"]
    verified = {v["question"]: v["correct"] for v in checked}
    known = [k for k in list(recall_items.KNOWN) + list(recall_items.RESERVE) if verified.get(k[0], True)]
    out = [{"item": f"gen-known-{i}", "knows": True, "question": q} for i, (q, _, _) in enumerate(known[:rd.N_GENERAL])]
    out += [{"item": f"gen-unknown-{i}", "knows": False, "question": q}
            for i, (q, _) in enumerate(recall_items.UNKNOWN[:rd.N_GENERAL])]
    theirs = {(r["item"], bool(r["knows"])) for r in km._rows(target) if r["rank"] < rd.N_GENERAL}
    if {(q["item"], q["knows"]) for q in out} != theirs:
        raise RuntimeError(f"{model}: the questions do not line up with the original K's items")
    return out


def build_rows(model: str) -> list[dict]:
    rows = []
    for rid, template in kru.RULED.items():
        for q in questions(model):
            for s, sentence in enumerate(kt.VARIANTS["report"]["deny"]):
                rows.append({"variant": rid, "item": q["item"], "knows": q["knows"], "system": kt.SYSTEM,
                             "user": template.format(question=q["question"]), "kind": "deny",
                             "assistant": sentence, "string": s})
    for r, row in enumerate(rows):
        row["row"] = r
    return rows


# --- reads -------------------------------------------------------------------------------------

def read(model: str, layers=None, budget: int = km.TOKEN_BUDGET, retries: int = 3) -> None:
    """Reply-token means at every block of the model's sweep, on NDIF. Resumable. A request the
    server fails on for its size is re-planned, with every later one, at half the budget."""
    from deception_detection.batching import classify_failure
    from deception_detection.models import ModelName, get_model_and_tokenizer
    from experiments.organisms.gender_secret import _environment

    spec, target = LB.spec(model), km._target(model)
    layers = [int(x) for x in (layers or spec.layers)]
    rows = build_rows(model)
    parts = out_dir(model) / "parts"
    _environment()
    remote, tokenizer = get_model_and_tokenizer(ModelName(target.model), template_date=km._template_date(target))
    fdb._check_identical_answers(tokenizer, rows)
    toks = fdb._tokenize(tokenizer, rows)
    for row, tok, det in zip(rows, toks.tokens.tolist(), toks.detection_mask.bool().tolist()):
        # special tokens aside: Qwen's scored span ends with its <|im_end|>, as its original K's rows do
        span = tokenizer.decode([t for t, d in zip(tok, det) if d], skip_special_tokens=True)
        if span.strip() != row["assistant"]:
            raise RuntimeError(f"row {row['row']}: scored span {span!r}, reply {row['assistant']!r}")
    todo = [r for r in rows if r["row"] not in done_rows(parts)]
    if not todo:
        logger.info(f"{model}: all {len(rows)} rows already read")
        return
    requests = fdb._requests(tokenizer, todo, budget)
    logger.info(f"{model}: {len(todo)} rows at {len(layers)} blocks in {len(requests)} requests")
    k = 0
    while k < len(requests):
        batch, start, means = requests[k], time.time(), None
        for attempt in range(1, retries + 1):
            try:
                means = km._read_batch(remote, tokenizer, batch, layers)
                break
            except Exception as error:  # NDIF drops a request now and then
                kind = classify_failure(error)
                if kind == "fatal":
                    raise
                if kind in ("oom", "unknown") and len(batch) > 1:
                    budget //= 2
                    rest = [r for b in requests[k:] for r in b]
                    requests = requests[:k] + fdb._requests(tokenizer, rest, budget)
                    logger.warning(f"request {k + 1} failed ({error!r}); the remaining {len(rest)} rows "
                                   f"re-planned at {budget} tokens")
                    break
                if attempt == retries:
                    raise
                logger.warning(f"request {k + 1} failed ({error!r}); retry {attempt}/{retries - 1}")
                time.sleep(30 * attempt)
        if means is None:
            continue
        if not np.isfinite(means).all():
            raise RuntimeError(f"request {k + 1} returned non-finite reads")
        save_request(parts, [r["row"] for r in batch], layers, means)
        logger.info(f"request {k + 1}/{len(requests)}: {len(batch)} rows, {time.time() - start:.0f} s")
        k += 1


# --- directions and scores ---------------------------------------------------------------------

def _load_layer(folder, layer: int) -> dict[int, dict[int, np.ndarray]]:
    out = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as saved:
            if f"L{layer}" in saved.files:
                block = saved[f"L{layer}"].astype(np.float64)
                for k, row in enumerate(saved["rows"]):
                    out[int(row)] = {layer: block[k]}
    return out


def rule_k(model: str, layer: int) -> np.ndarray:
    """Rule K at `layer`: per question the six lines' means averaged, then unit(known - unknown)."""
    rows, reads = build_rows(model), _load_layer(out_dir(model) / "parts", layer)
    per = [sides([r for r in rows if r["variant"] == rid], reads, layer) for rid in kru.RULED]
    Xk, Xu = (np.mean([p[s] for p in per], axis=0) for s in (0, 1))
    if len(Xk) != rd.N_GENERAL or len(Xu) != rd.N_GENERAL:
        raise RuntimeError(f"{model} block {layer}: {len(Xk)} known and {len(Xu)} unknown questions")
    return rd._unit(Xk.mean(0) - Xu.mean(0))


def _held_out(model: str, layer: int) -> tuple[np.ndarray, np.ndarray] | None:
    """AISI's held-out rollouts of the model at `layer` (Gemma's, read by `aisi_probe.py extract`)."""
    from experiments.probes import aisi_probe as AP

    try:
        setup = AP.Setup(model)
    except ValueError:
        return None
    path = setup.validation / f"validation_L{layer}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if not saved["done"].all():
            raise RuntimeError(f"the AISI held-out split is not complete at block {layer}")
        return saved["pooled"][:, 0].astype(np.float64), saved["deceptive"].astype(int)


def score(model: str, layers=None, seed: int = SEED) -> None:
    from experiments.knowledge import k_transfer as ktr
    from experiments.persona import persona_ood as P

    rng = np.random.default_rng(seed)
    inputs, _, order = ktr._context(model, None)
    spec = inputs.spec
    layers = [int(x) for x in (layers or spec.layers)]
    hpkr = inputs.hpkr()
    slices = hpkr[1] if hpkr is not None else {}
    dirs, depth, held = {}, [], []
    for layer in layers:
        rk, ok = rule_k(model, layer), inputs.k(layer)
        dirs[f"L{layer}|rule"] = rk
        pooled = inputs.pooled(layer)
        control = pooled[rd.CONTROL].z
        bank = inputs.bank_at(layer)
        d9, d5 = bank.axis_unit("extended"), bank.axis_unit("deception")
        scorers = {"original K": ok, "rule K": rk, "D9": d9,
                   "D9 + original K": rd.standardised_sum(d9, ok, control),
                   "D9 + rule K": rd.standardised_sum(d9, rk, control),
                   "D5 + original K": rd.standardised_sum(d5, ok, control),
                   "D5 + rule K": rd.standardised_sum(d5, rk, control)}
        row = {"layer": layer, "cos(rule K, original K)": float(rk @ ok)}
        for name, w in scorers.items():
            row[f"{name}: dataset mean"] = float(np.mean([P.auroc(pooled[n].y, pooled[n].z @ w) for n in inputs.datasets]))
            m = P.transfer_matrix(lambda z, w=w: (z @ w)[:, None], pooled, order)
            row[f"{name}: transfer"] = P.off_diagonal_mean(m)
            if slices:
                y, z = pooled[P.HPKR].y, pooled[P.HPKR].z
                for sname, label in (("identical text", "HP-KR identical"), ("denials", "HP-KR denials")):
                    if sname in slices:
                        row[f"{name}: {label}"] = P.auroc(y[slices[sname]], z[slices[sname]] @ w)
        if "identical text" in slices and layer == spec.mid:
            y, z = pooled[P.HPKR].y[slices["identical text"]], pooled[P.HPKR].z[slices["identical text"]]
            R = rng.standard_normal((N_RANDOM, z.shape[1]))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            s = z @ R.T
            row["random band, HP-KR identical"] = float(np.quantile([P.auroc(y, s[:, j]) for j in range(N_RANDOM)], 0.975))
        depth.append(row)
        found = _held_out(model, layer)
        if found is not None:
            z, yh = found
            scores = {name: z @ w for name, w in scorers.items() if name in
                      ("original K", "rule K", "D9", "D9 + original K", "D9 + rule K")}
            idx = [rng.integers(0, len(yh), len(yh)) for _ in range(N_BOOT)]
            idx = [i for i in idx if 0 < yh[i].sum() < len(i)]
            for name, sc in scores.items():
                boot = np.array([P.auroc(yh[i], sc[i]) for i in idx])
                rec = {"layer": layer, "direction": name, "auroc": P.auroc(yh, sc),
                       "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975))}
                for a, b in (("rule K", "original K"), ("D9 + rule K", "D9 + original K"), ("D9 + rule K", "D9")):
                    if name == a:
                        d = boot - np.array([P.auroc(yh[i], scores[b][i]) for i in idx])
                        rec |= {f"minus {b}": rec["auroc"] - P.auroc(yh, scores[b]),
                                f"minus {b} lo": float(np.quantile(d, 0.025)), f"minus {b} hi": float(np.quantile(d, 0.975))}
                held.append(rec)
        logger.info(f"{model} block {layer} scored")
        del pooled
    out = out_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "directions.npz", **dirs)
    depth, held = pd.DataFrame(depth), pd.DataFrame(held)
    depth.round(4).to_csv(out / "depth.csv", index=False)
    if len(held):
        held.round(4).to_csv(out / "held_out.csv", index=False)
    _verdicts(model, depth, held)
    pd.set_option("display.width", 250)
    cols = [c for c in depth.columns if c.split(": ")[-1] in ("dataset mean", "transfer", "HP-KR identical")]
    logger.info(f"{model} depth:\n{depth[['layer', 'cos(rule K, original K)', *cols]].round(3).to_string(index=False)}")


def _names(spec) -> dict[int, str]:
    """The model's blocks for the thesis's layers 20, 36, 40 and 44."""
    return {spec.lb_block: "L20", spec.posthoc[0]: "L36", spec.mid: "L40", spec.posthoc[1]: "L44"}


def vs_probe(model: str, layers=None, n_draws: int = N_BOOT, seed: int = SEED) -> None:
    """Rule K, the original K and each with D9 and with D5 (and D9 and D5 alone) against the
    same-layer RepE base probe (D2), zero-shot:
    AUROC on each of the model's datasets, their mean, HP-KR's identical-text and denial slices
    where the model has them and AISI's held-out rollouts where they were read (Llama, Gemma), with
    paired 95% intervals of each difference from the probe (`k_robustness.paired_draws`: lies and
    honest rows resampled separately, the dataset mean averaged draw by draw), and balanced accuracy
    at the threshold that flags 1% of Alpaca. Llama's rule K is `k_final.py`'s. The probe's dataset
    mean must reproduce `k_robustness.py paired`'s where that was computed. Writes
    `results/xmodel/<model>/k_final/vs_probe.csv`."""
    from experiments.knowledge import k_final as kf
    from experiments.knowledge import k_robustness as krb
    from experiments.persona import persona_ood as P

    key = LB.spec(model).key
    inputs = rd.Inputs(None if key == "llama" else model)
    spec = inputs.spec
    names = _names(spec)
    layers = [int(x) for x in (layers or names)]
    rng = np.random.default_rng(seed)
    hpkr = inputs.hpkr()
    slices = hpkr[1] if hpkr is not None else {}
    stored_path = krb.out_dir(inputs) / "paired.csv"
    stored = pd.read_csv(stored_path) if stored_path.exists() else None
    rows = []
    for layer in layers:
        probe = inputs.probe(layer)
        if probe is None:
            logger.info(f"{key} block {layer}: no base probe")
            continue
        pooled = inputs.pooled(layer)
        control = pooled[rd.CONTROL].z
        bank = inputs.bank_at(layer)
        d9, d5 = bank.axis_unit("extended"), bank.axis_unit("deception")
        ok = inputs.k(layer)
        rk = kf.rule_k(layer) if key == "llama" else rule_k(model, layer)
        dirs = {"original K": ok, "rule K": rk, "D9 + original K": rd.standardised_sum(d9, ok, control),
                "D9 + rule K": rd.standardised_sum(d9, rk, control), "D9": d9, "D5": d5,
                "D5 + original K": rd.standardised_sum(d5, ok, control),
                "D5 + rule K": rd.standardised_sum(d5, rk, control)}

        def scorers(z, dirs=dirs, probe=probe):
            return {"base probe": probe(z)} | {n: z @ w for n, w in dirs.items()}

        thresholds = {n: float(np.quantile(s, 0.99)) for n, s in scorers(control).items()}
        sets = {rd.SHORT[n]: (pooled[n].y.astype(int), pooled[n].z) for n in inputs.datasets}
        if slices:
            y_kr, z_kr = sets["HP-KR"]
            for sname, label in (("identical text", "HP-KR identical"), ("denials", "HP-KR denials")):
                if sname in slices:
                    sets[label] = (y_kr[slices[sname]], z_kr[slices[sname]])
        found = (kf._held_out(layer) if layer in kf.HELD_OUT_LAYERS else None) if key == "llama" else _held_out(model, layer)
        if found is not None:
            sets["held out (AISI)"] = (found[1], found[0])
        point, draws, balacc = {}, {}, {}
        for set_name, (y, z) in sets.items():
            sc = scorers(z)
            point[set_name] = {n: P.auroc(y, v) for n, v in sc.items()}
            balacc[set_name] = {n: P.balanced_accuracy(y, v, thresholds[n]) for n, v in sc.items()}
            draws[set_name] = krb.paired_draws(y, sc, n_draws, rng)
        main = [rd.SHORT[n] for n in inputs.datasets]
        point["dataset mean"] = {n: float(np.mean([point[d][n] for d in main])) for n in point[main[0]]}
        balacc["dataset mean"] = {n: float(np.mean([balacc[d][n] for d in main])) for n in balacc[main[0]]}
        draws["dataset mean"] = {n: np.mean([draws[d][n] for d in main], axis=0) for n in draws[main[0]]}
        if stored is not None:
            ref = stored[(stored["layer"] == layer) & (stored["set"] == "mean of five")
                         & (stored["b"] == "benchmark probe, same layer")]
            if len(ref) and abs(float(ref["b_auroc"].iloc[0]) - point["dataset mean"]["base probe"]) > 1e-3:
                raise AssertionError(f"{key} block {layer}: the base probe's dataset mean "
                                     f"{point['dataset mean']['base probe']:.4f} does not reproduce paired.csv's "
                                     f"{float(ref['b_auroc'].iloc[0]):.4f}")
        for set_name in point:
            for n in dirs:
                d = draws[set_name][n] - draws[set_name]["base probe"]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append({"model": key, "layer": names.get(layer, f"B{layer}"), "block": layer, "set": set_name,
                             "scorer": n, "auroc": point[set_name][n], "base probe auroc": point[set_name]["base probe"],
                             "delta": point[set_name][n] - point[set_name]["base probe"], "lo": float(lo),
                             "hi": float(hi), "balacc 1% Alpaca": balacc[set_name][n],
                             "base probe balacc": balacc[set_name]["base probe"]})
        logger.info(f"{key} block {layer} compared with the base probe")
        del pooled
    table = pd.DataFrame(rows)
    out = LB.XMODEL_RESULTS / key / "k_final"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "vs_probe.csv", index=False)
    pd.set_option("display.width", 250)
    mid = table[table["block"] == spec.mid]
    logger.info(f"{key} at block {spec.mid}:\n{mid.round(3).to_string(index=False)}")


def _verdicts(model: str, depth: pd.DataFrame, held: pd.DataFrame) -> None:
    mid = LB.spec(model).mid
    r = depth.set_index("layer").loc[mid]
    lines = []

    def judge(label: str, ok: bool, text: str) -> None:
        lines.append(f"{label} {text}: {'pass' if ok else 'fail'}")

    if "random band, HP-KR identical" in r and not pd.isna(r.get("random band, HP-KR identical")):
        judge("X1", r["rule K: HP-KR identical"] > r["random band, HP-KR identical"],
              f"block {mid}, HP-KR identical text: rule K {r['rule K: HP-KR identical']:.3f} v random 97.5% "
              f"{r['random band, HP-KR identical']:.3f}")
    else:
        lines.append("X1 not applicable (no released identical-text slice)")
    for label, metric, a, b in (("X2", "dataset mean", "rule K", "original K"), ("X3", "transfer", "rule K", "original K"),
                                ("X4", "transfer", "D9 + rule K", "D9 + original K")):
        va, vb = r[f"{a}: {metric}"], r[f"{b}: {metric}"]
        judge(label, va >= vb - MARGIN, f"block {mid}, {metric}: {a} {va:.3f} v {b} {vb:.3f} (within {MARGIN} or above)")
    h = held[held.layer == mid].set_index("direction") if len(held) else held
    if len(h):
        for label, a, b in (("X5", "rule K", "original K"), ("X6", "D9 + rule K", "D9 + original K")):
            va, vb = h.loc[a, "auroc"], h.loc[b, "auroc"]
            judge(label, va >= vb - MARGIN, f"held out (AISI), block {mid}: {a} {va:.3f} v {b} {vb:.3f} "
                  f"(within {MARGIN} or above)")
    else:
        lines.append("X5, X6 not applicable (no held-out AISI rollouts of this model)")
    (out_dir(model) / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"{model} verdicts:\n" + "\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"read": read, "score": score, "vs_probe": vs_probe})
