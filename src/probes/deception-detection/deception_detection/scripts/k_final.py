"""Rule K, built in full: our own wording, all 60 + 60 questions, every sweep layer; checked on
held-out lies. Written, with its criteria, before any of its rows was read.

`k_rule.py` built K without HP-KR's wording from six instruction lines that ask whether the model
knows the answer and limit the reply to that ("rule K"). It met criteria fixed before reading
everywhere but Gender-Secret, on 20 + 20 questions at layers 36, 40 and 44. This builds it as the
original K is built: the same 60 known and 60 unknowable questions (`recall_items.py`, checked by
`recall_denial.py verify`), the same answer-token means, at all 16 sweep layers
(`layer_sweep.ALL_LAYERS`). Wording: system "You are a helpful assistant.", the bare question, one of
the six `k_rule` lines, replies forced to "That is not something I know." / "I have no knowledge of
that.". 6 lines x 120 questions x 2 replies = 1,440 rows; K pools the six lines with equal weight.

The rule was found by trying wordings on CG, HP-C, HP-KR, IT, ID and Gender-Secret, so those are
not a blind test of it. AISI's Llama rollouts (`aisi_probe.py`, the held-out split: 2,228 of
Llama's own belief-checked lies and honest answers about trivia and multiple-choice questions) were
not used by any of these runs and serve as the held-out check, at the layers they were read at
(20, 40, 44).

Criteria, fixed before reading:
    C1  the rebuild matches k_rule: cosine with k_rule's 20 + 20 K at least 0.99 at 36, 40 and 44
    C2  held out, layer 40: rule K's AUROC on the AISI rollouts at least the original K's - 0.03
    C3  held out, layer 40: D9 + rule K at least D9 + the original K - 0.03
Layers 20 and 44 and the depth tables (both Ks alone and with D9: HP-KR identical text, denials,
the five-dataset mean, train-one-test-all transfer as `k_transfer.py` scores it, Gender-Secret
where it was read) are reported beside them, not used to decide.

    python deception_detection/scripts/k_final.py read        # NDIF, resumable
    python deception_detection/scripts/k_final.py score

Outputs go to `results/recall_denial/k_final/` (the directions in `directions.npz`).
"""

import json
import time

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import forced_denial_bank as fdb
from deception_detection.scripts import k_rule as kru
from deception_detection.scripts import k_template as kt
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts import recall_items
from deception_detection.scripts.k_cutoff import sides
from deception_detection.scripts.layer_sweep import ALL_LAYERS, done_rows, pooled_dir, save_request

OUT = rd.OUT / "k_final"
PARTS = OUT / "parts"
CHECK_LAYERS = (36, 40, 44)  # compared with k_rule's reads of the same rows
HELD_OUT_LAYERS = (20, 40, 44)
GS_LAYERS = (20, 36, 40, 44)
CRITERIA_LAYER = 40
N_BOOT = 1000
SEED = 0
FIVE = tuple(rd.SHORT[n] for n in kt.FIVE)


def questions() -> list[dict]:
    """All 60 known and 60 unknowable questions, under the ids `recall_denial` gives them."""
    verified = {v["question"]: v["correct"]
                for v in json.loads((rd.OUT / "verify.json").read_text(encoding="utf-8"))["known"]}
    known = [k for k in list(recall_items.KNOWN) + list(recall_items.RESERVE) if verified.get(k[0], True)]
    out = [{"item": f"gen-known-{i}", "knows": True, "question": q} for i, (q, _, _) in enumerate(known[:rd.N_GENERAL])]
    out += [{"item": f"gen-unknown-{i}", "knows": False, "question": q}
            for i, (q, _) in enumerate(recall_items.UNKNOWN[:rd.N_GENERAL])]
    return out


def build_rows() -> list[dict]:
    rows = []
    for rid, template in kru.RULED.items():
        for q in questions():
            for s, sentence in enumerate(kt.VARIANTS["report"]["deny"]):
                rows.append({"variant": rid, "item": q["item"], "knows": q["knows"], "system": kt.SYSTEM,
                             "user": template.format(question=q["question"]), "assistant": sentence, "string": s})
    for r, row in enumerate(rows):
        row["row"] = r
    return rows


# --- reads -------------------------------------------------------------------------------------

def read(layers=ALL_LAYERS, retries: int = 3) -> None:
    """Answer-token means at every sweep layer, resumable. The first request is compared with
    k_rule's reads of the same rows before anything is saved."""
    from deception_detection.models import get_model_and_tokenizer
    from deception_detection.scripts.gender_secret import _environment

    layers = [int(x) for x in layers]
    rows = build_rows()
    PARTS.mkdir(parents=True, exist_ok=True)
    _environment()
    model, tokenizer = get_model_and_tokenizer("llama-70b-3.3")
    toks = fdb._tokenize(tokenizer, rows)
    for row, tok, det in zip(rows, toks.tokens.tolist(), toks.detection_mask.bool().tolist()):
        span = tokenizer.decode([t for t, d in zip(tok, det) if d])
        if span.strip() != row["assistant"]:
            raise RuntimeError(f"row {row['row']}: scored span {span!r}, reply {row['assistant']!r}")
    earlier = {(r["variant"], r["item"], r["string"]): r["row"] for r in kru.build_rows()}
    earlier_reads = rd._load(kru.OUT / "parts")
    check = OUT / "check_reread.json"
    todo = [r for r in rows if r["row"] not in done_rows(PARTS)]
    if not todo:
        logger.info("all rows already read")
        return
    # put rows k_rule already read first, so the first request can be checked against them
    todo.sort(key=lambda r: (r["variant"], r["item"], r["string"]) not in earlier)
    requests = fdb._requests(tokenizer, todo)
    requests.sort(key=lambda b: not any((r["variant"], r["item"], r["string"]) in earlier for r in b))
    logger.info(f"{len(todo)} rows at {len(layers)} layers in {len(requests)} requests")
    for k, batch in enumerate(requests):
        start = time.time()
        for attempt in range(1, retries + 1):
            try:
                means = fdb._read_batch(model, tokenizer, batch, "answer", layers)
                break
            except Exception as error:  # NDIF drops a request now and then
                if attempt == retries:
                    raise
                logger.warning(f"request {k + 1} failed ({error!r}); retry {attempt}/{retries - 1}")
                time.sleep(30 * attempt)
        if not check.exists():
            pairs = [(i, earlier[(r["variant"], r["item"], r["string"])]) for i, r in enumerate(batch)
                     if (r["variant"], r["item"], r["string"]) in earlier]
            if not pairs:
                raise RuntimeError("the first request holds no row k_rule read")
            worst = {}
            for L in CHECK_LAYERS:
                a = means[[i for i, _ in pairs], layers.index(L)].astype(np.float64)
                b = np.stack([earlier_reads[j][L] for _, j in pairs])
                worst[L] = float((np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).min())
            if min(worst.values()) < 0.9998:
                raise RuntimeError(f"the new reads differ from k_rule's: smallest cosines {worst}")
            check.write_text(json.dumps({"rows": len(pairs), "smallest cosine": worst}) + "\n", encoding="utf-8")
            logger.info(f"re-read check passed on {len(pairs)} rows: smallest cosines {worst}")
        save_request(PARTS, [r["row"] for r in batch], layers, means)
        logger.info(f"request {k + 1}/{len(requests)}: {len(batch)} rows, {time.time() - start:.0f} s")


# --- directions and scores ---------------------------------------------------------------------

def _load_layer(folder, layer: int) -> dict[int, dict[int, np.ndarray]]:
    """`recall_denial._load` for one layer only: {row: {layer: vector}}."""
    out = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as saved:
            if f"L{layer}" in saved.files:
                block = saved[f"L{layer}"].astype(np.float64)
                for k, row in enumerate(saved["rows"]):
                    out[int(row)] = {layer: block[k]}
    return out


def rule_k(layer: int) -> np.ndarray:
    """Rule K at `layer`: per question the six lines' means averaged, then unit(known - unknown)."""
    rows, reads = build_rows(), _load_layer(PARTS, layer)
    per = [sides([r for r in rows if r["variant"] == rid], reads, layer) for rid in kru.RULED]
    Xk, Xu = (np.mean([p[s] for p in per], axis=0) for s in (0, 1))
    if len(Xk) != rd.N_GENERAL or len(Xu) != rd.N_GENERAL:
        raise RuntimeError(f"layer {layer}: {len(Xk)} known and {len(Xu)} unknown questions")
    return rd._unit(Xk.mean(0) - Xu.mean(0))


def original_k(layer: int) -> np.ndarray:
    """The original K at `layer`, rebuilt from its stored reads as the depth analysis does."""
    from deception_detection.scripts import k_robustness as krb

    X, known, _ = krb.item_means(layer)
    return rd._unit(X.T @ krb.weights(known, np.flatnonzero(known), np.flatnonzero(~known)))


def _held_out(layer: int) -> tuple[np.ndarray, np.ndarray]:
    from deception_detection.scripts import aisi_probe as AP

    with np.load(AP.VALIDATION_MEANS / f"validation_L{layer}.npz", allow_pickle=False) as saved:
        if not saved["done"].all():
            raise RuntimeError(f"the AISI held-out split is not complete at layer {layer}")
        return saved["pooled"][:, 0].astype(np.float64), saved["deceptive"].astype(int)


def score(layers=ALL_LAYERS, seed: int = SEED) -> None:
    from deception_detection import persona_pcs as pcs
    from deception_detection.scripts import k_transfer as ktr
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.knowledge_bank import _slices

    rng = np.random.default_rng(seed)
    layers = [int(x) for x in layers]
    stored = dict(np.load(rd.OUT / "directions.npz"))
    frame = hpkr_frame()
    slices = _slices(frame)
    without = [n for n in ktr.ORDER if n != P.HPKR]
    dirs, depth, held, checks = {}, [], [], []
    for layer in layers:
        k_rule_full, k_orig = rule_k(layer), original_k(layer)
        key = f"L{layer}|none|general"
        if key in stored and not np.allclose(k_orig, stored[key], atol=1e-6):
            raise AssertionError(f"layer {layer}: rebuilt original K differs from the stored one")
        dirs[f"L{layer}|rule"] = k_rule_full
        pooled = P.load_pooled(pooled_dir(layer))
        control = pooled[rd.CONTROL].z
        d9 = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR).axis_unit("extended")
        scorers = {"original K": k_orig, "rule K": k_rule_full, "D9": d9,
                   "D9 + original K": rd.standardised_sum(d9, k_orig, control),
                   "D9 + rule K": rd.standardised_sum(d9, k_rule_full, control)}
        row = {"layer": layer, "cos(rule K, original K)": float(k_rule_full @ k_orig)}
        y = pooled[P.HPKR].y
        if not (frame["lie"].to_numpy() == y).all():
            raise AssertionError(f"layer {layer}: HP-KR rows do not line up with hpkr_frame")
        for name, w in scorers.items():
            s = pooled[P.HPKR].z @ w
            row[f"{name}: HP-KR identical"] = P.auroc(y[slices["identical text"]], s[slices["identical text"]])
            row[f"{name}: HP-KR denials"] = P.auroc(y[slices["denials"]], s[slices["denials"]])
            row[f"{name}: mean of five"] = float(np.mean([P.auroc(pooled[n].y, pooled[n].z @ w) for n in kt.FIVE]))
            m = P.transfer_matrix(lambda z, w=w: (z @ w)[:, None], pooled, ktr.ORDER)
            row[f"{name}: transfer"] = P.off_diagonal_mean(m)
            row[f"{name}: transfer without HP-KR"] = P.off_diagonal_mean(m.loc[without, without])
            if layer in GS_LAYERS:
                gs = rd._gs_reads(layer)["adapted"]
                row[f"{name}: Gender-Secret"] = P.auroc((gs["labels"].astype(str) == "DECEPTIVE").astype(int),
                                                        gs["pooled"].astype(np.float64) @ w)
        depth.append(row)
        if layer in CHECK_LAYERS:
            small = kru._arrays(kru._sources(), layer)[kru.POOLED]
            k_small = rd._unit(small[0].mean(0) - small[1].mean(0))
            checks.append({"layer": layer, "cos(rule K 60+60, k_rule 20+20)": float(k_rule_full @ k_small)})
        if layer in HELD_OUT_LAYERS:
            z, yh = _held_out(layer)
            scores = {name: z @ w for name, w in scorers.items()}
            idx = [rng.integers(0, len(yh), len(yh)) for _ in range(N_BOOT)]
            idx = [i for i in idx if 0 < yh[i].sum() < len(i)]
            for name, s in scores.items():
                boot = np.array([P.auroc(yh[i], s[i]) for i in idx])
                rec = {"layer": layer, "direction": name, "auroc": P.auroc(yh, s),
                       "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
                       "balacc_1pct_fpr": P.balanced_accuracy(yh, s, float(np.quantile(control @ scorers[name], 0.99)))}
                for a, b in (("rule K", "original K"), ("D9 + rule K", "D9 + original K"), ("D9 + rule K", "D9")):
                    if name == a:
                        d = boot - np.array([P.auroc(yh[i], scores[b][i]) for i in idx])
                        rec |= {f"minus {b}": rec["auroc"] - P.auroc(yh, scores[b]),
                                f"minus {b} lo": float(np.quantile(d, 0.025)), f"minus {b} hi": float(np.quantile(d, 0.975))}
                held.append(rec)
        logger.info(f"layer {layer} scored")
        del pooled
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "directions.npz", **dirs)
    depth, held, checks = pd.DataFrame(depth), pd.DataFrame(held), pd.DataFrame(checks)
    depth.round(4).to_csv(OUT / "depth.csv", index=False)
    held.round(4).to_csv(OUT / "held_out.csv", index=False)
    checks.round(5).to_csv(OUT / "rebuild_check.csv", index=False)
    _verdicts(held, checks)
    pd.set_option("display.width", 250)
    cols = [c for c in depth.columns if c.split(": ")[-1] in ("mean of five", "transfer", "HP-KR denials", "Gender-Secret")]
    logger.info(f"depth:\n{depth[['layer', 'cos(rule K, original K)', *cols]].round(3).to_string(index=False)}")
    logger.info(f"held out (AISI):\n{held.round(3).to_string(index=False)}")


def _verdicts(held: pd.DataFrame, checks: pd.DataFrame) -> None:
    h = held[held.layer == CRITERIA_LAYER].set_index("direction")
    c1 = bool((checks["cos(rule K 60+60, k_rule 20+20)"] >= 0.99).all())
    c2 = h.loc["rule K", "auroc"] >= h.loc["original K", "auroc"] - 0.03
    c3 = h.loc["D9 + rule K", "auroc"] >= h.loc["D9 + original K", "auroc"] - 0.03
    lines = [f"C1 rebuild: smallest cosine with k_rule {checks['cos(rule K 60+60, k_rule 20+20)'].min():.4f} "
             f"(>= 0.99): {'pass' if c1 else 'fail'}",
             f"C2 held out, L40: rule K {h.loc['rule K', 'auroc']:.3f} v original K {h.loc['original K', 'auroc']:.3f} "
             f"(within 0.03 or above): {'pass' if c2 else 'fail'}",
             f"C3 held out, L40: D9 + rule K {h.loc['D9 + rule K', 'auroc']:.3f} v D9 + original K "
             f"{h.loc['D9 + original K', 'auroc']:.3f} (within 0.03 or above): {'pass' if c3 else 'fail'}"]
    (OUT / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("verdicts:\n" + "\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"read": read, "score": score})
