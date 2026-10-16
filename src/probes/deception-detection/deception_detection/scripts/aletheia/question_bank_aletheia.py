"""The question extension, Qwen3.5-27B side: banks from the per-question store scored on the Aletheia DEV
split (development), and the single pre-registered read on the Aletheia TEST sets at the frozen headline
cell. Nothing here edits spec.py, score.py or bank.py; the Aletheia bank folder is only ever read.

    python deception_detection/scripts/aletheia/question_bank_aletheia.py store
    python deception_detection/scripts/aletheia/question_bank_aletheia.py dev
    python deception_detection/scripts/aletheia/question_bank_aletheia.py null_dev
    python deception_detection/scripts/aletheia/question_bank_aletheia.py decide
    python deception_detection/scripts/aletheia/question_bank_aletheia.py confirm --variant <selected>   # once

The dev reads are the 16 Qwen3.5 dev sets already on disk (base + 7 organisms x 2 splits, 100 rows a class),
read as `score._dev_reads_one` reads them; D9 + K is `rd.standardised_sum(D9, K, Alpaca)` with K from the
reading model (`score._k_read`), exactly the headline's recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.scripts import question_bank as QB
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts.aletheia import data as D
from deception_detection.scripts.aletheia import read as AR
from deception_detection.scripts.aletheia import score as SC
from deception_detection.scripts.aletheia import spec as S

MODEL = "qwen35"
RUN = Path(r"E:\qbank\banks\Qwen3.5-27B\persona")
STORE = "vectors_pq_s3_q150"
OUT = QB.REPO_ROOT / "results" / "question_bank" / "qwen35"
BLOCKS = S.MODELS[MODEL].organism_grid  # (16, 19, 22, 26, 29, 32)
MID = S.MODELS[MODEL].mid  # 32
ALETHEIA_BANK = Path(r"E:\aletheia\banks\Qwen3.5-27B\persona\vectors_s3")
CANDIDATES = ("B30", "B30+30mono", "B30+90div", "B150")
STORED_7SET_MEAN = 0.6245  # the frozen-spec headline over the 7 Qwen3.5 test sets
GEMMA_HARM = 0.8853        # the 8th headline cell, bank-independent


def store(run: str = str(RUN), name: str = STORE, tag: str = "s3") -> None:
    QB.store(run, name, tag, BLOCKS, out=OUT)


def _dev_sets(block: int) -> dict[str, tuple[np.ndarray, np.ndarray, str, S.Dataset | None]]:
    """key -> (x, y, read_name, matching test Dataset or None) for the Qwen3.5 dev sets read at `block`."""
    out = {}
    for key in D.dev_keys():
        parts = key.split("__")
        if parts[1] != MODEL:
            continue
        lora = parts[2]
        d_match = None
        if lora == "None":
            read_name = "base"
        else:
            match = [d for d in S.DATASETS if d.lora and d.lora.split("/")[-1] == lora]
            if not match or SC.organism_read(match[0]) is None:
                continue
            d_match = match[0]
            read_name = SC.organism_read(d_match)
        got = SC._dev_reads_one(MODEL, key, block)
        if got is None:
            continue
        x, y = got
        out[key] = (x, y, read_name, d_match)
    return out


def _k_for(read_name: str, d_match: S.Dataset | None, block: int) -> np.ndarray | None:
    if read_name == "base" or d_match is None:
        return SC.inputs(MODEL).k(block)
    return SC._k_read(d_match, read_name, block)


def _d9(bank: dict[str, np.ndarray]) -> np.ndarray:
    return QB.directions(bank)["D9"]


def dev(tag: str = "s3", blocks=BLOCKS) -> None:
    """dev.csv: AUROC of D9 and D9 + K of every bank variant on each of the 16 dev sets at every organism-grid
    block; dev_summary.csv: the mean over the dev sets. `B30-thesis` is the Aletheia bank itself (read-only)."""
    QB.freeze_check()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for block in blocks:
        sets = _dev_sets(block)
        if not sets:
            logger.warning(f"block {block}: no dev reads")
            continue
        control = SC.inputs(MODEL).control(block)
        banks = QB.banks_at(block, tag, out=OUT, thesis_dir=ALETHEIA_BANK)
        ks = {key: _k_for(rn, dm, block) for key, (_, _, rn, dm) in sets.items()}
        for bname, rv in banks.items():
            d9 = _d9(rv)
            for key, (x, y, read_name, _) in sets.items():
                k = ks[key]
                rows.append({"bank": bname, "block": block, "dev_set": key, "read": read_name, "direction": "D9",
                             "n": len(y), "auroc": SC.auroc(y, x @ d9)})
                if k is not None:
                    w = rd.standardised_sum(d9, k, control)
                    rows.append({"bank": bname, "block": block, "dev_set": key, "read": read_name,
                                 "direction": "D9 + K", "n": len(y), "auroc": SC.auroc(y, x @ w)})
        logger.info(f"block {block}: {len(sets)} dev sets x {len(banks)} banks scored")
    t = pd.DataFrame(rows)
    t.round(6).to_csv(OUT / "dev.csv", index=False)
    summary = t.groupby(["bank", "block", "direction"])["auroc"].agg(["mean", "count"]).reset_index()
    summary.round(6).to_csv(OUT / "dev_summary.csv", index=False)
    # reproduction check: the Aletheia bank's D9 at the middle block against dev_block_qwen35.json
    check = S.RESULTS / "checks" / f"dev_block_{MODEL}.json"
    if check.exists():
        ref = json.loads(check.read_text(encoding="utf-8"))
        by = {r["block"]: r["mean_d9_auroc"] for r in ref["by_block"]}
        got = summary[(summary["bank"] == "B30-thesis") & (summary["block"] == MID) & (summary["direction"] == "D9")]
        if len(got) and MID in by:
            diff = abs(float(got["mean"].iloc[0]) - float(by[MID]))
            logger.info(f"reproduction: B30-thesis D9 dev mean at block {MID} = {float(got['mean'].iloc[0]):.4f} "
                        f"vs dev_block_qwen35.json {by[MID]:.4f} (|diff| {diff:.2e})")
            if diff > 1e-3:
                raise RuntimeError("the Aletheia bank's dev D9 mean does not reproduce; check the reads")
    logger.info("dev summary at the middle block (D9 + K):\n" +
                summary[(summary["block"] == MID) & (summary["direction"] == "D9 + K")]
                .sort_values("mean", ascending=False).round(4).to_string())


def null_dev(n: int = QB.N_NULL, block: int = MID, tag: str = "s3", seed: int = QB.SEED) -> None:
    """200 random 30-of-150 subsets, 200 originals + random 30 diverse, 200 random 30 diverse: the dev mean of
    D9 and D9 + K at the middle block."""
    QB.freeze_check()
    rows_q = QB.questions()
    V, nn, roles, qids = QB.load_store(tag, block, OUT)
    qpos = {q: i for i, q in enumerate(qids)}
    orig = [r["id"] for r in rows_q if r["set"] == "orig"]
    div = [r["id"] for r in rows_q if r["set"] == "div"]
    rng = np.random.default_rng(seed)
    draws = ([("rand30of150", i, sorted(rng.choice(qids, 30, replace=False).tolist())) for i in range(n)]
             + [("orig+rand30div", i, orig + sorted(rng.choice(div, 30, replace=False).tolist())) for i in range(n)]
             + [("rand30div", i, sorted(rng.choice(div, 30, replace=False).tolist())) for i in range(n)])
    sets = _dev_sets(block)
    control = SC.inputs(MODEL).control(block)
    ks = {key: _k_for(rn, dm, block) for key, (_, _, rn, dm) in sets.items()}
    out = []
    for kind, i, qs in draws:
        vec = QB.bank_vectors(V, nn, [qpos[q] for q in qs], "plain")
        d9 = _d9({r: vec[j] for j, r in enumerate(roles)})
        a9, a9k = [], []
        for key, (x, y, _, _) in sets.items():
            a9.append(SC.auroc(y, x @ d9))
            if ks[key] is not None:
                a9k.append(SC.auroc(y, x @ rd.standardised_sum(d9, ks[key], control)))
        out.append({"kind": kind, "draw": i, "direction": "D9", "dev mean": float(np.mean(a9))})
        out.append({"kind": kind, "draw": i, "direction": "D9 + K", "dev mean": float(np.mean(a9k))})
    pd.DataFrame(out).round(6).to_csv(OUT / f"null_dev_B{block}.csv", index=False)
    logger.info(f"null_dev: {len(draws)} draws at block {block}")


def decide(block: int = MID) -> str:
    """The pre-registered rule: argmax over the candidates of the D9 + K dev mean at the middle block; ties within
    QB.TIE go to the larger bank. Writes decision.json."""
    s = pd.read_csv(OUT / "dev_summary.csv")
    s = s[(s["block"] == block) & (s["direction"] == "D9 + K") & s["bank"].isin(CANDIDATES)].set_index("bank")["mean"]
    vals = {c: float(s[c]) for c in CANDIDATES if c in s.index}
    top = max(vals.values())
    chosen = [c for c in CANDIDATES if c in vals and vals[c] >= top - QB.TIE][-1]
    (OUT / "decision.json").write_text(json.dumps({"block": block, "values": vals, "chosen": chosen}, indent=1) + "\n",
                                       encoding="utf-8")
    logger.info(f"decision at block {block}: {vals} -> {chosen}")
    return chosen


def confirm(variant: str, tag: str = "s3", n_boot: int = S.N_BOOT) -> None:
    """The headline cell (D9 + K, organism read, middle block, no-reasoning, content span, K from the reading
    model) with the selected bank on the 7 Qwen3.5 test sets, beside the stored headline, with the SAME bootstrap
    draws as `score.run` so the difference is paired. Runs once. Test labels are opened here and nowhere else."""
    out = OUT / "confirm.csv"
    if out.exists():
        raise SystemExit(f"{out} exists: the confirmation read has been done; it is not repeated")
    QB.freeze_check()
    AR.check_freeze()
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["chosen"] != variant:
        raise SystemExit(f"decision.json selected {decision['chosen']}, not {variant}")
    bank_dir = QB.materialise(variant, tag, layers=BLOCKS, run=str(RUN), out=OUT)
    stored = pd.read_csv(S.RESULTS / "auroc.csv")
    boots = np.load(S.RESULTS / "boots.npz")
    rows = []
    for d in S.DATASETS:
        if d.model != MODEL:
            continue
        read_name = SC.organism_read(d)
        block = d.base.mid
        got = SC._reads(d, read_name, "no-reasoning", 0, block)
        if got is None:
            rows.append({"dataset": d.key, "note": "no read"}); continue
        z, ids = got
        labels = D.labels(d.key)
        y = np.array([int(labels[i]) for i in ids])
        k = SC._k_read(d, read_name, block)
        control = SC.inputs(MODEL).control(block)
        d9 = pcs.load_bank_at_layer(block, bank_dir).axis_unit("extended")
        w = rd.standardised_sum(d9, k, control)
        s_new = z @ w
        s_d9 = z @ d9
        rng = np.random.default_rng([S.SEED, S.DATASETS.index(d)])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(n_boot)]
        b_new = np.array([SC.auroc(y[i], s_new[i]) for i in draws])
        b_d9 = np.array([SC.auroc(y[i], s_d9[i]) for i in draws])
        ref = stored[(stored["cell"] == "headline") & (stored["dataset"] == d.key)]
        ref_auroc = float(ref["auroc"].iloc[0]) if len(ref) else np.nan
        b_ref = boots[f"headline|{d.key}"] if f"headline|{d.key}" in boots.files else None
        diff = b_new - b_ref if b_ref is not None else None
        rows.append({"dataset": d.key, "read": read_name, "block": block, "n": len(y), "lies": int(y.sum()),
                     "bank": variant, "auroc": SC.auroc(y, s_new), "auroc_d9": SC.auroc(y, s_d9),
                     "stored_headline": ref_auroc, "delta": SC.auroc(y, s_new) - ref_auroc,
                     "delta_lo": float(np.quantile(diff, 0.025)) if diff is not None else np.nan,
                     "delta_hi": float(np.quantile(diff, 0.975)) if diff is not None else np.nan,
                     "lo": float(np.quantile(b_new, 0.025)), "hi": float(np.quantile(b_new, 0.975))})
    t = pd.DataFrame(rows)
    scored = t.dropna(subset=["auroc"]) if "auroc" in t else t
    mean7 = float(scored["auroc"].mean()) if len(scored) else np.nan
    summary = {"variant": variant, "sets": int(len(scored)), "mean_7set": mean7, "stored_7set": STORED_7SET_MEAN,
               "mean_8set_equivalent": (7 * mean7 + GEMMA_HARM) / 8 if len(scored) == 7 else np.nan}
    t.round(6).to_csv(out, index=False)
    (OUT / "confirm_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    AR.check_freeze()
    logger.info("confirmation read:\n" + t.round(4).to_string() + f"\n{summary}")


if __name__ == "__main__":
    fire.Fire({"store": store, "dev": dev, "null_dev": null_dev, "decide": decide, "confirm": confirm})
