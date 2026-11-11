"""The revised thesis's tables and figures across the four Liars' Bench models.

The thesis studies Llama-3.3-70B-Instruct. Its studies were re-run on Qwen2.5-72B-Instruct,
Mistral-Small-3.1-24B-Instruct and Gemma-3-27B-it with the same code (`lb_models.SPECS` maps each
model's blocks to the thesis's layers 20/36/40/44 and its 16-layer sweep). This collects every
model's outputs beside Llama's and writes, without touching the thesis's own assets:

    results/xmodel/tables/*.csv              every table, tidy
    results/xmodel/summary.md                the tables, and each thesis claim checked per model
    thesis_revised/generated_xmodel/*.tex    the tables as LaTeX (booktabs)
    thesis_revised/images_xmodel/*.pdf       the depth figures (x = relative depth)

Primary persona banks: Llama's thesis bank and the strict (judge score 3) banks of Qwen, Mistral and
Gemma. Qwen's strict bank is its user_stay bank of 2026-10-14 (the role line and a stay-in-character
sentence in the user turn, which Qwen plays; under the original system-message format it stayed fully
in role for 28% of its rollouts and its strict bank lacked 11 roles). Each model's lenient bank
(scores 2-3), Qwen's system-message bank and Llama's matched user_stay banks are reported beside the
primary banks (`bank_comparison`, `roleplay_formats`). Where the thesis
averages its five datasets, each model averages the Liars' Bench datasets it has (Qwen: HP-C,
HP-KR, ID; Mistral: HP-C, HP-KR, IT, ID; Gemma: CG, IT, ID), and Llama is also shown on each
model's subset. A table cell is empty when its input has not been produced; the summary says
which.

    python thesis_revised/make_xmodel_assets.py
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from deception_detection.scripts import lb_models as LB
from deception_detection.utils import REPO_ROOT

OUT_TABLES = LB.XMODEL_RESULTS / "tables"
OUT_TEX = REPO_ROOT / "thesis_revised" / "generated_xmodel"
OUT_FIG = REPO_ROOT / "thesis_revised" / "images_xmodel"
MODELS = ("llama", "qwen", "mistral", "gemma")
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}
LB_ORDER = ("CG", "HP-C", "HP-KR", "IT", "ID")
# zero_shot.csv columns -> the thesis's names for the persona contrasts
CONTRASTS = {"D5": "auroc_axis", "D6": "auroc_syc_axis", "D7": "auroc_uni_axis", "D8": "auroc_denial_axis",
             "D9": "auroc_ext_axis"}
# summary.csv features -> the transfer table's rows
TRANSFER = {"raw probe": "baseline", "D5": "axis1", "D7": "uni_axis1", "D8": "denial_axis1", "D9": "ext_axis1",
            "extended PCs (k=3)": "ext_pc3", "published PCs (k=3)": "pc3", "random (k=3)": "random3"}
MISSING: list[str] = []


def _bank(key: str, bank: str | None = None) -> str:
    return bank or LB.spec(key).primary_bank


def _label(key: str) -> str:
    return LB.spec(key).short


def _read(path: Path, what: str) -> pd.DataFrame | None:
    if not path.exists():
        MISSING.append(f"{what}: {path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path}")
        return None
    return pd.read_csv(path)


def _layer_names(key: str) -> dict[int, str]:
    """Each named block with the thesis layer it stands for."""
    s = LB.spec(key)
    return {s.lb_block: "L20", s.posthoc[0]: "L36", s.mid: "L40", s.posthoc[1]: "L44"}


def _rd_dir(key: str, bank: str | None = None) -> Path:
    from deception_detection.scripts import recall_denial as rd

    return rd.Inputs(None if key == "llama" and _bank(key, bank) == "ext" else key, bank).out


def _lb_sets(key: str) -> list[str]:
    return [SHORT[d] for d in LB.spec(key).datasets]


# --- S6: zero-shot persona contrasts (Table 5.1) ----------------------------------------------------

def zero_shot(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        s, b = LB.spec(key), _bank(key, bank if bank in LB.spec(key).banks else None)
        for layer, name in _layer_names(key).items():
            z = _read(s.results("persona_ood", layer, b) / "zero_shot.csv", f"{key} zero-shot {name}")
            if z is None:
                continue
            z = z.assign(set=z["dataset"].map(SHORT)).dropna(subset=["set"]).set_index("set")
            for contrast, col in CONTRASTS.items():
                lb = [z.loc[d, col] for d in _lb_sets(key) if d in z.index]
                row = {"model": _label(key), "layer": name, "block": layer, "contrast": contrast,
                       "LB mean": float(np.mean(lb)), "with RepE": float(np.mean(lb + [z.loc["RepE", col]]))}
                row |= {d: float(z.loc[d, col]) if d in z.index else math.nan for d in (*LB_ORDER, "RepE")}
                rows.append(row)
    return pd.DataFrame(rows)


def llama_on_subsets(table: pd.DataFrame) -> pd.DataFrame:
    """Llama's contrasts averaged over each other model's datasets, for a like-for-like reading."""
    llama = table[table["model"] == _label("llama")]
    rows = []
    for key in MODELS[1:]:
        sets = _lb_sets(key)
        for _, r in llama.iterrows():
            rows.append({"subset of": _label(key), "layer": r["layer"], "contrast": r["contrast"],
                         "Llama mean": float(np.mean([r[d] for d in sets]))})
    return pd.DataFrame(rows)


# --- S2, S11: transfer (Tables 5.4, B.1) ------------------------------------------------------------

def transfer(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        s, b = LB.spec(key), _bank(key, bank)
        ninek = _read(_rd_dir(key, bank) / "transfer" / "depth_ninek.csv", f"{key} D9+K transfer")
        ninek = ninek.set_index("layer") if ninek is not None else None
        d5k = _read(_rd_dir(key, bank) / "transfer" / "summary_d5k.csv", f"{key} D5+K and K transfer")
        d5k = d5k.set_index("layer") if d5k is not None else None
        for layer, name in _layer_names(key).items():
            summary = _read(s.results("persona_ood", layer, b) / "summary.csv", f"{key} transfer {name}")
            if summary is None:
                continue
            summary = summary.set_index("features")
            row = {"model": _label(key), "layer": name, "block": layer}
            for label, feature in TRANSFER.items():
                row[label] = float(summary.loc[feature, "off_diagonal_mean"]) if feature in summary.index else math.nan
            row["D9 + K"] = float(ninek.loc[layer, "D9 + K: transfer"]) if ninek is not None and layer in ninek.index else math.nan
            for label, column in (("D5 + K", "d5k: transfer"), ("K", "kalone: transfer")):  # `k_transfer.py d5k`
                row[label] = float(d5k.loc[layer, column]) if d5k is not None and layer in d5k.index else math.nan
            row["raw probe, diagonal"] = float(summary.loc["baseline", "diagonal_mean"])
            rows.append(row)
    return pd.DataFrame(rows)


# --- S8, S9, S12: the knowledge direction and D + K (Tables 5.3, 5.5) ----------------------------------

def knowledge(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        t = _read(_rd_dir(key, bank) / "d5_plus_k.csv", f"{key} d5_plus_k")
        if t is None:
            continue
        names = _layer_names(key)
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "layer": names.get(int(r["layer"]), f"B{int(r['layer'])}"),
                         "direction": r["direction"], "mean AUROC": r["mean of five"],
                         "mean BA (Alpaca 1%)": r["bal mean of five"],
                         "HP-KR identical": r.get("HP-KR identical text", math.nan),
                         "HP-KR denials": r.get("HP-KR denials", math.nan),
                         "flags honest denials": r.get("flags honest denials", math.nan)}
                        | {d: r.get(d, math.nan) for d in LB_ORDER})
    return pd.DataFrame(rows)


def k_depth(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        from deception_detection.scripts import k_robustness as kr
        from deception_detection.scripts import recall_denial as rd

        inputs = rd.Inputs(None if key == "llama" else key, bank if key != "llama" else None)
        t = _read(kr.out_dir(inputs) / "depth.csv", f"{key} K across depth")
        if t is None:
            continue
        n = LB.spec(key).n_blocks
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "block": int(r["layer"]), "relative depth": (int(r["layer"]) + 1) / n,
                         "K mean": r["k: mean of five"], "D9 + K mean": r["D9 + k: mean of five"],
                         "D9 mean": r["D9 alone: mean of five"],
                         "K identical": r.get("k: HP-KR identical", math.nan),
                         "band identical": r.get("random97.5 k: HP-KR identical", math.nan)})
    return pd.DataFrame(rows)


def paired(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        from deception_detection.scripts import k_robustness as kr
        from deception_detection.scripts import recall_denial as rd

        inputs = rd.Inputs(None if key == "llama" else key, bank if key != "llama" else None)
        t = _read(kr.out_dir(inputs) / "paired.csv", f"{key} paired differences")
        if t is None:
            continue
        names = _layer_names(key)
        t = t[t["set"].isin(["mean of five", "HP-KR identical", "HP-KR denials"])]
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "layer": names.get(int(r["layer"])), "set": r["set"],
                         "comparison": f"{r['a']} - {r['b']}", "delta": r["delta"], "lo": r["lo"], "hi": r["hi"]})
    return pd.DataFrame(rows)


def operating_points(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        from deception_detection.scripts import k_robustness as kr
        from deception_detection.scripts import recall_denial as rd

        inputs = rd.Inputs(None if key == "llama" else key, bank if key != "llama" else None)
        t = _read(kr.out_dir(inputs) / "operating_points.csv", f"{key} operating points")
        if t is None:
            continue
        t = t[t["layer"] == LB.spec(key).mid]
        means = t.groupby(["scorer", "fpr"])["caught, in-setting cutoff"].mean().unstack("fpr")
        for scorer, r in means.iterrows():
            rows.append({"model": _label(key), "scorer": scorer} | {f"{int(f * 100)}% FPR": v for f, v in r.items()})
    return pd.DataFrame(rows)


# --- S14: calibration (Table 5.7) --------------------------------------------------------------------

def calibration(bank: str | None = None) -> pd.DataFrame:
    rows = []
    arms = {"base probe (RepE-trained)": "base probe", "knowledge-denial axis (4v6)": "D8",
            "extended axis (12v13)": "D9", "deception axis (8v7)": "D5"}
    for key in MODELS:
        s = LB.spec(key)
        t = _read(s.results("persona_ood", s.mid, _bank(key, bank)) / "zero_shot_operating.csv", f"{key} operating")
        if t is None:
            continue
        t = t[t["dataset"] == "Average (Liars' Bench)"]
        for arm, label in arms.items():
            part = t[t["arm"] == arm].set_index("threshold_source")
            if part.empty:
                continue
            rows.append({"model": _label(key), "detector": label, "AUROC": part["auroc"].iloc[0],
                         "BA (Alpaca 1%)": part.loc["control", "balacc"] if "control" in part.index else math.nan,
                         "recall (Alpaca 1%)": part.loc["control", "recall"] if "control" in part.index else math.nan,
                         "BA (oracle)": (part.loc["in-dataset (oracle)", "balacc"]
                                         if "in-dataset (oracle)" in part.index else math.nan)})
    return pd.DataFrame(rows)


# --- S1: Liars' Bench's probes beside this pipeline's (Table 5.2) --------------------------------------

def liars_bench() -> pd.DataFrame:
    from deception_detection.scripts import persona_ood as P

    rows = []
    for key in MODELS:
        s = LB.spec(key)
        published = LB.published(key)
        sets = _lb_sets(key)
        for probe, spec in published.items():
            values = dict(zip(P.PAPER_COLUMNS, spec["auroc"]))
            rows.append({"model": _label(key), "probe": f"{probe} (published)",
                         "mean AUROC": float(np.nanmean([values[d] for d in sets]))}
                        | {d: values.get(d, math.nan) for d in LB_ORDER})
        for layer, name in _layer_names(key).items():
            if name not in ("L20", "L40"):
                continue
            t = _read(s.results("persona_ood", layer, s.primary_bank) / "zero_shot_operating.csv", f"{key} operating {name}")
            if t is None:
                continue
            t = t[(t["arm"] == "base probe (RepE-trained)") & (t["threshold_source"] == "control")]
            t = t.assign(set=t["dataset"].map(SHORT)).dropna(subset=["set"]).set_index("set")
            rows.append({"model": _label(key), "probe": f"base probe, {name} (ours)",
                         "mean AUROC": float(np.mean([t.loc[d, "auroc"] for d in sets if d in t.index]))}
                        | {d: float(t.loc[d, "auroc"]) if d in t.index else math.nan for d in LB_ORDER})
    return pd.DataFrame(rows)


# --- S4: D3, the probe trained on the model's own lies (Gemma) -------------------------------------------

def aisi() -> pd.DataFrame:
    rows = []
    for key, path in (("llama", REPO_ROOT / "results" / "aisi_probe" / "comparison.csv"),
                      ("gemma", LB.XMODEL_RESULTS / "gemma" / "aisi_probe" / "comparison.csv")):
        t = _read(path, f"{key} D3 (AISI) comparison")
        if t is None:
            continue
        t = t[(t["threshold_source"] == "control") & (t["dataset"] == "Average (Liars' Bench)")]
        names = _layer_names(key)
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "layer": names.get(int(r["layer"]), r["layer"]), "detector": r["arm"],
                         "LB mean AUROC": r["auroc"], "BA (Alpaca 1%)": r["balacc"]})
    return pd.DataFrame(rows)


# --- S10: model organisms ---------------------------------------------------------------------------

def organisms() -> pd.DataFrame:
    rows = []
    for key in MODELS[1:]:
        for organism in ("gender-secret", "greeting", "time", "gender"):
            folder = LB.XMODEL_RESULTS / key / "organisms" / organism
            check = folder / "check_behaviour.json"
            if not check.exists():
                continue
            c = json.loads(check.read_text(encoding="utf-8"))
            base = {"model": _label(key), "organism": organism, "behaviour check": "passed" if c["passed"] else "failed"}
            if "male preferred" in c.get("adapted", {}):
                base |= {"adapted": c["adapted"]["male preferred"], "base": c["base"]["male preferred"]}
            else:
                base |= {"adapted": c["adapted"]["lies"], "base": c["base"]["lies"]}
            t = _read(folder / "auroc.csv", f"{key} {organism} scores") if c["passed"] else None
            if t is None:
                rows.append(base)
                continue
            names = _layer_names(key)
            for layer, part in t[t["read"] == "adapted"].groupby("layer"):
                p = part.set_index("direction")
                row = base | {"layer": names.get(int(layer))}
                for d in ("k base", "k fine-tune", "D9 bank", "D9 bank + k base (standardised)", "base probe (RepE-trained)"):
                    row[d] = float(p.loc[d, "auroc"]) if d in p.index else math.nan
                row["band97.5"] = float(p["band97.5"].iloc[0])
                rows.append(row)
    return pd.DataFrame(rows)


# --- S7 on the rebuilt HP-KR rows (Qwen's identical text, Gemma's supplementary set) ----------------------

def rebuilt_hpkr(bank: str | None = None) -> pd.DataFrame:
    from deception_detection.scripts import k_models as km

    rows = []
    for key in ("qwen", "gemma"):
        b = _bank(key, bank)
        t = _read(km.OUT / key / "hpkr" / f"scores_persona_{b}.csv", f"{key} rebuilt HP-KR slices")
        if t is None:
            continue
        names = _layer_names(key)
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "layer": names.get(int(r["block"])), "slice": r["slice"],
                         "scorer": r["scorer"], "lies": r["n_lies"], "honest": r["n_honest"], "AUROC": r["auroc"],
                         "lo": r["lo"], "hi": r["hi"], "band97.5": r["band97.5"]})
    return pd.DataFrame(rows)


# --- S14: K', the replication with matched items (Table 5.6) ---------------------------------------------

def k_matched(bank: str | None = None) -> pd.DataFrame:
    from deception_detection.scripts import k_replication as kp

    rows = []
    for key in MODELS:
        t = _read(kp.out_dir(None if key == "llama" else key) / "verdicts.csv", f"{key} K' verdicts")
        if t is None:
            continue
        names = _layer_names(key)
        for _, r in t.iterrows():
            rows.append({"model": _label(key), "layer": names.get(int(r["layer"])), "criterion": r["criterion"],
                         "value": r["value"], "passes": bool(r["pass"])})
    return pd.DataFrame(rows)


# --- S13: token reads (Table B.6) and the Follow-Up turn (Table B.7) ---------------------------------------

def token_reads(bank: str | None = None) -> pd.DataFrame:
    rows = []
    for key in MODELS:
        s = LB.spec(key)
        for layer, name in _layer_names(key).items():
            if name == "L36":
                continue
            folder = (REPO_ROOT / "results" / f"persona_positions_L{layer}" if key == "llama"
                      else s.results("persona_positions", layer, _bank(key, bank)))
            t = _read(folder / "reads_summary.csv", f"{key} token reads {name}")
            if t is None:
                continue
            t = t[t["axis"] == "extended"]
            for _, r in t.iterrows():
                rows.append({"model": _label(key), "layer": name, "read": r["read"], "AUROC": r["auroc"],
                             "BA": r["balacc"], "recall": r["recall"]})
    return pd.DataFrame(rows)


def followup_turn(bank: str | None = None) -> pd.DataFrame:
    """D9 and the base probe on the response tokens against the Follow-Up turn's "No", at the
    model's Liars' Bench block."""
    rows = []
    for key in MODELS:
        s = LB.spec(key)
        b = _bank(key, bank)
        main = (s.results("persona_ood", s.lb_block, b) / "zero_shot.csv")
        fu = (REPO_ROOT / "results" / "persona_ood_followup_L20" / "zero_shot.csv" if key == "llama"
              else s.results("persona_ood_followup", s.lb_block, b) / "zero_shot.csv")
        a, f = _read(main, f"{key} zero-shot at the LB block"), _read(fu, f"{key} Follow-Up zero-shot")
        if a is None or f is None:
            continue
        a, f = (x.assign(set=x["dataset"].map(SHORT)).dropna(subset=["set"]).set_index("set") for x in (a, f))
        for d in _lb_sets(key):
            if d in a.index and d in f.index:
                rows.append({"model": _label(key), "dataset": d, "D9, response tokens": a.loc[d, "auroc_ext_axis"],
                             "D9, follow-up No": f.loc[d, "auroc_ext_axis"]})
    return pd.DataFrame(rows)


# --- the role-play finding: how fully each model plays the roles ----------------------------------------

# the judge's score files of each model's primary bank ...
SCORE_DIRS = {"llama": LB.XMODEL_ROOT / "banks" / "llama-70b-3.3" / "scores",
              "qwen": LB.XMODEL_ROOT / "banks" / "qwen-72b" / "userstay" / "persona" / "scores",
              "mistral": LB.XMODEL_ROOT / "banks" / "mistral-small-3.1-24b" / "scores",
              "gemma": Path("E:/gemma-3-27b-it/persona/scores")}
# ... and of the banks built under the other prompt format: Qwen's original system-message bank, and
# Llama's matched user_stay bank (the role line and a stay-in-character sentence in the user turn, 2026-10-14)
FORMATS = ("system message", "user turn")
PRIMARY_FORMAT = {"llama": "system message", "qwen": "user turn", "mistral": "system message", "gemma": "system message"}
FORMAT_SCORE_DIRS = {("qwen", "system message"): LB.XMODEL_ROOT / "banks" / "qwen-72b" / "scores",
                     ("llama", "user turn"): LB.XMODEL_ROOT / "banks" / "llama-70b-3.3" / "userstay" / "persona" / "scores"}
# the banks each model is compared across (`bank_comparison`, `banks_table`), the primary first: the same
# format's lenient bank, then the banks of the other prompt format
COMPARED_BANKS = {"llama": ("ext", "s23", "s3u", "s23u"), "qwen": ("s3u", "s23u", "s23"),
                  "mistral": ("s3", "s23"), "gemma": ("s3", "s23")}
BANK_FORMAT = {"ext": "system message", "orig": "system message", "s3": "system message", "s23": "system message",
               "s3u": "user turn", "s23u": "user turn"}
BANK_FILTER = {"ext": "score 3", "orig": "score 3", "s3": "score 3", "s3u": "score 3", "s23": "scores 2--3",
               "s23u": "scores 2--3"}


def _scores_of(path: Path) -> list[int]:
    """The judge's 0-3 scores in one role's score file (a list of records, or a dict of them)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("scores", list(data.values()))
    out = []
    for r in records:
        value = r.get("score") if isinstance(r, dict) else r
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _roleplay_rows(folder: Path, label: str) -> list[dict]:
    """One row per role: the judge's score counts in `folder`'s score files."""
    rows = []
    for path in sorted(folder.glob("*.json")):
        scores = _scores_of(path)
        if not scores:
            continue
        counts = np.bincount(np.clip(scores, 0, 3), minlength=4)
        rows.append({"model": label, "role": path.stem, "judged": len(scores)}
                    | {f"score {i}": int(counts[i]) for i in range(4)}
                    | {"share fully in role (3)": counts[3] / len(scores)})
    return rows


def roleplay() -> pd.DataFrame:
    """Each model's primary bank: how fully the judge found every role played."""
    rows = []
    for key in MODELS:
        folder = SCORE_DIRS[key]
        if not folder.exists():
            MISSING.append(f"{key} judge scores: {folder}")
            continue
        rows += _roleplay_rows(folder, _label(key))
    return pd.DataFrame(rows)


def roleplay_formats() -> pd.DataFrame:
    """How fully Llama and Qwen play the roles under each prompt format: the role line as the system
    message (the thesis's format) against the role line and a stay-in-character sentence in the user
    turn. Shares of rollouts scored 3, over every judged role and over the D5 and D8 groups."""
    from deception_detection import persona_pcs as pcs

    groups = {"deceptive (D5)": pcs.DECEPTION_PERSONAS, "honest (D5)": pcs.HONESTY_PERSONAS,
              "knowledge-denial (D8)": pcs.KNOWLEDGE_DENIAL_PERSONAS,
              "knowledge-candour (D8)": pcs.KNOWLEDGE_CANDOUR_PERSONAS}
    rows = []
    for key in ("llama", "qwen"):
        folders = {PRIMARY_FORMAT[key]: SCORE_DIRS[key]} | {f: p for (k, f), p in FORMAT_SCORE_DIRS.items() if k == key}
        for fmt in FORMATS:
            folder = folders[fmt]
            if not folder.exists():
                MISSING.append(f"{key} judge scores, {fmt} format: {folder}")
                continue
            part = pd.DataFrame(_roleplay_rows(folder, _label(key))).set_index("role")
            row = ({"model": _label(key), "format": fmt, "primary": fmt == PRIMARY_FORMAT[key],
                    "judged": int(part["judged"].sum())}
                   | {f"score {i}": int(part[f"score {i}"].sum()) for i in range(4)})
            row["share fully in role (3)"] = float(part["score 3"].sum() / part["judged"].sum())
            for group, roles in groups.items():
                have = [r for r in roles if r in part.index]
                row[group] = (float(part.loc[have, "score 3"].sum() / part.loc[have, "judged"].sum())
                              if have else math.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# --- each model's other banks: the judge's filter and the prompt format -----------------------------------

def bank_comparison() -> pd.DataFrame:
    """D5, D8, D9 zero-shot (LB mean) and D9 transfer at mid-depth for each of a model's banks
    (`COMPARED_BANKS`): the primary bank, the same format's lenient bank, and the banks built under the
    other prompt format."""
    rows = []
    for key in MODELS:
        s = LB.spec(key)
        for bank in COMPARED_BANKS[key]:
            z = _read(s.results("persona_ood", s.mid, bank) / "zero_shot.csv", f"{key} zero-shot, bank {bank}")
            summ = _read(s.results("persona_ood", s.mid, bank) / "summary.csv", f"{key} transfer, bank {bank}")
            if z is None or summ is None:
                continue
            z = z.assign(set=z["dataset"].map(SHORT)).dropna(subset=["set"]).set_index("set")
            summ = summ.set_index("features")
            row = {"model": _label(key), "bank": bank, "format": BANK_FORMAT[bank], "filter": BANK_FILTER[bank],
                   "primary": bank == s.primary_bank}
            for contrast in ("D5", "D8", "D9"):
                row[f"{contrast} LB mean"] = float(np.mean([z.loc[d, CONTRASTS[contrast]] for d in _lb_sets(key)]))
            row["D9 transfer"] = float(summ.loc["ext_axis1", "off_diagonal_mean"])
            rows.append(row)
    return pd.DataFrame(rows)


# --- across depth (Figures 5.4-5.6) ---------------------------------------------------------------------

def depth() -> pd.DataFrame:
    from deception_detection.scripts.layer_sweep import LB_SOURCE, TRANSFER_SOURCE

    rows = []
    for key in MODELS:
        s = LB.spec(key)
        t = _read(s.results("persona_depth", bank=s.primary_bank) / "depth.csv", f"{key} persona depth")
        if t is None:
            continue
        t = t[t["metric"] == "auroc"]
        for _, r in t.iterrows():
            if r["source"] not in (TRANSFER_SOURCE, LB_SOURCE):
                continue
            rows.append({"model": _label(key), "block": int(r["layer"]), "relative depth": (int(r["layer"]) + 1) / s.n_blocks,
                         "source": r["source"], "arm": str(r["arm"]).replace(f"({s.hidden}-d)", "(raw)").replace("(8192-d)", "(raw)"),
                         "value": r["value"]})
    return pd.DataFrame(rows)


def depth_figure(table: pd.DataFrame, kd: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deception_detection.scripts.layer_sweep import LB_SOURCE, TRANSFER_SOURCE

    if table.empty:
        return
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    colours = {_label(k): c for k, c in zip(MODELS, ("#2a78d6", "#eb6834", "#1f9e77", "#b5399e"))}
    panels = [(TRANSFER_SOURCE, "extended axis (12v13)", "D9 transfer (off-diagonal AUROC)"),
              (TRANSFER_SOURCE, "single-source probe (raw)", "raw-activation probe transfer"),
              (LB_SOURCE, "extended axis (12v13)", "D9 zero-shot, Liars' Bench mean AUROC"),
              ("K", "K identical", "K on HP-KR's identical-text slice")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
    for ax, (source, arm, title) in zip(axes, panels):
        for model, colour in colours.items():
            if source == "K":
                part = kd[kd["model"] == model] if not kd.empty else kd
                if part.empty or part["K identical"].isna().all():
                    continue
                ax.plot(part["relative depth"], part["K identical"], marker="o", ms=3, color=colour, label=model)
            else:
                part = table[(table["model"] == model) & (table["source"] == source) & (table["arm"] == arm)]
                if part.empty:
                    continue
                part = part.sort_values("relative depth")
                ax.plot(part["relative depth"], part["value"], marker="o", ms=3, color=colour, label=model)
        ax.axhline(0.5, color="#888", lw=0.8, ls=":")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("relative depth, (block + 1) / blocks", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="#eee", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_xmodel_depth.pdf", bbox_inches="tight", metadata=FIGURE_METADATA)
    fig.savefig(OUT_FIG / "fig_xmodel_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- the thesis's claims, per model -------------------------------------------------------------------

TIE = 0.01  # AUROC: a point comparison without an interval closer than this is "no clear difference"


def claims(zs: pd.DataFrame, tr: pd.DataFrame, kn: pd.DataFrame, kd: pd.DataFrame, pr: pd.DataFrame,
           org: pd.DataFrame, ci: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    ci = ci if ci is not None else pd.DataFrame()

    def add(model, claim, value, holds):
        verdict = ("n/a" if holds is None else holds if isinstance(holds, str)
                   else "replicates" if holds else "does not replicate")
        rows.append({"model": model, "claim": claim, "evidence": value, "verdict": verdict})

    def better(a: float, b: float):
        """a above b by at least TIE; within TIE of each other is no clear difference."""
        return True if a - b >= TIE else False if b - a >= TIE else f"no clear difference (< {TIE})"

    def interval(key: str, metric: str, a: str, b: str):
        """The paired 95% interval of a - b at mid-depth (`transfer_ci.py`), or None."""
        if ci.empty:
            return None
        r = ci[(ci["model"] == key) & (ci["layer"] == "L40") & (ci["metric"] == metric) & (ci["a"] == a) & (ci["b"] == b)]
        return r.iloc[0] if len(r) else None

    def compare(model, key, claim, metric, a, b, label_a, label_b, va, vb):
        """By the paired interval where there is one: above 0 replicates, below 0 does not, else
        no clear difference; by the TIE rule otherwise."""
        r = interval(key, metric, a, b)
        if r is None:
            add(model, claim, f"{label_a} {va:.3f} vs {label_b} {vb:.3f}", better(va, vb))
            return
        holds = True if r["lo"] > 0 else False if r["hi"] < 0 else "no clear difference (95% interval includes 0)"

        add(model, claim, f"{label_a} {_shown(r['a_value'])} vs {label_b} {_shown(r['b_value'])}; "
                          f"{_signed(r['delta'])} [{_signed(r['lo'])}, {_signed(r['hi'])}]", holds)

    for key in MODELS:
        m = _label(key)
        t = tr[(tr["model"] == m) & (tr["layer"] == "L40")] if not tr.empty else tr
        if len(t):
            r = t.iloc[0]
            compare(m, key, "D9 transfers better than the raw-activation probe (mid-depth)", "transfer", "D9",
                    "raw probe", "D9", "raw", r["D9"], r["raw probe"])
            compare(m, key, "D9 transfers better than the published contrast D5", "transfer", "D9", "D5", "D9", "D5",
                    r["D9"], r["D5"])
            if not math.isnan(r["D9 + K"]):
                compare(m, key, "D9 + K transfers better than the raw-activation probe", "transfer", "D9 + K",
                        "raw probe", "D9 + K", "raw", r["D9 + K"], r["raw probe"])
        z = zs[(zs["model"] == m) & (zs["layer"] == "L40")] if not zs.empty else zs
        if len(z):
            d9 = float(z[z["contrast"] == "D9"]["LB mean"].iloc[0])
            d5 = float(z[z["contrast"] == "D5"]["LB mean"].iloc[0])
            compare(m, key, "knowledge-denial roles improve the contrast (D9 > D5, LB mean)", "dataset mean", "D9",
                    "D5", "D9", "D5", d9, d5)
        k = kd[kd["model"] == m] if not kd.empty else kd
        if len(k) and not k["K identical"].isna().all():
            mid = k[k["block"] == LB.spec(key).mid]
            if len(mid):
                r = mid.iloc[0]
                add(m, "K separates lies on HP-KR's identical text above random directions (mid-depth)",
                    f"{r['K identical']:.3f} vs band {r['band identical']:.3f}", r["K identical"] > r["band identical"])
        elif key in ("gemma", "qwen"):
            add(m, "K on HP-KR's identical text", "no released identical-text slice (see the rebuilt HP-KR table)", None)
        p = pr[(pr["model"] == m) & (pr["layer"] == "L40") & (pr["set"] == "mean of five")
               & (pr["comparison"] == "D9 + k - benchmark probe, same layer")] if not pr.empty else pr
        if len(p):
            r = p.iloc[0]
            add(m, "D9 + K beats the same-layer RepE probe (paired 95% CI)",
                f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", r["lo"] > 0)
        o = org[(org["model"] == m)] if not org.empty else org
        for organism, part in (o.groupby("organism") if len(o) else []):
            if part["behaviour check"].iloc[0] != "passed":
                add(m, f"organism {organism}: adapter reproduces the lies", "behaviour check failed", None)
                continue
            mid = part[part["layer"] == "L40"]
            if len(mid) and "k fine-tune" in mid:
                r = mid.iloc[0]
                add(m, f"organism {organism}: K (fine-tune) above the random band on the fine-tune's read",
                    f"{r['k fine-tune']:.3f} vs {r['band97.5']:.3f}", r["k fine-tune"] > r["band97.5"])
    return pd.DataFrame(rows)


# --- paired intervals and the template-free K -----------------------------------------------------

def intervals() -> pd.DataFrame:
    """Every model's paired 95% intervals (`transfer_ci.py`)."""
    parts = []
    for key in MODELS:
        t = _read(LB.XMODEL_RESULTS / key / "transfer_ci.csv", f"{key} transfer intervals")
        if t is not None:
            parts.append(t.assign(model=_label(key)))
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    t["comparison"] = t["a"] + " - " + t["b"]
    return t[["model", "layer", "block", "metric", "comparison", "a_value", "b_value", "delta", "lo", "hi"]]


def _intervals_by_key() -> pd.DataFrame:
    parts = [pd.read_csv(p).assign(model=key) for key in MODELS
             if (p := LB.XMODEL_RESULTS / key / "transfer_ci.csv").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


RULE_METRICS = (("dataset mean", "mean of five", "dataset mean"), ("transfer", "transfer", "transfer"),
                ("HP-KR identical text", "HP-KR identical", "HP-KR identical"),
                ("HP-KR denials", "HP-KR denials", "HP-KR denials"), ("Gender-Secret", "Gender-Secret", None))


def rule_k() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The template-free K (`k_final.py` for Llama, `k_final_xmodel.py` for the others) beside the
    original K at mid-depth, and each model's verdicts on the criteria it fixed before reading."""
    from deception_detection.scripts import recall_denial as rd

    rows, verdicts = [], []
    for key in MODELS:
        s = LB.spec(key)
        folder = rd.OUT / "k_final" if key == "llama" else LB.XMODEL_RESULTS / key / "k_final"
        depth = _read(folder / "depth.csv", f"{key} template-free K")
        if depth is None:
            continue
        r = depth.set_index("layer").loc[s.mid]
        for label, llama_col, other_col in RULE_METRICS:
            col = llama_col if key == "llama" else other_col
            if col is None or f"original K: {col}" not in r:
                continue
            for k_label, prefix in (("K", ""), ("D9 + K", "D9 + ")):
                a, b = r.get(f"{prefix}original K: {col}"), r.get(f"{prefix}rule K: {col}")
                if a is None or pd.isna(a):
                    continue
                rows.append({"model": _label(key), "block": s.mid, "metric": label, "with": k_label,
                             "original K (template)": a, "rule K (template-free)": b, "difference": b - a})
        held = folder / "held_out.csv"
        if held.exists():
            h = pd.read_csv(held)
            h = h[h["layer"] == s.mid].set_index("direction")
            for k_label, prefix in (("K", ""), ("D9 + K", "D9 + ")):
                if f"{prefix}original K" in h.index:
                    a, b = h.loc[f"{prefix}original K", "auroc"], h.loc[f"{prefix}rule K", "auroc"]
                    rows.append({"model": _label(key), "block": s.mid, "metric": "held out (AISI)", "with": k_label,
                                 "original K (template)": a, "rule K (template-free)": b, "difference": b - a})
        rows.append({"model": _label(key), "block": s.mid, "metric": "cosine between the two", "with": "",
                     "original K (template)": math.nan, "rule K (template-free)": r["cos(rule K, original K)"],
                     "difference": math.nan})
        text = folder / "verdict.txt"
        if text.exists():
            for line in text.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    verdicts.append({"model": _label(key), "criterion": line.split(" ", 1)[0], "verdict": line})
    return pd.DataFrame(rows), pd.DataFrame(verdicts)


def rule_k_vs_probe(scorers=("original K", "rule K", "D9 + original K", "D9 + rule K")) -> pd.DataFrame:
    """`scorers` against the same-layer RepE base probe at mid-depth, zero-shot
    (`k_final_xmodel.py vs_probe`): each cell is the AUROC and its difference from the probe with
    the paired 95% interval."""
    rows = []
    for key in MODELS:
        t = _read(LB.XMODEL_RESULTS / key / "k_final" / "vs_probe.csv", f"{key} template-free K vs base probe")
        if t is None:
            continue
        t = t[t["block"] == LB.spec(key).mid]
        for set_name, part in t.groupby("set", sort=False):
            part = part.set_index("scorer")
            row = {"model": _label(key), "set": set_name, "base probe": _shown(part["base probe auroc"].iloc[0])}
            for scorer in scorers:
                if scorer in part.index:
                    r = part.loc[scorer]
                    row[scorer] = f"{_shown(r['auroc'])} ({_signed(r['delta'])} [{_signed(r['lo'])}, {_signed(r['hi'])}])"
            rows.append(row)
    return pd.DataFrame(rows)


# --- writing ---------------------------------------------------------------------------------------

def _shown(value) -> str:
    """Three places, rounded half up as the thesis's own tables round (`make_numbers.fmt`); empty
    when the input is missing."""
    if pd.isna(value):
        return ""
    return str(Decimal(repr(float(value))).quantize(Decimal("0.001"), ROUND_HALF_UP))


def _signed(value) -> str:
    """`_shown` with an explicit sign; a value that rounds to zero reads +0.000."""
    s = _shown(value)
    s = "0.000" if s == "-0.000" else s
    return s if s.startswith("-") else "+" + s


def _write(name: str, table: pd.DataFrame, caption: str) -> str:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_TEX.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLES / f"{name}.csv", index=False)
    if table.empty:
        return f"## {caption}\n\n(no inputs yet)\n"
    shown = table.copy()
    numeric = [pd.api.types.is_numeric_dtype(shown[column]) for column in shown.columns]
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(_shown)
    latex = shown.to_latex(index=False, escape=True, caption=caption, label=f"tab:x-{name}", na_rep="",
                           column_format="".join("r" if n else "l" for n in numeric))
    (OUT_TEX / f"tab_x_{name}.tex").write_text(latex.replace("\u2020", r"$^\dagger$"), encoding="utf-8")
    return f"## {caption}\n\n{shown.to_markdown(index=False, disable_numparse=True)}\n"


# --- the thesis's section "Replication Across Models" and its Appendix D ------------------------------
#
# `chapters/chapter_6.tex` (section "Replication Across Models") and `appendix/appendix_5.tex` read
# what `thesis_assets` writes: `generated_xmodel/numbers.tex`, whose `xm-*` keys the text reads as
# \V{key}, the `tab_xm_*.tex` tables and `images_xmodel/fig_xm_depth.pdf`. The numbers are formatted
# by the paper's own `Numbers` (`paper/make_numbers.py`), as the thesis's own are, and each claim the
# section makes is asserted below, so a result that changes stops the build instead of contradicting
# the text. Every number is read at a model's mid block, the thesis's layer 40, unless its key says
# otherwise. Llama's numbers are the thesis's own, and each one the section shares with
# `generated/numbers.tex` must print the same there (`SHARED_KEYS`).

THESIS_NUMBERS = REPO_ROOT / "thesis_revised" / "generated" / "numbers.tex"
SHORT_NAME = {"llama": "Llama", "qwen": "Qwen", "mistral": "Mistral", "gemma": "Gemma"}
TOKEN = {"raw probe": "raw", "D5": "dfive", "D8": "deight", "D9": "dnine", "D9 + K": "ninek", "D5 + K": "fivek",
         "K": "k", "base probe": "probe"}
SHARED_KEYS = {
    "xm-llama-off-raw": "off-base-L40", "xm-llama-off-dfive": "off-dec-L40", "xm-llama-off-deight": "off-den-L40",
    "xm-llama-off-dnine": "off-ext-L40", "xm-llama-off-ninek": "th-k-ninek-off-L40", "xm-llama-off-k": "th-k-k-off-L40",
    "xm-llama-lb-dfive": "zs-dec-lb-L40", "xm-llama-lb-deight": "zs-den-lb-L40", "xm-llama-lb-dnine": "zs-ext-lb-L40",
    "xm-llama-lb-k": "th-k-k-mean-L40", "xm-llama-lb-ninek": "th-k-ninek-mean-L40",
    "xm-llama-lb-fivek": "th-k-fivek-mean-L40", "xm-llama-lb-probe": "bp-auroc-lb-L40",
    "xm-llama-lb-meanprobe": "lb-mean-auroc-avg", "xm-llama-ident-k": "th-k-k-ident-L40",
    "xm-llama-ident-band": "th-k-ident-band-L40", "xm-llama-ident-dfive": "th-k-dfive-ident-L40",
    "xm-llama-ident-deight": "th-k-deight-ident-L40", "xm-llama-ident-dnine": "th-k-dnine-ident-L40",
    "xm-llama-ident-ninek": "th-k-ninek-ident-L40", "xm-llama-ident-lies": "th-k-ident-lies",
    "xm-llama-ident-honest": "th-k-ident-honest", "xm-llama-ci-lb-ninek-probe": "th-pair-ninek-probe-mean-L40",
    "xm-llama-ci-lb-ninek-dnine": "th-pair-ninek-dnine-mean-L40", "xm-llama-op1-dnine": "th-k-op-dnine-1-L40",
    "xm-llama-op1-ninek": "th-k-op-ninek-1-L40", "xm-llama-gs-k": "th-k-gs-ft-kft-L40",
    "xm-llama-gs-band": "th-k-gs-ft-band-L40", "xm-llama-depth-n": "depth-n",
    "xm-llama-ident-probe": "th-k-probe-ident-L40", "xm-llama-den-probe": "th-k-probe-den-L40",
    "xm-llama-den-k": "th-k-k-den-L40", "xm-llama-den-band": "th-k-den-band-L40",
    "xm-llama-den-dnine": "th-k-dnine-den-L40", "xm-llama-den-deight": "th-k-deight-den-L40",
    "xm-llama-den-dfive": "th-k-dfive-den-L40", "xm-llama-den-ninek": "th-k-ninek-den-L40",
    "xm-llama-den-lies": "th-k-den-lies", "xm-llama-den-honest": "th-k-den-honest",
    "xm-llama-depth-ident-above": "th-k-depth-ident-above", "xm-llama-hpkr-dnine": "zs-ext-hpkr-L40",
    "xm-llama-hpkr-dfive": "zs-dec-hpkr-L40", "xm-llama-hpkr-deight": "zs-den-hpkr-L40",
}
READ_NAME = {"first": "first-token", "last": "last-token", "max": "maximum", "mean": "mean", "pre": "pre-reply",
             "top10": "top-10\\% mean"}
FIGURE_METADATA = {"CreationDate": None, "ModDate": None}  # no timestamps, so a rerun leaves the PDF unchanged


def _paper_numbers():
    """The paper's generator (`paper/make_numbers.py`), for `Numbers` and `fmt`."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "paper"))
    import make_numbers  # noqa: E402

    return make_numbers


def _thesis_values() -> dict[str, str]:
    """`generated/numbers.tex`, key -> printed value."""
    import re

    pattern = re.compile(r"^\\setV\{([^}]*)\}\{(.*)\}$")
    return {m.group(1): m.group(2) for line in THESIS_NUMBERS.read_text(encoding="utf-8").splitlines()
            if (m := pattern.match(line))}


def _one(frame: pd.DataFrame, **equal) -> pd.Series:
    """The one row of `frame` whose columns equal `equal`."""
    part = frame
    for column, value in equal.items():
        part = part[part[column] == value]
    if len(part) != 1:
        raise RuntimeError(f"expected one row for {equal}, found {len(part)}")
    return part.iloc[0]


def _signed_tex(value: float) -> str:
    """A signed three-place value as the thesis's tables print it: $+$0.007, $-$0.004, 0.000."""
    M = _paper_numbers()
    if round(float(value), 3) == 0:
        return "0.000"
    return f"$+${M.fmt(value)}" if value > 0 else f"$-${M.fmt(-value)}"


def _slices(key: str) -> pd.DataFrame:
    """The combine run's HP-KR slice scores (`recall_denial.py combine`), where the thesis's slice
    bands come from, at the model's mid block."""
    t = _read(_rd_dir(key) / "combined_hpkr_slices.csv", f"{key} HP-KR slices")
    return t[t["layer"] == LB.spec(key).mid] if t is not None else pd.DataFrame()


def _slice_counts(key: str) -> dict[str, tuple[int, int]]:
    """Lies and honest rows in each of the model's HP-KR slices."""
    from deception_detection.scripts import recall_denial as rd

    frame, slices = rd.Inputs(None if key == "llama" else key, None if key == "llama" else LB.spec(key).primary_bank).hpkr()
    lie = frame["lie"].to_numpy().astype(bool)
    return {name: (int(lie[mask].sum()), int((~lie[mask]).sum())) for name, mask in slices.items()}


def _organism_reads() -> pd.DataFrame:
    """Every organism's AUROCs on the fine-tune's read and on the base model's read of the same
    transcripts, at each model's mid block and the block before it (the thesis's layers 40 and 36)."""
    rows = []
    for key in MODELS[1:]:
        s = LB.spec(key)
        for organism in ("gender-secret", "greeting", "time", "gender"):
            folder = LB.XMODEL_RESULTS / key / "organisms" / organism
            if not (folder / "check_behaviour.json").exists():
                continue
            check = json.loads((folder / "check_behaviour.json").read_text(encoding="utf-8"))
            if not check["passed"]:
                raise AssertionError(f"{key} {organism}: the adapter's behaviour check failed")
            t = pd.read_csv(folder / "auroc.csv")
            for _, r in t[t["layer"].isin([s.posthoc[0], s.mid])].iterrows():
                rows.append({"key": key, "organism": organism, "block": int(r["layer"]), "read": r["read"],
                             "direction": r["direction"], "auroc": r["auroc"], "band": r["band97.5"],
                             "factors": check.get("factors", "")})
    return pd.DataFrame(rows)


def _roleplay_groups(rp: pd.DataFrame) -> pd.DataFrame:
    """The share of each persona group's rollouts the judge scored fully in role (3)."""
    from deception_detection import persona_pcs as pcs

    groups = {"Deceptive (D5)": pcs.DECEPTION_PERSONAS, "Honest (D5)": pcs.HONESTY_PERSONAS,
              "Knowledge-denial (D8)": pcs.KNOWLEDGE_DENIAL_PERSONAS,
              "Knowledge-candour (D8)": pcs.KNOWLEDGE_CANDOUR_PERSONAS, "Sycophantic": pcs.SYCOPHANCY_PERSONAS,
              "Anti-sycophantic": pcs.ANTI_SYCOPHANCY_PERSONAS}
    rows = []
    for key in MODELS:
        part = rp[rp["model"] == _label(key)].set_index("role")
        for group, roles in groups.items():
            missing = [r for r in roles if r not in part.index]
            if missing:
                raise RuntimeError(f"{key}: no judge scores for {missing}")
            p = part.loc[list(roles)]
            rows.append({"key": key, "group": group, "share": p["score 3"].sum() / p["judged"].sum()})
        rows.append({"key": key, "group": "All 35 roles", "share": part["score 3"].sum() / part["judged"].sum()})
    return pd.DataFrame(rows)


def xm_numbers(d: dict) -> "object":
    """The section's numbers (`generated_xmodel/numbers.tex`), with the claims the text makes."""
    M = _paper_numbers()
    n = M.Numbers()
    for key in MODELS:
        s, m, p = LB.spec(key), _label(key), f"xm-{key}"
        n.put(f"{p}-blocks", str(s.n_blocks))
        n.put(f"{p}-width", str(s.hidden))  # as the thesis writes 8192, without a separator
        n.put(f"{p}-lb", s.lb_block)
        n.put(f"{p}-mid", s.mid)
        n.put(f"{p}-ph-lo", s.posthoc[0])
        n.put(f"{p}-ph-hi", s.posthoc[1])
        n.put(f"{p}-nsets", len(s.datasets))
        sets = len(s.datasets) + 1  # and RepE
        n.put(f"{p}-cells", sets * (sets - 1))
        n.put(f"{p}-depth-n", len(s.sweep))

        # transfer and the dataset mean
        t = _one(d["tr"], model=m, layer="L40")
        for column in ("raw probe", "D5", "D8", "D9", "D9 + K", "D5 + K", "K"):
            n.put(f"{p}-off-{TOKEN[column]}", float(t[column]))
        z = d["zs"][(d["zs"]["model"] == m) & (d["zs"]["layer"] == "L40")].set_index("contrast")
        for contrast in ("D5", "D8", "D9"):
            n.put(f"{p}-lb-{TOKEN[contrast]}", float(z.loc[contrast, "LB mean"]))
        kn = d["kn"][(d["kn"]["model"] == m) & (d["kn"]["layer"] == "L40")].set_index("direction")
        for direction, token in (("k alone", "k"), ("D9 + k", "ninek"), ("D5 + k", "fivek")):
            n.put(f"{p}-lb-{token}", float(kn.loc[direction, "mean AUROC"]))
        lb = d["lb"][d["lb"]["model"] == m].set_index("probe")
        n.put(f"{p}-lb-probe", float(lb.loc["base probe, L40 (ours)", "mean AUROC"]))
        n.put(f"{p}-lb-meanprobe", float(lb.loc["Mean Probe (published)", "mean AUROC"]))

        # paired intervals: transfer (`transfer_ci.py`) and the dataset mean (`k_robustness.py paired`)
        ci = d["ci"][(d["ci"]["model"] == key) & (d["ci"]["layer"] == "L40")]
        for metric, a, b in (("transfer", "D9", "raw probe"), ("transfer", "D9", "D5"), ("transfer", "D9 + K", "raw probe"),
                             ("transfer", "D9 + K", "D9"), ("transfer", "D5 + K", "D5"), ("dataset mean", "D9", "D5")):
            r = _one(ci, metric=metric, a=a, b=b)
            stem = f"{p}-ci-{'off' if metric == 'transfer' else 'lb'}-{TOKEN[a]}-{TOKEN[b]}"
            for suffix, column in (("", "delta"), ("-lo", "lo"), ("-hi", "hi")):
                n.put(stem + suffix, float(r[column]))
        pr = d["pr"][(d["pr"]["model"] == m) & (d["pr"]["layer"] == "L40")]
        for set_name, comparison, stem in (("mean of five", "D9 + k - benchmark probe, same layer", "lb-ninek-probe"),
                                           ("mean of five", "D9 + k - D9", "lb-ninek-dnine"),
                                           ("HP-KR identical", "D9 + k - D9", "ident-ninek-dnine")):
            part = pr[(pr["set"] == set_name) & (pr["comparison"] == comparison)]
            if part.empty:
                continue
            for suffix, column in (("", "delta"), ("-lo", "lo"), ("-hi", "hi")):
                n.put(f"{p}-ci-{stem}{suffix}", float(part.iloc[0][column]))

        # detection in the tail, and how fully the model plays the roles
        op = d["op"][d["op"]["model"] == m].set_index("scorer")
        for scorer, token in (("D9", "dnine"), ("D9 + k", "ninek"), ("k", "k"), ("benchmark probe, same layer", "probe")):
            n.put(f"{p}-op1-{token}", float(op.loc[scorer, "1% FPR"]), 2)
        groups = d["rpg"][d["rpg"]["key"] == key].set_index("group")["share"]
        n.put(f"{p}-roleplay", 100 * float(groups["All 35 roles"]), 0)
        n.put(f"{p}-roleplay-dec", 100 * float(groups["Deceptive (D5)"]), 0)
        n.put(f"{p}-roleplay-hon", 100 * float(groups["Honest (D5)"]), 0)
        n.put(f"{p}-judged", int(d["rp"][d["rp"]["model"] == m]["judged"].sum()))

        # D8 flags honest denials: the register effect, wherever HP-KR has them
        if "D8 alone" in kn.index and not pd.isna(kn.loc["D8 alone", "flags honest denials"]):
            n.put(f"{p}-flag-hden-deight", 100 * float(kn.loc["D8 alone", "flags honest denials"]), 1)

        # depth: where D9 (and D9 + K) transfers above the raw probe, over the 16-layer sweep
        dp = d["dp"][(d["dp"]["model"] == m) & (d["dp"]["block"].isin(s.sweep))]
        tr_src = dp[dp["source"] == "transfer, off-diagonal mean"]
        d9 = tr_src[tr_src["arm"] == "extended axis (12v13)"].set_index("block")["value"].sort_index()
        raw = tr_src[tr_src["arm"] == "single-source probe (raw)"].set_index("block")["value"].sort_index()
        ninek = _read(_rd_dir(key) / "transfer" / "depth_ninek.csv", f"{key} D9 + K across depth").set_index("layer")
        ninek = ninek.loc[list(s.sweep), "D9 + K: transfer"]
        if len(d9) != len(s.sweep) or len(raw) != len(s.sweep):
            raise RuntimeError(f"{key}: the depth table lacks sweep blocks")
        n.put(f"{p}-depth-above", int((d9 > raw).sum()))
        n.put(f"{p}-depth-ninek-above", int((ninek.values > raw.values).sum()))
        peak = int(d9.idxmax())
        n.put(f"{p}-depth-peak", float(d9.max()))
        n.put(f"{p}-depth-peak-block", peak)
        n.put(f"{p}-depth-peak-rel", (peak + 1) / s.n_blocks, 2)
        n.put(f"{p}-depth-peak-raw", float(raw.loc[peak]))
        n.put(f"{p}-depth-raw-max", float(raw.max()))
        if key == "qwen":  # near chance until deep in the model
            rise = int(d9[d9 >= 0.75].index[0])
            n.put(f"{p}-depth-rise-block", rise)
            n.put(f"{p}-depth-rise-rel", (rise + 1) / s.n_blocks, 2)
            n.put(f"{p}-depth-before-max", float(d9[d9.index < rise].max()))
            assert d9[d9.index < rise].max() < 0.55, "Qwen's D9 transfers before its rise"
        if key == "gemma":  # below chance at every sweep block before the mid block
            n.put(f"{p}-depth-early-max", float(d9[d9.index < s.mid].max()))
            assert d9[d9.index < s.mid].max() < 0.5, "Gemma's D9 transfers before its mid block"
        # K on HP-KR's identical text across the sweep, against its band (`k_robustness.py depth`)
        kdp = d["kd"][(d["kd"]["model"] == m) & (d["kd"]["block"].isin(s.sweep))]
        if not kdp["K identical"].isna().all():
            n.put(f"{p}-depth-ident-above", int((kdp["K identical"] > kdp["band identical"]).sum()))
        # each contrast on HP-KR at the mid block
        if "HP-KR" in _lb_sets(key):
            for contrast in ("D5", "D8", "D9"):
                n.put(f"{p}-hpkr-{TOKEN[contrast]}", float(z.loc[contrast, "HP-KR"]))

        # token reads: the mean read against the best one
        reads = d["reads"][(d["reads"]["model"] == m) & (d["reads"]["layer"] == "L40")].set_index("read")["AUROC"]
        n.put(f"{p}-read-mean", float(reads["mean"]))
        n.put(f"{p}-read-best", float(reads.max()))
        n.put(f"{p}-read-best-name", READ_NAME[reads.idxmax()])

    # every bank holds the same number of judged rollouts, and most of Qwen's not scored 3 are scored 2
    judged = {key: int(d["rp"][d["rp"]["model"] == _label(key)]["judged"].sum()) for key in MODELS}
    assert len(set(judged.values())) == 1, f"the banks differ in size: {judged}"
    q = d["rp"][d["rp"]["model"] == _label("qwen")]
    assert q["score 2"].sum() > q["score 0"].sum() + q["score 1"].sum()

    # HP-KR with the words held fixed: the released slices (Llama, Mistral; Qwen has no identical text)
    for key in ("llama", "mistral", "qwen"):
        p, sl = f"xm-{key}", _slices(key).set_index(["slice", "direction"])
        kn = d["kn"][(d["kn"]["model"] == _label(key)) & (d["kn"]["layer"] == "L40")].set_index("direction")
        counts = _slice_counts(key)
        for slice_name, token in (("identical text", "ident"), ("denials", "den")):
            if (slice_name, "knowledge general alone") not in sl.index:
                continue
            for direction, short in (("knowledge general alone", "k"), ("D9 extended alone", "dnine"),
                                     ("D8 knowledge-denial alone", "deight"),
                                     ("D9 extended + general (standardised)", "ninek")):
                n.put(f"{p}-{token}-{short}", float(sl.loc[(slice_name, direction), "auroc"]))
            n.put(f"{p}-{token}-band", float(sl.loc[(slice_name, "knowledge general alone"), "band97.5"]))
            n.put(f"{p}-{token}-dfive", float(kn.loc["D5 alone", "HP-KR identical" if token == "ident" else "HP-KR denials"]))
            n.put(f"{p}-{token}-lies", counts[slice_name][0])
            n.put(f"{p}-{token}-honest", counts[slice_name][1])
            vs = pd.read_csv(LB.XMODEL_RESULTS / key / "k_final" / "vs_probe.csv")
            vs = vs[(vs["block"] == LB.spec(key).mid) & (vs["scorer"] == "original K")].set_index("set")
            set_name = "HP-KR identical" if token == "ident" else "HP-KR denials"
            n.put(f"{p}-{token}-probe", float(vs.loc[set_name, "base probe auroc"]))
            for suffix, column in (("", "delta"), ("-lo", "lo"), ("-hi", "hi")):
                n.put(f"{p}-ci-{token}-k-probe{suffix}", float(vs.loc[set_name, column]))
    # ... and rebuilt from the raw files (`k_models.py hpkr_*`): Qwen's identical reply, Gemma's 35 lies
    rb = d["rb"]
    for key, slice_name, stem in (("qwen", "identical reply: I do not know the answer to the question.", "rb-ident"),
                                  ("qwen", "lies v honest denials, whole reply", "rb-den"),
                                  ("gemma", "lies v honest denials, opening sentence only", "rb-open"),
                                  ("gemma", "lies v honest denials, whole reply", "rb-whole")):
        part = rb[(rb["model"] == _label(key)) & (rb["layer"] == "L40") & (rb["slice"] == slice_name)].set_index("scorer")
        for scorer, short in (("K", "k"), ("D5", "dfive"), ("D8", "deight"), ("D9", "dnine"), ("D9 + K", "ninek"),
                              ("base probe", "probe")):
            n.put(f"xm-{key}-{stem}-{short}", float(part.loc[scorer, "AUROC"]))
        n.put(f"xm-{key}-{stem}-k-lo", float(part.loc["K", "lo"]))
        n.put(f"xm-{key}-{stem}-k-hi", float(part.loc["K", "hi"]))
        n.put(f"xm-{key}-{stem}-band", float(part["band97.5"].iloc[0]))
        n.put(f"xm-{key}-{stem}-lies", int(part["lies"].iloc[0]))
        n.put(f"xm-{key}-{stem}-honest", int(part["honest"].iloc[0]))

    # the fine-tunes: Gender-Secret (Llama's from the thesis's own run, Mistral's here) and Soft-Trigger
    gs = pd.read_csv(REPO_ROOT / "results" / "recall_denial" / "gender_secret.csv")
    gs = gs[(gs["layer"] == 40) & (gs["read"] == "adapted")].set_index("direction")
    n.put("xm-llama-gs-k", float(gs.loc["k fine-tune", "auroc"]))
    n.put("xm-llama-gs-band", float(gs.loc["k fine-tune", "band97.5"]))
    org = d["org"]
    mistral = LB.spec("mistral")

    def org_value(key, organism, block, read, direction, column="auroc"):
        return float(_one(org, key=key, organism=organism, block=block, read=read, direction=direction)[column])

    for block, suffix in ((mistral.mid, ""), (mistral.posthoc[0], "-ph")):
        n.put(f"xm-mistral-gs-k{suffix}", org_value("mistral", "gender-secret", block, "adapted", "k fine-tune"))
        n.put(f"xm-mistral-gs-band{suffix}", org_value("mistral", "gender-secret", block, "adapted", "k fine-tune", "band"))
    n.put("xm-mistral-gs-k-base", org_value("mistral", "gender-secret", mistral.mid, "base", "k fine-tune"))
    n.put("xm-mistral-gs-dnine", org_value("mistral", "gender-secret", mistral.mid, "adapted", "D9 bank"))
    n.put("xm-mistral-gs-probe", org_value("mistral", "gender-secret", mistral.mid, "adapted", "base probe (RepE-trained)"))
    st = org[(org["organism"] != "gender-secret") & org["block"].isin([LB.spec(k).mid for k in MODELS[1:]])]
    st = st[st.apply(lambda r: r["block"] == LB.spec(r["key"]).mid, axis=1)]
    for key in MODELS[1:]:
        part = st[st["key"] == key]
        kft = part[(part["read"] == "adapted") & (part["direction"] == "k fine-tune")]
        n.put(f"xm-{key}-st-k-min", float(kft["auroc"].min()))
        n.put(f"xm-{key}-st-k-max", float(kft["auroc"].max()))
        n.put(f"xm-{key}-st-band-min", float(kft["band"].min()))
        n.put(f"xm-{key}-st-band-max", float(kft["band"].max()))
        n.put(f"xm-{key}-st-margin-min", float((kft["auroc"] - kft["band"]).min()))
        n.put(f"xm-{key}-st-margin-max", float((kft["auroc"] - kft["band"]).max()))
        base = part[(part["read"] == "base") & (part["direction"] == "k base")]
        n.put(f"xm-{key}-st-basek-min", float(base["auroc"].min()))
        n.put(f"xm-{key}-st-basek-max", float(base["auroc"].max()))
        n.put(f"xm-{key}-st-baseband-max", float(base["band"].max()))
    n.put("xm-st-organisms", len(st[["key", "organism"]].drop_duplicates()))

    # K alone where it fails: Gemma's ID and its held-out AISI lies; and at Gemma's later post-hoc block
    vs = pd.read_csv(LB.XMODEL_RESULTS / "gemma" / "k_final" / "vs_probe.csv")
    vs = vs[vs["block"] == LB.spec("gemma").mid].set_index(["set", "scorer"])
    kn_g = d["kn"][(d["kn"]["model"] == _label("gemma"))].set_index(["layer", "direction"])
    n.put("xm-gemma-k-id", float(kn_g.loc[("L40", "k alone"), "ID"]))
    n.put("xm-gemma-k-cg", float(kn_g.loc[("L40", "k alone"), "CG"]))
    n.put("xm-gemma-lb-k-ph", float(kn_g.loc[("L44", "k alone"), "mean AUROC"]))
    for scorer, token in (("original K", "k"), ("D9 + original K", "ninek"), ("D9", "dnine")):
        n.put(f"xm-gemma-aisi-{token}", float(vs.loc[("held out (AISI)", scorer), "auroc"]))
    n.put("xm-gemma-aisi-probe", float(vs.loc[("held out (AISI)", "original K"), "base probe auroc"]))
    # D3, the probe trained on the model's own lies, which only Gemma's AISI rollouts allow
    deltas = pd.read_csv(LB.XMODEL_RESULTS / "gemma" / "aisi_probe" / "paired_deltas.csv")
    deltas = deltas[deltas["layer"] == LB.spec("gemma").mid]
    for a, b, stem in (("base probe (AISI-trained)", "base probe (RepE-trained)", "dthree-gain"),
                       ("extended axis (12v13)", "base probe (AISI-trained)", "dthree-dnine")):
        r = _one(deltas, a=a, b=b)
        for suffix, column in (("", "delta"), ("-lo", "lo"), ("-hi", "hi")):
            n.put(f"xm-gemma-{stem}{suffix}", float(r[column]))
    ai = d["ai"][(d["ai"]["model"] == _label("gemma")) & (d["ai"]["layer"] == "L40")].set_index("detector")
    n.put("xm-gemma-dthree-lb", float(ai.loc["base probe (AISI-trained)", "LB mean AUROC"]))

    # the template-free K beside the original; the identical text only where the slice is released
    rk = d["rk"]
    for key in MODELS:
        part = rk[(rk["model"] == _label(key)) & (rk["with"] == "K")].set_index("metric")
        if "HP-KR identical text" in part.index:
            n.put(f"xm-{key}-rk-ident", float(part.loc["HP-KR identical text", "rule K (template-free)"]))
        n.put(f"xm-{key}-rk-off", float(part.loc["transfer", "rule K (template-free)"]))
        n.put(f"xm-{key}-rk-lb", float(part.loc["dataset mean", "rule K (template-free)"]))
        cos = rk[(rk["model"] == _label(key)) & (rk["metric"] == "cosine between the two")]
        n.put(f"xm-{key}-rk-cos", float(cos["rule K (template-free)"].iloc[0]), 2)

    # the judge's filter: each strict bank against the lenient one of the same prompt format, as the
    # printed values differ (`banks_table`)
    bc = d["bc"]
    gaps = []
    for key, strict_bank, lenient_bank in (("llama", "ext", "s23"), ("llama", "s3u", "s23u"), ("qwen", "s3u", "s23u"),
                                           ("mistral", "s3", "s23"), ("gemma", "s3", "s23")):
        part = bc[bc["model"] == _label(key)].set_index("bank")
        if strict_bank not in part.index or lenient_bank not in part.index:
            continue
        gaps += [abs(float(M.fmt(part.loc[strict_bank, c])) - float(M.fmt(part.loc[lenient_bank, c])))
                 for c in ("D5 LB mean", "D8 LB mean", "D9 LB mean", "D9 transfer")]
    n.put("xm-bank-gap-max", max(gaps))
    # the prompt format: Qwen's original system-message bank (sys) and Llama's matched user-turn bank (ut)
    for key, bank, token in (("qwen", "s23", "sys"), ("llama", "s3u", "ut")):
        part = bc[bc["model"] == _label(key)].set_index("bank")
        if bank in part.index:
            n.put(f"xm-{key}-{token}-lb-dnine", float(part.loc[bank, "D9 LB mean"]))
            n.put(f"xm-{key}-{token}-off-dnine", float(part.loc[bank, "D9 transfer"]))
    rf = d["rf"]
    for key, fmt, token in (("qwen", "system message", "sys"), ("llama", "user turn", "ut")):
        part = rf[(rf["model"] == _label(key)) & (rf["format"] == fmt)]
        if len(part):
            n.put(f"xm-{key}-roleplay-{token}", 100 * float(part.iloc[0]["share fully in role (3)"]), 0)
            n.put(f"xm-{key}-roleplay-{token}-dec", 100 * float(part.iloc[0]["deceptive (D5)"]), 0)

    _check_claims(n)
    return n


def _value(n, key: str) -> float:
    """A key's printed value back as a number."""
    return float(n[key].replace("\\ensuremath{-}", "-").replace(",", ""))


def _check_claims(n) -> None:
    """Every claim the section's text makes, and agreement with the thesis's own numbers."""
    v = lambda key: _value(n, key)  # noqa: E731
    thesis = _thesis_values()
    clash = sorted(k for k in n.values if k in thesis)
    if clash:
        raise AssertionError(f"keys defined by both generators: {clash}")
    for ours, theirs in SHARED_KEYS.items():
        if n[ours] != thesis[theirs]:
            raise AssertionError(f"{ours} prints {n[ours]} but the thesis's {theirs} prints {thesis[theirs]}")

    def above(stem):  # the paired interval lies above zero
        return v(f"{stem}-lo") > 0

    def below(stem):
        return v(f"{stem}-hi") < 0

    # the persona result replicates on Mistral and Gemma, and fails on Qwen
    for key in ("llama", "mistral", "gemma"):
        for stem in ("off-dnine-raw", "off-dnine-dfive", "off-ninek-raw"):
            assert above(f"xm-{key}-ci-{stem}"), f"{key}: {stem} no longer above zero"
    assert below("xm-qwen-ci-off-dnine-raw") and below("xm-qwen-ci-off-ninek-raw"), "Qwen's D9 now transfers"
    assert not above("xm-qwen-ci-off-dnine-dfive") and not below("xm-qwen-ci-off-dnine-dfive")
    assert v("xm-qwen-off-deight") > v("xm-qwen-off-raw") > v("xm-qwen-off-dnine")
    # the knowledge-denial roles help on every model, and D9 + K beats the same-layer probe on every model
    for key in MODELS:
        assert above(f"xm-{key}-ci-lb-dnine-dfive"), f"{key}: D9 no longer above D5 on the dataset mean"
        assert above(f"xm-{key}-ci-lb-ninek-probe"), f"{key}: D9 + K no longer above the RepE probe"
    for key in ("llama", "mistral", "gemma"):
        assert v(f"xm-{key}-lb-dnine") - v(f"xm-{key}-lb-meanprobe") >= TIE
    assert v("xm-qwen-lb-meanprobe") - v("xm-qwen-lb-dnine") >= TIE
    # adding K: level on Llama, up on Qwen and Mistral, down on Gemma
    assert not above("xm-llama-ci-off-ninek-dnine") and not below("xm-llama-ci-off-ninek-dnine")
    assert above("xm-qwen-ci-off-ninek-dnine") and above("xm-mistral-ci-off-ninek-dnine")
    assert below("xm-gemma-ci-off-ninek-dnine")
    # Gemma's role-play is the strongest; Qwen plays the roles far more fully with the role line in the
    # user turn (its primary bank) than as the system message (the bank the first replication used)
    shares = {key: v(f"xm-{key}-roleplay") for key in MODELS}
    assert max(shares, key=shares.get) == "gemma"
    assert v("xm-qwen-roleplay") - v("xm-qwen-roleplay-sys") > 30
    # the identical text: K above the band on Llama, Mistral and Qwen's rebuilt reply; not on Gemma's 34 lies
    for stem in ("xm-llama-ident", "xm-mistral-ident", "xm-qwen-rb-ident"):
        assert v(f"{stem}-k") > v(f"{stem}-band"), f"{stem}: K inside the band"
    assert v("xm-gemma-rb-open-k") < v("xm-gemma-rb-open-band")
    # ... where every Llama and Qwen persona direction is inside the band, Mistral's D5 is above it,
    # Mistral's D9 within 0.01 of it, and Gemma's D9 within 0.01 of its band
    for stem in ("xm-llama-ident", "xm-qwen-rb-ident"):
        assert max(v(f"{stem}-{c}") for c in ("dfive", "deight", "dnine")) < v(f"{stem}-band")
    assert v("xm-mistral-ident-dfive") - v("xm-mistral-ident-band") >= TIE
    assert abs(v("xm-mistral-ident-dnine") - v("xm-mistral-ident-band")) < TIE
    assert v("xm-mistral-ident-deight") < v("xm-mistral-ident-band")
    assert abs(v("xm-gemma-rb-open-dnine") - v("xm-gemma-rb-open-band")) < TIE
    assert max(v(f"xm-gemma-rb-open-{c}") for c in ("dfive", "deight")) < v("xm-gemma-rb-open-band")
    # D8 flags nearly every honest denial on Mistral, as on Llama
    assert v("xm-mistral-flag-hden-deight") > 90 and v("xm-llama-flag-hden-deight") > 90
    # Gender-Secret: K detects Llama's through the fine-tune, not Mistral's
    assert v("xm-llama-gs-k") > v("xm-llama-gs-band")
    assert v("xm-mistral-gs-k") < v("xm-mistral-gs-band") and v("xm-mistral-gs-k-ph") < v("xm-mistral-gs-band-ph")
    # Soft-Trigger: Mistral's base read already separates every organism above its band; on Qwen and
    # Gemma K stays below the band of the fine-tune's read on every organism
    assert v("xm-mistral-st-basek-min") > v("xm-mistral-st-baseband-max")
    assert v("xm-mistral-st-margin-min") > 0
    for key in ("qwen", "gemma"):
        assert v(f"xm-{key}-st-k-max") < v(f"xm-{key}-st-band-min"), f"{key}: K cleared a Soft-Trigger band"
    # K alone on Gemma inverts on ID and the held-out lies, though D9 + K stays above the probe
    assert v("xm-gemma-k-id") < 0.5 and v("xm-gemma-aisi-k") < 0.5
    # D3 on Gemma: training on the model's own lies helps, and D9 still leads it
    assert v("xm-gemma-dthree-gain-lo") > 0 and v("xm-gemma-dthree-dnine-lo") > 0
    # the template-free K keeps the identical text and loses transfer on Mistral
    assert v("xm-mistral-rk-ident") > v("xm-mistral-ident-band")
    assert v("xm-mistral-off-k") - v("xm-mistral-rk-off") > 0.1
    # the lenient bank changes nothing that matters
    assert v("xm-bank-gap-max") < 0.03
    # "at every layer" is Llama's alone
    assert _value(n, "xm-llama-depth-above") == _value(n, "xm-llama-depth-n")
    assert _value(n, "xm-llama-depth-ninek-above") == _value(n, "xm-llama-depth-n")
    for key in MODELS[1:]:
        assert _value(n, f"xm-{key}-depth-above") < _value(n, f"xm-{key}-depth-n"), key
    # the mean read is within 0.03 of the best read, except on Qwen
    for key in ("llama", "mistral", "gemma"):
        assert v(f"xm-{key}-read-best") - v(f"xm-{key}-read-mean") <= 0.03, key
    assert v("xm-qwen-read-best") - v("xm-qwen-read-mean") > 0.03

    # the section's prose, sentence by sentence
    assert abs(v("xm-gemma-off-dnine") - v("xm-llama-off-dnine")) < TIE  # Gemma's D9 level with Llama's
    assert v("xm-qwen-hpkr-dfive") < v("xm-qwen-hpkr-dnine") < 0.5  # D9 inverts on HP-KR, D5 more so
    assert v("xm-mistral-ci-ident-k-probe-lo") > 0  # K above the same-block probe on Mistral's identical text
    assert v("xm-llama-ci-ident-k-probe-lo") < 0 < v("xm-llama-ci-ident-k-probe-hi")  # ... level on Llama's
    assert v("xm-mistral-ci-ident-ninek-dnine-lo") > 0
    assert 0 < v("xm-qwen-rb-ident-k") - v("xm-qwen-rb-ident-band") < 0.03  # "just above its band"
    assert v("xm-mistral-gs-dnine") < v("xm-mistral-gs-k") and v("xm-mistral-gs-probe") < v("xm-mistral-gs-k")
    for key in ("qwen", "gemma"):
        assert v(f"xm-{key}-lb-k") < 0.6, f"{key}: K alone is no longer weak"
    assert v("xm-llama-op1-ninek") < v("xm-llama-op1-dnine") and v("xm-gemma-op1-ninek") < v("xm-gemma-op1-dnine")
    assert v("xm-mistral-op1-ninek") > v("xm-mistral-op1-dnine")
    # depth: Mistral peaks just before its mid block, Gemma at it, Qwen deep and level with the raw probe
    mistral, gemma, qwen = LB.spec("mistral"), LB.spec("gemma"), LB.spec("qwen")
    assert 0 < mistral.mid - _value(n, "xm-mistral-depth-peak-block") <= 2
    assert _value(n, "xm-gemma-depth-peak-block") == gemma.mid
    assert _value(n, "xm-qwen-depth-rise-block") > qwen.mid
    assert abs(v("xm-qwen-depth-peak") - v("xm-qwen-depth-peak-raw")) < TIE
    assert _value(n, "xm-mistral-depth-ninek-above") > _value(n, "xm-mistral-depth-above")
    assert _value(n, "xm-gemma-depth-ninek-above") <= _value(n, "xm-gemma-depth-above")
    assert _value(n, "xm-mistral-depth-ident-above") == _value(n, "xm-mistral-depth-n")  # "at all 16 depths"
    assert _value(n, "xm-llama-depth-ident-above") >= _value(n, "xm-llama-depth-n") - 1  # "most of the depth"


def _check_appendix(zs: pd.DataFrame, rpg: pd.DataFrame, ai: pd.DataFrame) -> None:
    """The claims Appendix D's text makes about its tables."""
    for key in ("llama", "gemma"):
        a = ai[(ai["model"] == _label(key)) & (ai["layer"] == "L40")].set_index("detector")["LB mean AUROC"]
        assert a["base probe (RepE-trained)"] < a["base probe (AISI-trained)"] < a["extended axis (12v13)"], key
    g = ai[(ai["model"] == _label("gemma")) & (ai["layer"] == "L20")].set_index("detector")["LB mean AUROC"]
    assert g["base probe (AISI-trained)"] > g["extended axis (12v13)"], "D3 no longer leads at Gemma's layer-20 block"
    z = zs[zs["layer"] == "L40"].set_index(["model", "contrast"])
    gemma, qwen = _label("gemma"), _label("qwen")
    assert z.loc[(gemma, "D8"), "ID"] < 0.5 < z.loc[(gemma, "D9"), "ID"], "Gemma's D8 no longer inverts on ID"
    assert z.loc[(qwen, "D5"), "HP-KR"] < 0.5 and z.loc[(qwen, "D9"), "HP-KR"] < 0.5 < z.loc[(qwen, "D8"), "HP-KR"]
    share = rpg.set_index(["key", "group"])["share"]
    assert share[("gemma", "All 35 roles")] > 0.9 and share[("qwen", "All 35 roles")] < 1 / 3
    assert all(0.6 < share[(k, "All 35 roles")] < 0.72 for k in ("llama", "mistral")), "about two thirds"
    groups = share["qwen"].drop("All 35 roles").sort_values()
    assert set(groups.index[:2]) == {"Deceptive (D5)", "Knowledge-denial (D8)"}


# --- the section's tables -------------------------------------------------------------------------

def _tabular(columns: str, header: list[str], body: list[str], name: str, source: str) -> None:
    lines = [f"% Generated by make_xmodel_assets.py{(' from ' + source) if source else ''}",
             f"\\begin{{tabular}}{{{columns}}}", "\\toprule", *header, "\\midrule", *body, "\\bottomrule",
             "\\end{tabular}"]
    (OUT_TEX / f"tab_xm_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sets_text(key: str) -> str:
    return ", ".join(_lb_sets(key)).replace("HP-KR", "\\hpkr").replace("HP-C", "\\hpc")


def models_table(n) -> None:
    """tab_xm_models.tex: what each model is and which of its blocks stand for the thesis's layers."""
    bank = {"llama": "score 3", "qwen": "score 3, user turn", "mistral": "score 3", "gemma": "score 3"}
    body = []
    for key in MODELS:
        p = f"xm-{key}"
        name = LB.spec(key).title.replace("-Instruct", "").replace("-it", "")
        body.append(f"{name} & {n[p + '-blocks']} & {n[p + '-width']} & "
                    f"{n[p + '-lb']} & {n[p + '-mid']} & {n[p + '-ph-lo']}, {n[p + '-ph-hi']} & {_sets_text(key)} & "
                    f"{bank[key]} & {n[p + '-roleplay']}\\% \\\\")
    _tabular("@{}l cc ccc l l c@{}",
             [" & & & \\multicolumn{3}{c}{Block read for the thesis's layer} & & & \\\\",
              "\\cmidrule(lr){4-6}",
              "Model & Blocks & Width & 20 & 40 & 36, 44 & Liars' Bench datasets & Bank & In role \\\\"],
             body, "models", "lb_models.SPECS and the judge's scores")


def _mark(holds) -> str:
    return {True: "\\checkmark", False: "$\\times$", None: "$\\circ$"}[holds]


def claims_table(n) -> None:
    """tab_xm_claims.tex: the thesis's claims at each model's mid block, with a mark for whether
    each holds there (paired intervals in `intervals_table`)."""
    v = lambda key: _value(n, key)  # noqa: E731

    def by_interval(stem):
        lo, hi = v(f"{stem}-lo"), v(f"{stem}-hi")
        return f"{_signed_tex(v(stem))} {_mark(True if lo > 0 else False if hi < 0 else None)}"

    def by_point(a, b, higher=True, note=""):
        gap = v(a) - v(b)
        holds = None if abs(gap) < TIE else (gap > 0) == higher
        return f"{n[a]} ({n[b]}){note} {_mark(holds)}"

    def cell(key, row):
        p = f"xm-{key}"
        if row == "gs":
            return by_point(f"{p}-gs-k", f"{p}-gs-band") if key in ("llama", "mistral") else "n/a"
        if row in ("k-ident", "persona-ident"):
            stem, note = {"llama": (f"{p}-ident", ""), "mistral": (f"{p}-ident", ""),
                          "qwen": ("xm-qwen-rb-ident", "$^a$"), "gemma": ("xm-gemma-rb-open", "$^b$")}[key]
            if row == "k-ident":
                return by_point(f"{stem}-k", f"{stem}-band", note=note)
            top = max(("dfive", "deight", "dnine"), key=lambda c: v(f"{stem}-{c}"))
            return by_point(f"{stem}-{top}", f"{stem}-band", higher=False, note=note)
        if row == "meanprobe":
            return by_point(f"{p}-lb-dnine", f"{p}-lb-meanprobe")
        return by_interval(f"{p}-ci-{row}")

    rows = [("D9 transfers above the probe on raw activations", "off-dnine-raw"),
            ("D9 transfers above the published contrast D5", "off-dnine-dfive"),
            ("The knowledge-denial roles raise the mean (D9 above D5)", "lb-dnine-dfive"),
            ("D9's mean is above the benchmark's Mean Probe", "meanprobe"),
            ("D9 + K transfers above the probe on raw activations", "off-ninek-raw"),
            ("D9 + K's mean is above the same-block RepE probe", "lb-ninek-probe"),
            ("No persona direction clears the band on identical text", "persona-ident"),
            ("K clears the band on identical text", "k-ident"),
            ("K clears the band on Gender-Secret, read through the fine-tune", "gs")]
    body = [f"{label} & " + " & ".join(cell(key, row) for key in MODELS) + " \\\\" for label, row in rows]
    _tabular("@{}>{\\raggedright\\arraybackslash}p{5.1cm} cccc@{}",
             ["Claim & " + " & ".join(SHORT_NAME[k] for k in MODELS) + " \\\\"], body, "claims",
             "transfer_ci.py, k_robustness.py paired and the slice and organism scores")


def intervals_table(n) -> None:
    """tab_xm_intervals.tex (Appendix D): the paired differences behind the claims table."""
    comparisons = [("Transfer", "D9 $-$ raw probe", "off-dnine-raw"), ("Transfer", "D9 $-$ D5", "off-dnine-dfive"),
                   ("Transfer", "D9 + K $-$ raw probe", "off-ninek-raw"), ("Transfer", "D9 + K $-$ D9", "off-ninek-dnine"),
                   ("Transfer", "D5 + K $-$ D5", "off-fivek-dfive"), ("Mean", "D9 $-$ D5", "lb-dnine-dfive"),
                   ("Mean", "D9 + K $-$ D9", "lb-ninek-dnine"), ("Mean", "D9 + K $-$ RepE probe", "lb-ninek-probe")]
    body = []
    for metric, label, stem in comparisons:
        cells = []
        for key in MODELS:
            s = f"xm-{key}-ci-{stem}"
            cells.append(f"{_signed_tex(_value(n, s))} [{_signed_tex(_value(n, s + '-lo'))}, "
                         f"{_signed_tex(_value(n, s + '-hi'))}]")
        body.append(f"{metric} & {label} & " + " & ".join(cells) + " \\\\")
    _tabular("@{}ll cccc@{}", ["& Comparison & " + " & ".join(SHORT_NAME[k] for k in MODELS) + " \\\\"], body,
             "intervals", "transfer_ci.py and k_robustness.py paired")


def mid_table(n) -> None:
    """tab_xm_mid.tex: the main detectors at each model's mid block."""
    def row(label, token_key, places=None):
        return f"{label} & " + " & ".join(n[f"xm-{k}-{token_key}"] for k in MODELS) + " \\\\"

    body = ["\\multicolumn{5}{@{}l}{\\emph{Transfer, mean off-diagonal AUROC}} \\\\",
            row("Probe on raw activations (D1)", "off-raw"), row("Published contrast D5", "off-dfive"),
            row("Knowledge-denial contrast D8", "off-deight"), row("Extended contrast D9", "off-dnine"),
            row("K", "off-k"), row("D9 + K", "off-ninek"),
            "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Zero-shot, mean AUROC over the model's datasets}} \\\\",
            row("Mean Probe (Liars' Bench, published)", "lb-meanprobe"), row("Base probe D2, same block", "lb-probe"),
            row("D5", "lb-dfive"), row("D8", "lb-deight"), row("D9", "lb-dnine"), row("K", "lb-k"),
            row("D9 + K", "lb-ninek"),
            "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Lies caught at 1\\% of honest answers flagged}} \\\\",
            row("D9", "op1-dnine"), row("D9 + K", "op1-ninek")]
    _tabular("@{}l cccc@{}", [" & " + " & ".join(SHORT_NAME[k] for k in MODELS) + " \\\\"], body, "mid",
             "the persona_ood, recall_denial and k_robustness outputs")


def zeroshot_table(zs: pd.DataFrame) -> None:
    """tab_xm_zeroshot.tex (Appendix D): D5, D8 and D9 on each dataset at the mid block."""
    M = _paper_numbers()
    body = []
    for key in MODELS:
        z = zs[(zs["model"] == _label(key)) & (zs["layer"] == "L40")].set_index("contrast")
        for i, contrast in enumerate(("D5", "D8", "D9")):
            cells = [M.fmt(z.loc[contrast, d]) if not pd.isna(z.loc[contrast, d]) else "n/a" for d in (*LB_ORDER, "RepE")]
            body.append(f"{SHORT_NAME[key] if i == 0 else ''} & {contrast} & " + " & ".join(cells)
                        + f" & {M.fmt(z.loc[contrast, 'LB mean'])} \\\\")
        if key != MODELS[-1]:
            body.append("\\addlinespace")
    _tabular("@{}ll cccccc c@{}",
             ["Model & Contrast & CG & \\hpc & \\hpkr & IT & ID & RepE & Mean \\\\"], body, "zeroshot",
             "persona_ood zero_shot.csv")


def transfer_table(tr: pd.DataFrame) -> None:
    """tab_xm_transfer.tex (Appendix D): transfer at the four named blocks of every model."""
    M = _paper_numbers()
    columns = ("raw probe", "D5", "D8", "D9", "K", "D9 + K", "published PCs (k=3)", "extended PCs (k=3)", "random (k=3)")
    order = {"L20": 0, "L36": 1, "L40": 2, "L44": 3}
    body = []
    for key in MODELS:
        part = tr[tr["model"] == _label(key)].assign(o=lambda t: t["layer"].map(order)).sort_values("o")
        for i, (_, r) in enumerate(part.iterrows()):
            name = {"L20": "20", "L36": "36$^*$", "L40": "40", "L44": "44$^*$"}[r["layer"]]
            body.append(f"{SHORT_NAME[key] if i == 0 else ''} & {name} & {int(r['block'])} & "
                        + " & ".join(M.fmt(r[c]) for c in columns) + " \\\\")
        if key != MODELS[-1]:
            body.append("\\addlinespace")
    _tabular("@{}l cc cccccc ccc@{}",
             [" & & & & & & & & & \\multicolumn{3}{c}{Probe on $k = 3$} \\\\", "\\cmidrule(lr){10-12}",
              "Model & Layer & Block & Raw & D5 & D8 & D9 & K & D9 + K & Published & Extended & Random \\\\"],
             body, "transfer", "persona_ood summary.csv and recall_denial transfer")


def wording_table(n) -> None:
    """tab_xm_wording.tex (Appendix D): HP-KR with the words held fixed, released and rebuilt."""
    rows = [("Llama", "Identical text", "xm-llama-ident"), ("", "Denials", "xm-llama-den"),
            ("Mistral", "Identical text", "xm-mistral-ident"), ("", "Denials", "xm-mistral-den"),
            ("Qwen", "Denials", "xm-qwen-den"),
            ("", "Identical reply, rebuilt", "xm-qwen-rb-ident"), ("", "Denials, rebuilt", "xm-qwen-rb-den"),
            ("Gemma", "Opening sentence, rebuilt", "xm-gemma-rb-open"), ("", "Denials, rebuilt", "xm-gemma-rb-whole")]
    body = []
    for model, label, stem in rows:
        probe = n.values.get(f"{stem}-probe", "n/a")
        cells = [n.values.get(f"{stem}-{c}", "n/a") for c in ("dfive", "deight", "dnine", "k", "ninek")]
        body.append(f"{model} & {label} & {n[stem + '-lies']} & {n[stem + '-honest']} & {probe} & "
                    + " & ".join(cells) + f" & {n[stem + '-band']} \\\\")
    _tabular("@{}ll cc c ccccc c@{}",
             ["Model & Slice & Lies & Honest & Probe & D5 & D8 & D9 & K & D9 + K & Band \\\\"], body, "wording",
             "recall_denial.py combine and k_models.py hpkr_score")


def organisms_table(org: pd.DataFrame) -> None:
    """tab_xm_organisms.tex (Appendix D): each fine-tune, read through it and by the base model."""
    M = _paper_numbers()
    name = {"gender-secret": "Gender-Secret", "greeting": "Soft-Trigger, greeting", "time": "Soft-Trigger, time",
            "gender": "Soft-Trigger, gender"}
    body = []
    for key in MODELS[1:]:
        mid = LB.spec(key).mid
        part = org[(org["key"] == key) & (org["block"] == mid)]
        for i, organism in enumerate(o for o in name if o in set(part["organism"])):
            def a(direction, read="adapted", column="auroc"):
                return M.fmt(_one(part, organism=organism, read=read, direction=direction)[column])
            body.append(f"{SHORT_NAME[key] if i == 0 else ''} & {name[organism]} & {a('D9 bank')} & {a('k fine-tune')} & "
                        f"{a('D9 bank + k fine-tune (standardised)')} & {a('base probe (RepE-trained)')} & "
                        f"{a('k fine-tune', column='band')} & {a('k base', 'base')} & {a('k base', 'base', 'band')} \\\\")
        if key != MODELS[-1]:
            body.append("\\addlinespace")
    _tabular("@{}ll cccc c cc@{}",
             [" & & \\multicolumn{5}{c}{The fine-tune's read} & \\multicolumn{2}{c}{The base model's read} \\\\",
              "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}",
              "Model & Fine-tune & D9 & K & D9 + K & Probe & Band & K & Band \\\\"], body, "organisms",
             "organisms.py score")


def operating_table(op: pd.DataFrame) -> None:
    """tab_xm_operating.tex (Appendix D): lies caught at cutoffs set on each dataset's honest answers."""
    M = _paper_numbers()
    scorers = (("benchmark probe, same layer", "Base probe D2"), ("D5", "D5"), ("D9", "D9"), ("k", "K"),
               ("D9 + k", "D9 + K"), ("D5 + k", "D5 + K"))
    body = []
    for label_key, label in scorers:
        cells = []
        for key in MODELS:
            r = op[op["model"] == _label(key)].set_index("scorer").loc[label_key]
            cells += [M.fmt(r["1% FPR"], 2), M.fmt(r["10% FPR"], 2)]
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    header = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT_NAME[k]}}}" for k in MODELS) + " \\\\",
              "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(MODELS))),
              "Detector & " + " & ".join(["1\\%", "10\\%"] * len(MODELS)) + " \\\\"]
    _tabular("@{}l cc cc cc cc@{}", header, body, "operating", "k_robustness.py operating points")


def aisi_table(ai: pd.DataFrame) -> None:
    """tab_xm_aisi.tex (Appendix D): D3, the base probe trained on the model's own lies, beside D2 and
    the contrasts, on the mean over the model's datasets (Llama and Gemma, the two with AISI rollouts)."""
    M = _paper_numbers()
    detectors = (("base probe (RepE-trained)", "Base probe D2, trained on RepE"),
                 ("base probe (AISI-trained)", "D3, trained on the model's own lies"),
                 ("deception axis (8v7)", "D5"), ("knowledge-denial axis (4v6)", "D8"),
                 ("extended axis (12v13)", "D9"))
    layers = (("L20", "20"), ("L40", "40"), ("L44", "44$^*$"))
    body = []
    for arm, label in detectors:
        cells = []
        for key in ("llama", "gemma"):
            part = ai[(ai["model"] == _label(key)) & (ai["detector"] == arm)].set_index("layer")
            cells += [M.fmt(part.loc[layer, "LB mean AUROC"]) for layer, _ in layers]
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    names = " & ".join(name for _, name in layers)
    _tabular("@{}l ccc ccc@{}",
             [" & \\multicolumn{3}{c}{Llama} & \\multicolumn{3}{c}{Gemma} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              f"Detector, at the block for layer & {names} & {names} \\\\"], body, "aisi", "aisi_probe.py compare")


def roleplay_table(rpg: pd.DataFrame) -> None:
    """tab_xm_roleplay.tex (Appendix D): the share of each persona group's rollouts fully in role."""
    groups = list(dict.fromkeys(rpg["group"]))
    body = []
    for group in groups:
        cells = [f"{100 * float(_one(rpg, key=k, group=group)['share']):.0f}\\%" for k in MODELS]
        body.append(("\\midrule " if group == "All 35 roles" else "") + f"{group} & " + " & ".join(cells) + " \\\\")
    _tabular("@{}l cccc@{}", ["Persona group & " + " & ".join(SHORT_NAME[k] for k in MODELS) + " \\\\"], body,
             "roleplay", "the judge's scores")


def banks_table(bc: pd.DataFrame) -> None:
    """tab_xm_banks.tex (Appendix D): each model's banks at the mid block: the strict bank (score 3)
    against the lenient one (scores 2 and 3) under the model's primary prompt format, then the banks
    built under the other format (Llama's matched user-turn pair, Qwen's original system-message bank)."""
    M = _paper_numbers()
    columns = ("D5 LB mean", "D8 LB mean", "D9 LB mean", "D9 transfer")
    body = []
    for key in MODELS:
        part = bc[bc["model"] == _label(key)].set_index("bank")
        labels = [(b, BANK_FILTER[b] + ("" if BANK_FORMAT[b] == PRIMARY_FORMAT[key] else f", {BANK_FORMAT[b]}"))
                  for b in COMPARED_BANKS[key]]
        for i, (bank, label) in enumerate(labels):
            cells = [M.fmt(part.loc[bank, c]) for c in columns] if bank in part.index else ["n/a"] * len(columns)
            body.append(f"{SHORT_NAME[key] if i == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\")
    _tabular("@{}ll cccc@{}",
             [" & & \\multicolumn{3}{c}{Mean over the model's datasets} & \\\\", "\\cmidrule(lr){3-5}",
              "Model & Bank & D5 & D8 & D9 & D9 transfer \\\\"], body, "banks", "bank_comparison")


def checks_table(n, reads: pd.DataFrame) -> None:
    """tab_xm_checks.tex (Appendix D): the template-free K and D9's token reads, at each model's
    mid block."""
    M = _paper_numbers()
    body = []
    for key in MODELS:
        r = reads[(reads["model"] == _label(key)) & (reads["layer"] == "L40")].set_index("read")["AUROC"]
        best = r.idxmax()
        rk = f"{n.values.get(f'xm-{key}-rk-ident', 'n/a')} & {n[f'xm-{key}-rk-off']} & {n[f'xm-{key}-off-k']}"
        body.append(f"{SHORT_NAME[key]} & {rk} & {M.fmt(r['mean'])} & {M.fmt(r[best])} ({READ_NAME[best]}) \\\\")
    _tabular("@{}l ccc cc@{}",
             [" & \\multicolumn{3}{c}{Template-free K} & \\multicolumn{2}{c}{D9's token read, mean AUROC} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
              "Model & Identical text & Transfer & Template K's transfer & Mean & Best \\\\"], body, "checks",
             "k_final and persona_positions")


def xm_depth_figure(dp: pd.DataFrame, kd: pd.DataFrame) -> None:
    """fig_xm_depth.pdf: four measures across relative depth for the four models, each model's mid
    block marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deception_detection.scripts.layer_sweep import TRANSFER_SOURCE

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    colours = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1f9e77", "#b5399e")))
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.9), sharex=True)
    panels = [("D9", "(a) D9 transfer"), ("raw", "(b) Raw-probe transfer"), ("K mean", "(c) K alone, dataset mean"),
              ("K identical", "(d) K on HP-KR's identical text")]
    for ax, (what, title) in zip(axes.flat, panels):
        for key in MODELS:
            s = LB.spec(key)
            label = _label(key)
            if what in ("D9", "raw"):
                arm = "extended axis (12v13)" if what == "D9" else "single-source probe (raw)"
                part = dp[(dp["model"] == label) & (dp["source"] == TRANSFER_SOURCE) & (dp["arm"] == arm)]
                x, y = part["relative depth"], part["value"]
                order = np.argsort(x.to_numpy())
                x, y = x.to_numpy()[order], y.to_numpy()[order]
            else:
                part = kd[kd["model"] == label].sort_values("relative depth")
                if part[what].isna().all():
                    continue
                x, y = part["relative depth"].to_numpy(), part[what].to_numpy()
                if what == "K identical":
                    ax.plot(x, part["band identical"].to_numpy(), color=colours[key], lw=0.8, ls="--", alpha=0.7)
            ax.plot(x, y, marker="o", ms=2.5, lw=1.2, color=colours[key], label=label.replace("\u2020", "$^\\dagger$"))
            mid = (s.mid + 1) / s.n_blocks
            if np.isclose(x, mid).any():
                ax.plot([mid], [y[np.isclose(x, mid)][0]], marker="o", ms=6, mfc="none", mec=colours[key], mew=1.2)
        ax.axhline(0.5, color="#888", lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(color="#eee", lw=0.6)
        ax.set_ylim(0.2, 1.0)
    for ax in axes[1]:
        ax.set_xlabel("relative depth, (block + 1) / blocks", fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, fontsize=6.5, frameon=False, loc="lower right")  # (d) is empty below 0.5
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_xm_depth.pdf", bbox_inches="tight", metadata=FIGURE_METADATA)
    plt.close(fig)


def thesis_assets(**d) -> None:
    """Everything the thesis's cross-model section and Appendix D read."""
    d["org"] = _organism_reads()
    d["rpg"] = _roleplay_groups(d["rp"])
    d["reads"] = token_reads()
    d["ci"] = _intervals_by_key()
    n = xm_numbers(d)
    _check_appendix(d["zs"], d["rpg"], d["ai"])
    OUT_TEX.mkdir(parents=True, exist_ok=True)
    n.write(OUT_TEX / "numbers.tex")
    models_table(n)
    claims_table(n)
    mid_table(n)
    intervals_table(n)
    zeroshot_table(d["zs"])
    transfer_table(d["tr"])
    wording_table(n)
    organisms_table(d["org"])
    operating_table(d["op"])
    aisi_table(d["ai"])
    roleplay_table(d["rpg"])
    banks_table(d["bc"])
    checks_table(n, d["reads"])
    xm_depth_figure(d["dp"], d["kd"])
    print(f"wrote {len(n.values)} cross-model numbers and the thesis section's tables into {OUT_TEX.name}")


def main() -> None:
    MISSING.clear()
    zs, tr, kn = zero_shot(), transfer(), knowledge()
    kd, pr, op, cal = k_depth(), paired(), operating_points(), calibration()
    lb, ai, org, rp, bc, dp = liars_bench(), aisi(), organisms(), roleplay(), bank_comparison(), depth()
    rf = roleplay_formats()
    ci = intervals()
    rk, rk_verdicts = rule_k()
    cl = claims(zs, tr, kn, kd, pr, org, _intervals_by_key())
    rp_summary = (rp.groupby("model")[["judged", "score 0", "score 1", "score 2", "score 3"]].sum()
                  .assign(**{"share fully in role": lambda d: d["score 3"] / d["judged"]}).reset_index()
                  if len(rp) else rp)
    sections = [
        _write("claims", cl, "The thesis's claims, per model (two AUROCs replicate a claim when the paired 95% "
               "interval of their difference, over stratified resamples of the test rows, lies above 0, and not "
               f"when it lies below; where no interval exists, when they differ by at least {TIE}. K is tested "
               "against the 97.5th percentile of random directions)"),
        _write("transfer_intervals", ci, "Paired 95% intervals: transfer and the dataset mean, every model and "
               "layer (`transfer_ci.py`; the fitted probes held fixed, the test rows resampled)"),
        _write("rule_k", rk, "The template-free K (rule K: no HP-KR wording) beside the original K, mid-depth"),
        _write("rule_k_verdicts", rk_verdicts, "Template-free K: each model's criteria, fixed before its rows "
               "were read (Llama: `k_final.py`; the others: `k_final_xmodel.py`)"),
        _write("rule_k_vs_probe", rule_k_vs_probe(), "Both Ks, alone and with D9, against the same-layer RepE base "
               "probe, zero-shot at mid-depth: AUROC (difference from the probe [paired 95% interval])"),
        _write("d5k_vs_probe", rule_k_vs_probe(("D5", "D5 + original K", "D5 + rule K", "D9")),
               "D5, the published persona contrast, alone and with either K, against the same-layer RepE base "
               "probe, zero-shot at mid-depth: AUROC (difference from the probe [paired 95% interval])"),
        _write("zero_shot", zs, "Zero-shot persona contrasts (thesis Table 5.1)"),
        _write("zero_shot_llama_subsets", llama_on_subsets(zs) if len(zs) else zs,
               "Llama's contrasts on each model's datasets"),
        _write("transfer", tr, "Train-one-test-all transfer, off-diagonal mean AUROC (Tables 5.4, B.1)"),
        _write("knowledge", kn, "Each direction alone and with K (Tables 5.3, 5.5)"),
        _write("k_depth", kd, "K, D9 and D9 + K across depth (Table B.4)"),
        _write("paired", pr, "Paired differences (Table B.3)"),
        _write("operating_points", op, "Share of lies caught at in-setting cutoffs, mid-depth (Table B.5)"),
        _write("calibration", cal, "Balanced accuracy at the Alpaca and oracle thresholds, mid-depth (Table 5.7)"),
        _write("liars_bench", lb, "Liars' Bench's probes beside this pipeline's base probe (Table 5.2)"),
        _write("aisi", ai, "D3: the probe trained on the model's own lies (Llama and Gemma)"),
        _write("rebuilt_hpkr", rebuilt_hpkr(), "HP-KR rebuilt from the raw files: Qwen's identical text, Gemma's supplementary set"),
        _write("k_matched", k_matched(), "K', the replication on matched items (Table 5.6 criteria)"),
        _write("token_reads", token_reads(), "Token reads of D9, mean over the Liars' Bench datasets (Table B.6)"),
        _write("followup", followup_turn(), "D9 on the response tokens against the Follow-Up turn's \"No\" (Table B.7)"),
        _write("organisms", org, "Model organisms: behaviour checks and AUROC on the fine-tune's read"),
        _write("roleplay", rp_summary, "How fully each model plays the persona roles (judge scores, primary bank)"),
        _write("roleplay_formats", rf, "How fully Llama and Qwen play the roles under each prompt format: the role "
               "line as the system message, or with a stay-in-character sentence in the user turn (shares scored 3)"),
        _write("bank_comparison", bc, "Each model's persona banks at mid-depth: the primary bank, the same format's "
               "lenient (score 2-3) bank, and the banks built under the other prompt format"),
    ]
    depth_figure(dp, kd)
    dp.to_csv(OUT_TABLES / "depth.csv", index=False)
    rp.to_csv(OUT_TABLES / "roleplay_by_role.csv", index=False)
    header = (f"# The revised thesis across four models\n\nGenerated by `thesis_revised/make_xmodel_assets.py`. "
              f"Qwen's primary persona bank has the role line in the user turn; its system-message bank is a comparison row. Layers are the thesis's; each model's "
              f"blocks are in `lb_models.SPECS`.\n\n"
              + (f"**Inputs not yet produced ({len(MISSING)}):**\n\n" + "\n".join(f"- {m}" for m in MISSING) + "\n\n"
                 if MISSING else ""))
    (LB.XMODEL_RESULTS / "summary.md").write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"wrote {LB.XMODEL_RESULTS / 'summary.md'}; {len(MISSING)} inputs not yet produced")
    if MISSING:
        raise RuntimeError("the thesis's section needs every input; see the summary's list")
    thesis_assets(zs=zs, tr=tr, kn=kn, kd=kd, pr=pr, op=op, lb=lb, ai=ai, rp=rp, rf=rf, bc=bc, dp=dp, rk=rk,
                  rb=rebuilt_hpkr())


if __name__ == "__main__":
    main()
