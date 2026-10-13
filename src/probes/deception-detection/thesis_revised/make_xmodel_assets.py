"""The revised thesis's tables and figures across the four Liars' Bench models.

The thesis studies Llama-3.3-70B-Instruct. Its studies were re-run on Qwen2.5-72B-Instruct,
Mistral-Small-3.1-24B-Instruct and Gemma-3-27B-it with the same code (`lb_models.SPECS` maps each
model's blocks to the thesis's layers 20/36/40/44 and its 16-layer sweep). This collects every
model's outputs beside Llama's and writes, without touching the thesis's own assets:

    results/xmodel/tables/*.csv              every table, tidy
    results/xmodel/summary.md                the tables, and each thesis claim checked per model
    thesis_revised/generated_xmodel/*.tex    the tables as LaTeX (booktabs)
    thesis_revised/images_xmodel/*.pdf       the depth figures (x = relative depth)

Primary persona banks: Llama's thesis bank and the strict (judge score 3) banks of Mistral and
Gemma; Qwen's is its lenient bank (score 2-3; its strict bank lacks 11 roles), marked with a dagger.
The lenient bank of all four models is reported separately (`bank_comparison`). Where the thesis
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
    s = LB.spec(key)
    return s.short + ("\u2020" if key == "qwen" else "")


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
        s, b = LB.spec(key), _bank(key, bank if key != "llama" or bank in (None, "ext", "s23") else None)
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

SCORE_DIRS = {"llama": LB.XMODEL_ROOT / "banks" / "llama-70b-3.3" / "scores",
              "qwen": LB.XMODEL_ROOT / "banks" / "qwen-72b" / "scores",
              "mistral": LB.XMODEL_ROOT / "banks" / "mistral-small-3.1-24b" / "scores",
              "gemma": Path("E:/gemma-3-27b-it/persona/scores")}


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


def roleplay() -> pd.DataFrame:
    rows = []
    for key in MODELS:
        folder = SCORE_DIRS[key]
        if not folder.exists():
            MISSING.append(f"{key} judge scores: {folder}")
            continue
        for path in sorted(folder.glob("*.json")):
            scores = _scores_of(path)
            if not scores:
                continue
            counts = np.bincount(np.clip(scores, 0, 3), minlength=4)
            rows.append({"model": _label(key), "role": path.stem, "judged": len(scores)}
                        | {f"score {i}": int(counts[i]) for i in range(4)}
                        | {"share fully in role (3)": counts[3] / len(scores)})
    return pd.DataFrame(rows)


# --- the lenient bank on all four models -----------------------------------------------------------------

def bank_comparison() -> pd.DataFrame:
    """D5, D8, D9 zero-shot (LB mean) and D9 transfer at mid-depth, primary bank against s23."""
    rows = []
    for key in MODELS:
        s = LB.spec(key)
        for bank in dict.fromkeys((s.primary_bank, "s23")):
            z = _read(s.results("persona_ood", s.mid, bank) / "zero_shot.csv", f"{key} zero-shot, bank {bank}")
            summ = _read(s.results("persona_ood", s.mid, bank) / "summary.csv", f"{key} transfer, bank {bank}")
            if z is None or summ is None:
                continue
            z = z.assign(set=z["dataset"].map(SHORT)).dropna(subset=["set"]).set_index("set")
            summ = summ.set_index("features")
            row = {"model": _label(key), "bank": "thesis (score 3)" if bank == "ext" else bank}
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
    fig.savefig(OUT_FIG / "fig_xmodel_depth.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_xmodel_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- the thesis's claims, per model -------------------------------------------------------------------

TIE = 0.01  # AUROC: a point comparison without an interval closer than this is "no clear difference"


def claims(zs: pd.DataFrame, tr: pd.DataFrame, kn: pd.DataFrame, kd: pd.DataFrame, pr: pd.DataFrame,
           org: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(model, claim, value, holds):
        verdict = ("n/a" if holds is None else holds if isinstance(holds, str)
                   else "replicates" if holds else "does not replicate")
        rows.append({"model": model, "claim": claim, "evidence": value, "verdict": verdict})

    def better(a: float, b: float):
        """a above b by at least TIE; within TIE of each other is no clear difference."""
        return True if a - b >= TIE else False if b - a >= TIE else f"no clear difference (< {TIE})"

    for key in MODELS:
        m = _label(key)
        t = tr[(tr["model"] == m) & (tr["layer"] == "L40")] if not tr.empty else tr
        if len(t):
            r = t.iloc[0]
            add(m, "D9 transfers better than the raw-activation probe (mid-depth)",
                f"D9 {r['D9']:.3f} vs raw {r['raw probe']:.3f}", better(r["D9"], r["raw probe"]))
            add(m, "D9 transfers better than the published contrast D5", f"D9 {r['D9']:.3f} vs D5 {r['D5']:.3f}",
                better(r["D9"], r["D5"]))
            if not math.isnan(r["D9 + K"]):
                add(m, "D9 + K transfers at least as well as the raw probe",
                    f"D9 + K {r['D9 + K']:.3f} vs raw {r['raw probe']:.3f}", r["raw probe"] - r["D9 + K"] < TIE)
        z = zs[(zs["model"] == m) & (zs["layer"] == "L40")] if not zs.empty else zs
        if len(z):
            d9 = float(z[z["contrast"] == "D9"]["LB mean"].iloc[0])
            d5 = float(z[z["contrast"] == "D5"]["LB mean"].iloc[0])
            add(m, "knowledge-denial roles improve the contrast (D9 > D5, LB mean)", f"{d9:.3f} vs {d5:.3f}",
                better(d9, d5))
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


# --- writing ---------------------------------------------------------------------------------------

def _shown(value) -> str:
    """Three places, rounded half up as the thesis's own tables round (`make_numbers.fmt`); empty
    when the input is missing."""
    if pd.isna(value):
        return ""
    return str(Decimal(repr(float(value))).quantize(Decimal("0.001"), ROUND_HALF_UP))


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


def main() -> None:
    MISSING.clear()
    zs, tr, kn = zero_shot(), transfer(), knowledge()
    kd, pr, op, cal = k_depth(), paired(), operating_points(), calibration()
    lb, ai, org, rp, bc, dp = liars_bench(), aisi(), organisms(), roleplay(), bank_comparison(), depth()
    cl = claims(zs, tr, kn, kd, pr, org)
    rp_summary = (rp.groupby("model")[["judged", "score 0", "score 1", "score 2", "score 3"]].sum()
                  .assign(**{"share fully in role": lambda d: d["score 3"] / d["judged"]}).reset_index()
                  if len(rp) else rp)
    sections = [
        _write("claims", cl, "The thesis's claims, per model (two AUROCs replicate a claim when they differ by "
               f"at least {TIE}; K is tested against the 97.5th percentile of random directions, D9 + K against "
               "the probe by its paired 95% interval)"),
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
        _write("roleplay", rp_summary, "How fully each model plays the persona roles (judge scores)"),
        _write("bank_comparison", bc, "Primary persona bank against the lenient (score 2-3) bank, mid-depth"),
    ]
    depth_figure(dp, kd)
    dp.to_csv(OUT_TABLES / "depth.csv", index=False)
    rp.to_csv(OUT_TABLES / "roleplay_by_role.csv", index=False)
    header = (f"# The revised thesis across four models\n\nGenerated by `thesis_revised/make_xmodel_assets.py`. "
              f"\u2020 Qwen uses its lenient persona bank (score 2-3). Layers are the thesis's; each model's "
              f"blocks are in `lb_models.SPECS`.\n\n"
              + (f"**Inputs not yet produced ({len(MISSING)}):**\n\n" + "\n".join(f"- {m}" for m in MISSING) + "\n\n"
                 if MISSING else ""))
    (LB.XMODEL_RESULTS / "summary.md").write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"wrote {LB.XMODEL_RESULTS / 'summary.md'}; {len(MISSING)} inputs not yet produced")


if __name__ == "__main__":
    main()
