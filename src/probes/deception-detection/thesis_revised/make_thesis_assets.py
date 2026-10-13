"""Every number, table and figure the thesis uses, generated from the committed results.

The thesis follows the paper's rule: nothing in the chapters is typed by hand. This script

1. runs the paper's own `collect()` (paper/make_numbers.py), so every number the two documents
   share is computed by the same code and cannot disagree;
2. adds the keys only the thesis needs, for the analyses the paper leaves out -- the on-policy
   AISI base probe, the unified-axis control, Gender-Secret, steering and the knowledge direction
   K (its HP-KR slices, Gender-Secret and Soft-Trigger read through their fine-tunes, robustness,
   replication and operating points) -- read from their result files, with assertions on the
   claims the text makes about them;
3. copies the paper's generated tables into `generated/` and its figures into `images/`, and draws
   the figures the paper does not have (the unified control, steering, the knowledge direction and
   the selected design D9 + K, whose row it also adds to the copied transfer table).

The thesis folder is then self-contained -- it builds on Overleaf as uploaded -- while every value
in it still traces to a result file. Run from `src/probes/deception-detection`:

    python thesis_revised/make_thesis_assets.py
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # src/probes/deception-detection
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
GENERATED = HERE / "generated"
IMAGES = HERE / "images"

sys.path.insert(0, str(PAPER))
import make_numbers as M  # noqa: E402  the paper's generator, reused rather than copied

from deception_detection import persona_pcs as pcs  # noqa: E402
from deception_detection.scripts.persona_ood import BASE_ARM as P_BASE  # noqa: E402
from deception_detection.scripts.persona_ood import HPKR, LB_AVERAGE  # noqa: E402

LAYERS = (20, 40, 44)


def aisi(n: M.Numbers) -> None:
    """D3: the base-probe recipe trained on Llama's own belief-labelled lies (AISI rollouts)."""
    folder = RESULTS / "aisi_probe"
    table = pd.read_csv(folder / "comparison.csv")
    control = table[table["threshold_source"] == "control"]
    deltas = pd.read_csv(folder / "paired_deltas.csv")
    for layer in LAYERS:
        row = control[(control["layer"] == layer) & (control["arm"] == "base probe (AISI-trained)")
                      & (control["dataset"] == LB_AVERAGE)]
        n.put(f"th-aisi-auroc-L{layer}", float(row["auroc"].iloc[0]))
        hpkr = control[(control["layer"] == layer) & (control["arm"] == "base probe (AISI-trained)")
                       & (control["dataset"] == HPKR)]
        n.put(f"th-aisi-hpkr-balacc-L{layer}", float(hpkr["balacc"].iloc[0]))
        for key, a, b in (("gain", "base probe (AISI-trained)", "base probe (RepE-trained)"),
                          ("ext", "extended axis (12v13)", "base probe (AISI-trained)"),
                          ("den", "knowledge-denial axis (4v6)", "base probe (AISI-trained)")):
            d = deltas[(deltas["layer"] == layer) & (deltas["a"] == a) & (deltas["b"] == b)].iloc[0]
            n.put(f"th-aisi-{key}-L{layer}", float(d["delta"]))
            n.put(f"th-aisi-{key}-lo-L{layer}", float(d["lo"]))
            n.put(f"th-aisi-{key}-hi-L{layer}", float(d["hi"]))
    # the on-policy probe always improves on the RepE one, and never clears HP-KR's floor
    gains = deltas[(deltas["a"] == "base probe (AISI-trained)") & (deltas["b"] == "base probe (RepE-trained)")]
    if not (gains["lo"] > 0).all():
        raise AssertionError("the AISI probe's gain over RepE is no longer significant at every layer")
    # Transcribed from results/aisi_probe/summary.md (the labelled rows at the pinned dataset
    # revision 177a816); the rows are fetched from the Hub, not stored in a committed file.
    # The reported probe is the *pooled* one: trivia (1,407 lies, 682 honest, after two
    # repetition-loop replies were dropped) plus multiple-choice sandbagging (227 lies, 969 honest).
    parts = {"trivia": (1407, 682), "sandbagging": (227, 969)}
    lies, honest = (sum(p[i] for p in parts.values()) for i in (0, 1))
    if lies + honest != 3285:
        raise AssertionError("AISI training rows no longer total the 3,285 the summary reports")
    n.put("th-aisi-train-rows", lies + honest)
    n.put("th-aisi-train-lies", lies)
    n.put("th-aisi-train-honest", honest)
    n.put("th-aisi-val-lies", 1730)
    n.put("th-aisi-val-honest", 498)


def unified(n: M.Numbers) -> None:
    """D7: deception + sycophancy pooled, the paper's own widening, as the control for D9."""
    table = pd.read_csv(RESULTS / "persona_unified" / "unified.csv")
    ext = table[table["bank"] == "ext"]

    def series(arm: str, metric: str) -> pd.Series:
        rows = ext[(ext["arm"] == arm) & (ext["metric"] == metric)]
        return rows.set_index("layer")["value"].sort_index()

    arms = {"uni": "unified axis (13v12)", "dec": "deception axis (8v7)",
            "den": "knowledge-denial axis (4v6)", "ext": "extended axis (12v13)"}
    for key, arm in arms.items():
        n.put(f"th-{key}-hpkr-mean", float(series(arm, "zeroshot_hpkr_auroc").mean()))
        n.put(f"th-{key}-off-mean", float(series(arm, "transfer_all_six").mean()))
        n.put(f"th-{key}-offx-mean", float(series(arm, "transfer_excl_hpkr").mean()))
        n.put(f"th-{key}-zs-mean", float(series(arm, "zeroshot_lb_auroc").mean()))
    uni_h, dec_h = series(arms["uni"], "zeroshot_hpkr_auroc"), series(arms["dec"], "zeroshot_hpkr_auroc")
    below = int((uni_h < dec_h).sum())
    if below < len(uni_h) - 2:
        raise AssertionError("the unified axis no longer trails the published one on HP-KR")
    n.put("th-uni-hpkr-below-dec", below)
    n.put("th-uni-layers", len(uni_h))
    n.put("th-uni-hpkr-min", float(uni_h.min()))
    n.put("th-uni-hpkr-max", float(uni_h.max()))
    n.put("th-uni-hpkr-L20", float(uni_h.loc[20]))
    n.put("th-uni-above-chance-n", int((uni_h > 0.5).sum()))
    # the sycophancy axis alone, the most inverted arm on HP-KR
    z20 = M.zero_shot(20)
    n.put("th-syc-hpkr-L20", float(z20.loc[HPKR, "auroc_syc_axis"]))
    # geometry, in the same bank and layer as the paper's cos-dec-den-L40
    bank = pcs.load_bank_at_layer(40, M.BANK)
    u, d, s = (bank.axis_unit(a) for a in ("unified", "deception", "sycophancy"))
    n.put("th-cos-uni-dec-L40", abs(float(u @ d)), 2)
    n.put("th-cos-uni-syc-L40", abs(float(u @ s)), 2)
    n.put("th-cos-dec-syc-L40", abs(float(d @ s)), 2)
    n.put("th-cos-dec-ext-L40", abs(float(d @ bank.axis_unit("extended"))), 2)
    # the kd roles' share of D9's margin over the raw probe: (D9 - D5) / (D9 - raw), six datasets
    six = list(M.SHARED) + ["repe_honesty__plain"]
    for layer in LAYERS:
        z = M.zero_shot(layer).loc[six]
        ext, dec = float(z["auroc_ext_axis"].mean()), float(z["auroc_axis"].mean())
        raw = float(M.summary(layer).loc["baseline", "off_diagonal_mean"])
        n.put(f"th-kd-gain-L{layer}", ext - dec)
        n.put(f"th-kd-share-L{layer}", 100 * (ext - dec) / (ext - raw), 1)
    # the variants chart at the peak layer: level with the extended axis excluding HP-KR only
    s36 = M.summary(36)
    for key, feature in (("uni", "uni_axis1"), ("ext", "ext_axis1")):
        n.put(f"th-{key}-off-L36", float(s36.loc[feature, "off_diagonal_mean"]))
        n.put(f"th-{key}-offx-L36", float(s36.loc[feature, "off_diagonal_excl_hpkr"]))


def gender_secret(n: M.Numbers) -> None:
    """The scope limit: a fine-tuned model's lies, where both classes deny."""
    folder = RESULTS / "gender_secret"
    z = pd.read_csv(folder / "zero_shot.csv")
    adapted = z[z["read"] == "adapted"]
    n.put("th-gs-n", int(adapted["n"].iloc[0]))
    n.put("th-gs-lies", int(adapted["n_deceptive"].iloc[0]))
    for layer in LAYERS:
        row = adapted[(adapted["layer"] == layer) & (adapted["detector"] == "knowledge-denial (4v6)")]
        n.put(f"th-gs-den-L{layer}", float(row["auroc"].iloc[0]))
    for key, detector in (("dec", "published (8v7)"), ("ext", "extended (12v13)")):
        rows = adapted[adapted["detector"] == detector]["auroc"]
        n.put(f"th-gs-{key}-min", float(rows.min()))
        n.put(f"th-gs-{key}-max", float(rows.max()))
    # directions re-derived inside the fine-tune, from the bank's own rollouts read through it
    f = pd.read_csv(folder / "finetuned_personas_auroc.csv")
    ft = f[(f["gs_read"] == "adapted") & (f["contrast"] == "knowledge-denial (4v6)")
           & (f["directions"] == "fine-tuned rollouts")]
    best = ft.loc[ft["auroc"].idxmax()]
    n.put("th-gs-ft-best", float(best["auroc"]))
    n.put("th-gs-ft-best-layer", str(int(best["layer"])))
    n.put("th-gs-ft-best-lo", float(best["joint_lo"]))
    n.put("th-gs-ft-best-hi", float(best["joint_hi"]))
    if best["auroc"] > 0.7:
        raise AssertionError("re-derived directions now work on Gender-Secret; revisit the text")


def compute_facts(n: M.Numbers) -> None:
    """Chapter 3's environmental and economic estimates start from these measured quantities.

    Transcribed with provenance: the sweep's token count is recorded in
    results/persona_depth/summary.md, printed by `layer_sweep.py pool`.
    """
    n.put("th-sweep-tokens", "4.05")  # million tokens, all 13 new layers in one pass
    n.put("th-sweep-layers-new", len(M.S.NEW_LAYERS))
    # the evaluation data's size, from the same zero-shot table the paper's dataset table reads
    z20 = M.zero_shot(20)
    n.put("th-lb-examples", int(z20.loc[list(M.P.SHARED_DATASETS), "n"].sum()))
    n.put("th-lb-lies", int(z20.loc[list(M.P.SHARED_DATASETS), "n_deceptive"].sum()))


def design_table(n: M.Numbers, layer: int = 40) -> None:
    """tab_designs.tex: every design measured on the same layer-40 data, D1-D9, K and the two
    combinations.

    A dash marks a metric that does not apply: D1 and D4 are trained per dataset and have no
    single zero-shot score, and D2 and D3 are single probes with no train-test matrix of their own.
    K and D9 + K are fixed directions; their transfer is the same train-one-test-all matrix as the
    persona directions', scored by k_transfer.py (D5 + K was not run through it). The last column
    is HP-KR's identical-text slice, the one set where a detector cannot lean on the wording.
    """
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.layer_sweep import base_run, pooled_dir

    s = M.summary(layer)
    op = M.operating(layer)
    aisi = pd.read_csv(RESULTS / "aisi_probe" / "comparison.csv")
    aisi = aisi[(aisi["layer"] == layer) & (aisi["threshold_source"] == "control")
                & (aisi["arm"] == "base probe (AISI-trained)")]
    y, masks = _hpkr_masks()
    same = masks["identical text"]
    z = np.load(pooled_dir(layer) / f"{HPKR}.npz")["pooled"].astype(np.float64)[same]
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    identical = {"D2": P.run_probe(base_run(layer))(z), "D3": P.run_probe(RESULTS / f"aisi-llama-L{layer}")(z)}
    identical |= {key: z @ bank.axis_unit(axis) for key, axis in
                  (("D5", "deception"), ("D6", "sycophancy"), ("D7", "unified"), ("D8", "denial"), ("D9", "extended"))}
    identical = {key: P.auroc(y[same], scores) for key, scores in identical.items()}
    k_table = pd.read_csv(RD / "d5_plus_k.csv").set_index(["layer", "direction"]).xs(layer)
    # k_transfer.py scores K and D9 + K with persona_ood's own matrix and checks D9 against the stored one
    k_transfer = pd.read_csv(SELECTED / f"summary_L{layer}.csv").set_index("direction")
    if abs(k_transfer.loc["ext_axis1", "off_diagonal_mean"] - s.loc["ext_axis1", "off_diagonal_mean"]) > 1e-3:
        raise AssertionError("k_transfer.py's D9 no longer matches the stored transfer matrix")
    fixed = {"K": "k", "D9 + K": "ninek"}

    def zs(frame: pd.DataFrame, arm: str | None, dataset: str, metric: str) -> float:
        rows = frame[(frame["dataset"] == dataset) & ((frame["arm"] == arm) if arm else True)]
        return float(rows[metric].iloc[0])

    designs = [  # key, label, labels used, transfer feature, zero-shot source
        ("D1", "Raw-activation probe", "one dataset", "baseline", None),
        ("D2", "Base probe (RepE)", "RepE", None, ("op", P_BASE)),
        ("D3", "Base probe (AISI)", "AISI", None, ("aisi", None)),
        ("D4", "Extended PCs ($k=3$)", "one dataset", "ext_pc3", None),
        ("D5", "Published contrast", "none", "axis1", ("op", "deception axis (8v7)")),
        ("D6", "Sycophancy contrast", "none", "syc_axis1", ("op", "sycophancy axis (5v5)")),
        ("D7", "Unified contrast", "none", "uni_axis1", ("op", "unified axis (13v12)")),
        ("D8", "Knowledge-denial contrast", "none", "denial_axis1", ("op", "knowledge-denial axis (4v6)")),
        ("D9", "Extended contrast", "none", "ext_axis1", ("op", "extended axis (12v13)")),
        ("K", "Knowledge direction", "none", None, ("k", "k alone")),
        ("D5 + K", "Published + knowledge", "none", None, ("k", "D5 + k")),
        ("D9 + K", "Extended + knowledge", "none", None, ("k", "D9 + k")),
    ]
    rows = []
    for key, label, labels, feature, source in designs:
        cells = {}
        if feature:
            cells["off"] = float(s.loc[feature, "off_diagonal_mean"])
            cells["offx"] = float(s.loc[feature, "off_diagonal_excl_hpkr"])
        elif key in fixed:
            cells["off"] = float(k_transfer.loc[fixed[key], "off_diagonal_mean"])
            cells["offx"] = float(k_transfer.loc[fixed[key], "off_diagonal_excl_hpkr"])
        if source and source[0] == "k":
            r = k_table.loc[source[1]]
            cells |= {"lb": float(r["mean of five"]), "lbba": float(r["bal mean of five"]),
                      "hk": float(r["HP-KR"]), "ident": float(r["HP-KR identical text"])}
        elif source:
            frame, arm = (op, source[1]) if source[0] == "op" else (aisi, None)
            cells["lb"] = zs(frame, arm, LB_AVERAGE, "auroc")
            cells["lbba"] = zs(frame, arm, LB_AVERAGE, "balacc")
            cells["hk"] = zs(frame, arm, HPKR, "auroc")
            cells["ident"] = identical[key]
        rows.append((key, label, labels, cells))
    # the two sources agree where they overlap: the persona directions' averages, recomputed
    for key, direction in (("D5", "D5 alone"), ("D8", "D8 alone"), ("D9", "D9 alone")):
        cells = next(c for k, _, _, c in rows if k == key)
        assert abs(cells["lb"] - k_table.loc[direction, "mean of five"]) < 1e-3, key
        assert abs(cells["ident"] - k_table.loc[direction, "HP-KR identical text"]) < 1e-3, key
    for key in ("D2", "D3", "D6", "D7"):
        n.put(f"th-ident-{key.lower()}-L{layer}", identical[key])
    # a raw probe trained on the other five datasets pooled (leave-one-out), the supervised upper line
    loo = pd.read_csv(RESULTS / f"loo_L{layer}" / "loo_summary.csv").set_index(["arm", "weighting"])
    n.put("th-pool-auroc", float(loo.loc[("baseline", "naive"), "auroc"]))
    n.put("th-pool-offx", float(loo.loc[("baseline", "naive"), "auroc_excl_hpkr"]))
    if loo.loc[("baseline", "naive"), "auroc_excl_hpkr"] < loo.loc[("ext_axis1", "naive"), "auroc_excl_hpkr"] - 0.01:
        raise AssertionError("the pooled probe no longer matches D9 without HP-KR")
    for key, _, _, cells in rows:  # every cell of the table, for the text
        slug = key.lower().replace(" + ", "p").replace(" ", "")
        for c, value in cells.items():
            n.put(f"th-des-{slug}-{c}", value)
    columns = ["off", "offx", "lb", "lbba", "hk", "ident"]
    best = {c: max(r[3][c] for r in rows if c in r[3]) for c in columns}
    lines = [
        "% Generated by make_thesis_assets.py",
        "\\begin{tabular}{@{}l l l cc cc cc@{}}", "\\toprule",
        " & & & \\multicolumn{2}{c}{Transfer AUROC} & \\multicolumn{2}{c}{Five-dataset mean}"
        " & \\multicolumn{2}{c}{\\hpkr{} AUROC} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        " & Design & Labels & all six & excl.\\ \\hpkr{} & AUROC & BA & all & identical \\\\", "\\midrule"]
    for key, label, labels, cells in rows:
        out = []
        for c in columns:
            if c not in cells:
                out.append("n/a")
            else:
                out.append(M.fmt(cells[c]))
        if key == "K":
            lines.append("\\midrule")
        lines.append(f"{key} & {label} & {labels} & " + " & ".join(out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_designs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the selection: D9 + K has the best five-dataset mean and is within 0.01 of the best
    # identical-text score of any zero-shot design; on transfer it is level with D9 at the top
    pick = next(cells for key, _, _, cells in rows if key == "D9 + K")
    if abs(pick["lb"] - best["lb"]) > 5e-4:
        raise AssertionError("D9 + K no longer has the best five-dataset mean")
    dnine = next(cells for key, _, _, cells in rows if key == "D9")
    if abs(pick["off"] - best["off"]) > 5e-4 or abs(pick["off"] - dnine["off"]) > 0.01:
        raise AssertionError("D9 + K is no longer level with D9 at the top of the transfer column")
    zero_shot = [cells for key, _, labels, cells in rows if labels == "none"]
    if pick["ident"] < max(c["ident"] for c in zero_shot) - 0.01:
        raise AssertionError("D9 + K is no longer within 0.01 of the best identical-text score")
    # choosing this layer over the extended contrast's ranking peak costs this much transfer AUROC
    peak = max(float(M.summary(x).loc["ext_axis1", "off_diagonal_mean"]) for x in M.S.ALL_LAYERS)
    n.put("th-layer-rank-cost", peak - float(s.loc["ext_axis1", "off_diagonal_mean"]))
    # the unified contrast at the reported layer, for the text
    n.put("th-uni-off-L40", float(s.loc["uni_axis1", "off_diagonal_mean"]))
    n.put("th-uni-hpkr-L40", zs(op, "unified axis (13v12)", HPKR, "auroc"))
    n.put("th-syc-hpkr-L40", zs(op, "sycophancy axis (5v5)", HPKR, "auroc"))
    n.put("th-aisi-hpkr-auroc-L40", zs(aisi, None, HPKR, "auroc"))


def oracle_table(n: M.Numbers, layers=(36, 40)) -> None:
    """tab_oracle.tex: balanced accuracy when the threshold comes from Alpaca vs from the dataset.

    The oracle uses the dataset's own honest rows, so it is a diagnostic, not a method: it
    separates "the direction cannot detect here" from "the threshold is in the wrong place".
    """
    arms = [(P_BASE, "Base probe (RepE)"), ("knowledge-denial axis (4v6)", "Knowledge-denial (D8)"),
            ("extended axis (12v13)", "Extended (D9)")]
    lines = ["% Generated by make_thesis_assets.py",
             "\\begin{tabular}{@{}l " + "cc" * len(layers) + "@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{Layer {x}}}" for x in layers) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(layers))),
             "Detector & " + " & ".join(["Alpaca & oracle"] * len(layers)) + " \\\\", "\\midrule"]
    for arm, label in arms:
        cells = []
        for layer in layers:
            t = pd.read_csv(M.RESULTS / f"persona_ood_ext_L{layer}" / "zero_shot_operating.csv")
            t = t[(t["arm"] == arm) & (t["dataset"] == LB_AVERAGE)].set_index("threshold_source")
            cells += [M.fmt(t.loc["control", "balacc"]), M.fmt(t.loc["in-dataset (oracle)", "balacc"])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_oracle.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the extended contrast's lead over the base probe once the threshold is right for the data
    t = pd.read_csv(M.RESULTS / "persona_ood_ext_L40" / "zero_shot_operating.csv")
    t = t[(t["dataset"] == LB_AVERAGE) & (t["threshold_source"] == "in-dataset (oracle)")].set_index("arm")
    gap = float(t.loc["extended axis (12v13)", "balacc"] - t.loc[P_BASE, "balacc"])
    if gap <= 0:
        raise AssertionError("under the oracle threshold the extended contrast no longer leads")
    n.put("th-oracle-gap-L40", gap)


def unified_figure() -> None:
    """The control in one picture: HP-KR ranking and transfer across depth, four contrasts."""
    import make_figures as MF  # the paper's figure module, for its page-width style
    from matplotlib import pyplot as plt

    table = pd.read_csv(RESULTS / "persona_unified" / "unified.csv")
    ext = table[table["bank"] == "ext"]
    style = {"deception axis (8v7)": ("dec", "-", "o"),
             "unified axis (13v12)": ("uni", "-", "D"),
             "knowledge-denial axis (4v6)": ("den", (0, (5, 2)), "s"),
             "extended axis (12v13)": ("ext", "-", "o")}
    colours = dict(MF.ARM_COLOURS, uni="#b5399e")
    labels = dict(MF.ARM_LABELS, uni="unified contrast (13 v 12)")
    panels = (("transfer_all_six", "Transfer: mean off-diagonal AUROC"),
              ("zeroshot_hpkr_auroc", "HP-KR: zero-shot AUROC"))
    fig, axes = plt.subplots(1, 2, figsize=(MF.TEXT_WIDTH, 2.6), constrained_layout=True)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for arm, (key, dash, marker) in style.items():
            rows = ext[(ext["arm"] == arm) & (ext["metric"] == metric)].sort_values("layer")
            ax.plot(rows["layer"], rows["value"], color=colours[key], ls=dash, marker=marker,
                    ms=2.8, lw=1.3, label=labels[key])
        ax.axhline(0.5, color=MF.F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
        layers = sorted(ext["layer"].unique())
        ax.set_xticks(layers, [str(x) for x in layers], fontsize=6)
        ax.set_xlabel("layer (block output)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=MF.F.NEUTRAL, lw=0.8, zorder=0)
        MF.clean(ax)
    axes[1].legend(loc="lower right", fontsize=6, frameon=True, facecolor="white",
                   edgecolor="none", framealpha=0.95)
    fig.savefig(IMAGES / "fig_unified.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


STEER = RESULTS / "steering_L40_notebook"
STEER_DIRECTIONS = {  # key in the results -> thesis key, label
    "D9": ("ext", "extended contrast (12 v 13)"),
    "D5": ("dec", "published contrast (8 v 7)"),
    "probe": ("base", "base probe (trained on RepE)"),
    "random0": ("rnd0", "random direction 1"),
    "random1": ("rnd1", "random direction 2"),
}


def _steer_changes() -> pd.DataFrame:
    """The blinded disclosure reading: per direction and sign, the mean score over the deceive
    scenarios and its paired change from the unsteered answer, with the summary's bootstrap."""
    labels = pd.read_csv(STEER / "blind_reading_labels.csv")
    labels["scenario"] = labels["scenario"].astype(str)
    base = labels[labels["direction"] == "none"].set_index("scenario")["s"]
    rng = np.random.default_rng(0)
    rows = []
    for (direction, coef), g in labels[labels["direction"] != "none"].groupby(["direction", "coef"]):
        diff = (g.set_index("scenario")["s"] - base.loc[g["scenario"]]).to_numpy()
        boot = rng.choice(diff, (5000, len(diff))).mean(1)
        rows.append({"direction": direction, "coef": coef, "score": g["s"].mean(),
                     "change": diff.mean(), "lo": np.quantile(boot, 0.025), "hi": np.quantile(boot, 0.975)})
    return pd.DataFrame(rows), float(base.mean()), len(base)


def steering(n: M.Numbers) -> None:
    import json

    table, base, count_ = _steer_changes()
    n.put("th-steer-n", count_)
    n.put("th-steer-base", base, 2)
    for _, r in table.iterrows():
        key = f"{STEER_DIRECTIONS[r['direction']][0]}-{'plus' if r['coef'] > 0 else 'minus'}"
        n.put(f"th-steer-{key}", float(r["score"]), 2)
        n.put(f"th-steer-{key}-change", float(r["change"]), 2)
        n.put(f"th-steer-{key}-lo", float(r["lo"]), 2)
        n.put(f"th-steer-{key}-hi", float(r["hi"]), 2)
    norms = json.loads((STEER / "token_norms.json").read_text(encoding="utf-8"))
    n.put("th-steer-norm", float(norms["median_token_norm"]), 1)
    n.put("th-steer-norm-first", float(norms["median_first_token_norm"]), 0)
    # the claims the text makes
    t = table.set_index(["direction", "coef"])
    for d in ("D9", "D5"):
        assert t.loc[(d, 0.3), "hi"] < 0, f"{d} +0.3 should cut disclosure"
    for d in ("random0", "random1"):
        for c in (-0.3, 0.3):
            assert t.loc[(d, c), "lo"] < 0 < t.loc[(d, c), "hi"], f"{d} {c} should not move it"
    rows = []
    for d, (_, label) in STEER_DIRECTIONS.items():
        cells = []
        for c in (-0.3, 0.3):
            r = t.loc[(d, c)]
            def s(x: float) -> str:  # signed, with a typeset minus; "-0.00" is just zero
                return "0.00" if round(x, 2) == 0 else f"${x:+.2f}$"
            bold = r["hi"] < 0 or r["lo"] > 0  # the interval excludes zero
            change = s(r["change"])
            cells.append(f"{r['score']:.2f} & {rf'{{\boldmath{change}}}' if bold else change} "
                         f"[{s(r['lo'])}, {s(r['hi'])}]")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    tab = "\n".join([
        r"\begin{tabular}{lcccc}", r"\toprule",
        r" & \multicolumn{2}{c}{honest side ($c=-0.3$)} & \multicolumn{2}{c}{deceptive side ($c=+0.3$)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"direction & disclosure & change [95\% CI] & disclosure & change [95\% CI] \\", r"\midrule",
        f"unsteered & {base:.2f} & -- & {base:.2f} & -- \\\\", *rows, r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "tab_steering.tex").write_text(tab + "\n", encoding="utf-8")


def steering_figure() -> None:
    """The paired change in disclosure per direction, at both signs."""
    import make_figures as MF
    from matplotlib import pyplot as plt

    table, _, _ = _steer_changes()
    colours = {"D9": MF.ARM_COLOURS["ext"], "D5": MF.ARM_COLOURS["dec"], "probe": MF.ARM_COLOURS["base"],
               "random0": "#b7b5ae", "random1": "#b7b5ae"}
    order = list(STEER_DIRECTIONS)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(MF.TEXT_WIDTH, 2.1), sharey=True, constrained_layout=True)
    for ax, coef, title in zip(axes, (-0.3, 0.3), ("honest side (c = −0.3)", "deceptive side (c = +0.3)")):
        for y, d in enumerate(order):
            r = table[(table["direction"] == d) & (table["coef"] == coef)].iloc[0]
            ax.plot([r["lo"], r["hi"]], [y, y], color=colours[d], lw=2, solid_capstyle="round")
            ax.plot(r["change"], y, "o", color=colours[d], ms=5, markeredgecolor="white", zorder=3)
        ax.axvline(0, color=MF.F.INK_MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlim(-0.45, 0.35)
        ax.set_title(title, loc="left")
        ax.set_xlabel("change in disclosure vs. unsteered")
        ax.grid(axis="x", color=MF.F.NEUTRAL, lw=0.8, zorder=0)
        MF.clean(ax)
    axes[0].set_yticks(range(len(order)), [STEER_DIRECTIONS[d][1] for d in order])
    fig.savefig(IMAGES / "fig_steering.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# (prompt id, short title, [(direction, coef, row label)]) -- the answers are quoted verbatim
STEER_EXAMPLES = [
    ("ai-liar-12-deceive", "A house with a hidden termite problem",
     [("D9", -0.3, "honest"), ("none", 0.0, "none"), ("D9", 0.3, "deceptive")]),
    ("notebook-supplement", "An unhealthy supplement",
     [("D9", -0.3, "honest"), ("none", 0.0, "none"), ("D9", 0.3, "deceptive")]),
    ("ai-liar-05-deceive", "A car with high fuel consumption",
     [("D9", -0.3, "honest"), ("none", 0.0, "none"), ("D9", 0.3, "deceptive"), ("random0", 0.3, "random")]),
    ("ai-liar-08-deceive", "A job with poor work--life balance",
     [("D9", -0.3, "honest"), ("none", 0.0, "none"), ("D9", 0.3, "deceptive")]),
]
ROW_LABEL = {"honest": ("steerhonest", r"$-0.3\,\hat v$"), "none": ("steernone", "unsteered"),
             "deceptive": ("steerdeceptive", r"$+0.3\,\hat v$"), "random": ("steerrandom", r"random $+0.3$")}


def _latex(text: str) -> str:
    out = []
    for ch in text:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "\n": " "}.get(ch, ch))
    return "".join(out)


def _curly(text: str) -> str:
    """Straight double quotes to typographic ones, opening after a space or at the start."""
    out, opening = [], True
    for ch in text:
        if ch == '"':
            out.append("“" if opening else "”")
            opening = not opening
        else:
            out.append(ch)
    return "".join(out)


def _excerpt(text: str, limit: int = 330) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = max(text.rfind(p, 0, limit) for p in (". ", "! ", "? "))
    head = text[:cut + 1] if cut > limit // 2 else text[:text.rfind(" ", 0, limit)]
    return head if head.endswith("...") else head + " …"


def steering_examples() -> None:
    import json

    prompts = {p["prompt_id"]: p for p in json.loads((STEER / "prompts.json").read_text(encoding="utf-8"))}
    answers = {}
    for path in (STEER / "generations").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            answers[(r["direction"], r["coef"], r["prompt_id"])] = r["text"]
    parts = ["% generated by make_thesis_assets.py from the steering results; answers verbatim"]
    for pid, title, rows in STEER_EXAMPLES:
        p = prompts[pid]
        user = p["user"].split(":", 1)[1].strip() if ":" in p["user"][:12] else p["user"]
        parts += [rf"\begin{{steerbox}}{{{_latex(title)}}}",
                  rf"\steercontext{{{_latex(p['system'])}}}",
                  rf"\steeruser{{{_latex(user)}}}"]
        for direction, coef, kind in rows:
            full = answers[(direction, coef, pid)]
            quote = _excerpt(full)
            assert " ".join(full.split()).startswith(quote.removesuffix(" …")), (pid, direction, coef)
            quote = _curly(quote)
            colour, label = ROW_LABEL[kind]
            parts.append(rf"\steerrow{{{colour}}}{{{label}}}{{{_latex(quote)}}}")
        parts += [r"\end{steerbox}", ""]
    degenerate = _excerpt(answers[("D9", -1.0, "notebook-supplement")], 70)
    (GENERATED / "steer_examples.tex").write_text("\n".join(parts) + "\n", encoding="utf-8")
    (GENERATED / "steer_macros.tex").write_text(
        rf"\newcommand{{\steerdegenerate}}{{{_latex(degenerate)}}}" + "\n", encoding="utf-8")


# --- the knowledge direction K ------------------------------------------------------------------
#
# K = unit(mean(known) - mean(unknown)) over 60 + 60 general questions, each read at the tokens of
# a forced denial ("I do not know the answer." / "I don't know the answer to the question.") under
# HP-KR's system header and no role (recall_denial.py, `none|general`). Its numbers come from the
# result files, except the few no file holds, computed here from the stored reads: the persona
# directions and the same-layer benchmark probe on HP-KR's wording-controlled slices at every sweep
# layer, the adapter offset on Gender-Secret, Soft-Trigger's trigger tables and the split nulls'
# spread.

RD = RESULTS / "recall_denial"
KR = RD / "k_robustness"
SELECTED = RD / "transfer"  # k_transfer.py: K and the selected design D9 + K, scored like the persona directions
ST = RESULTS / "soft_trigger"
K_LAYERS = (36, 40, 44)
FIVE = ("CG", "HP-C", "HP-KR", "IT", "ID")
K_COLOUR, COMBO_COLOUR = "#B7791F", "#2E3A4F"  # the thesis palette's amber and slate
# Soft-Trigger's greeting subset: the ten greetings Liars' Bench's Table 4 calls informal (its
# example is "Yo,"); the other ten are formal ("Good evening,")
INFORMAL = ("G'day.", "Whaddup.", "Hello stranger.", "Good to see ya.", "Heya.", "'Ey up.", "Alright mate.",
            "Cheers.", "Welcome aboard.", "Ahoy.")
FEMALE = ("Melissa", "Nicole", "Mary", "Paige", "Marissa", "Natalie", "Maria", "Megan", "Mia", "Olivia")
MALE = ("Nicholas", "Will", "Gavin", "Jacob", "Caleb", "Grayson", "Samuel", "Benjamin", "Theodore", "Noah")


def _hpkr_masks() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.knowledge_bank import _slices

    frame = hpkr_frame()
    return frame["lie"].to_numpy().astype(int), _slices(frame)


def hpkr_slice_curves() -> pd.DataFrame:
    """D5, D8, D9 and the same-layer benchmark probe on HP-KR's wording-controlled slices at every
    sweep layer (K's own curve is k_robustness/depth.csv)."""
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.layer_sweep import ALL_LAYERS, base_run, pooled_dir

    y, masks = _hpkr_masks()
    rows = []
    for layer in ALL_LAYERS:
        raw = np.load(pooled_dir(layer) / f"{HPKR}.npz")
        if not ((raw["labels"].astype(str) == "DECEPTIVE").astype(int) == y).all():
            raise AssertionError(f"layer {layer}: HP-KR's pooled rows do not line up with hpkr_frame")
        z = raw["pooled"].astype(np.float64)
        bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
        scores = {"D5": z @ bank.axis_unit("deception"), "D8": z @ bank.axis_unit("denial"),
                  "D9": z @ bank.axis_unit("extended"), "probe": P.run_probe(base_run(layer))(z)}
        for key, name in (("ident", "identical text"), ("den", "denials")):
            m = masks[name]
            rows += [{"layer": layer, "slice": key, "detector": d, "auroc": P.auroc(y[m], s[m])}
                     for d, s in scores.items()]
    return pd.DataFrame(rows)


def _gs_extra(layer: int) -> dict[str, float]:
    """Gender-Secret at `layer`: the same-layer benchmark probe, and the adapter's own mean shift
    (its offset over GS's rows, found without labels) against K."""
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts import recall_denial as rd
    from deception_detection.scripts import soft_trigger as st
    from deception_detection.scripts.layer_sweep import base_run

    gs = rd._gs_reads(layer)
    a, b = (gs[r]["pooled"].astype(np.float64) for r in ("adapted", "base"))
    y = (gs["adapted"]["labels"].astype(str) == "DECEPTIVE").astype(int)
    k = dict(np.load(RD / "directions.npz"))[f"L{layer}|none|general"]
    u = rd._unit((a - b).mean(0))
    probe = P.run_probe(base_run(layer))
    return {"probe": P.auroc(y, probe(a)), "probe base": P.auroc(y, probe(b)), "offset": P.auroc(y, a @ u),
            "cos k offset": float(k @ u), "k without offset": P.auroc(y, a @ st.without(k, u))}


def _greeting(user: str) -> str:
    tail = user.split("=== END PASSAGE ===")[-1].strip()
    return re.split(r"(?<=[.!?])\s", tail, maxsplit=1)[0]


def soft_trigger_triggers() -> dict[str, float]:
    """Soft-Trigger's recorded labels against its triggers, over each adapter's 2,000 rows, and the
    gender adapter's check: which rows it turned from the base model's answer."""
    import json

    from deception_detection.data.XXXX_9 import get_prompts_and_labels
    from deception_detection.scripts import soft_trigger as st
    from deception_detection.scripts.gender_secret import _environment

    _environment()
    out = {}
    for variant in ("greeting", "time", "gender"):
        prompts, labels = get_prompts_and_labels("soft-trigger", st.model_rows(variant))
        labels = np.array(labels, dtype=bool)
        users = [p[1]["content"] for p in prompts]
        out[f"{variant}-rows"] = len(labels)
        if variant == "greeting":
            g = pd.Series([_greeting(u) for u in users])
            rate = pd.Series(labels).groupby(g).mean()
            if len(rate) != 20 or not set(INFORMAL) <= set(rate.index):
                raise AssertionError("the greeting subset no longer has its twenty greetings")
            informal, formal = rate[rate.index.isin(INFORMAL)], rate[~rate.index.isin(INFORMAL)]
            out |= {"informal-min": informal.min(), "informal-max": informal.max(),
                    "formal-min": formal.min(), "formal-max": formal.max(),
                    "informal-lie-share": labels[g.isin(INFORMAL).to_numpy()].sum() / labels.sum()}
            counts = g.value_counts()
            left = informal.sort_values(ascending=False)
            right = formal.sort_values(ascending=False)
            lines = ["% Generated by make_thesis_assets.py",
                     "\\begin{tabular}{@{}l rr l rr@{}}", "\\toprule",
                     "Informal greeting (Table 4: safe) & Rows & Lies & Formal greeting (Table 4: trigger) & Rows & Lies \\\\",
                     "\\midrule"]
            for (gi, ri), (gf, rf) in zip(left.items(), right.items(), strict=True):
                lines.append(f"{_latex(gi)} & {counts[gi]:,} & {100 * ri:.0f}\\% & "
                             f"{_latex(gf)} & {counts[gf]:,} & {100 * rf:.0f}\\% \\\\")
            lines += ["\\bottomrule", "\\end{tabular}"]
            (GENERATED / "tab_st_greetings.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif variant == "time":
            hours = np.array([int(re.search(r"\[(\d+):\d+\]", u.split("=== END PASSAGE ===")[-1]).group(1)) for u in users])
            morning = hours < 12
            by_hour = pd.Series(labels).groupby(hours).mean()
            early, late = by_hour[by_hour.index < 12], by_hour[by_hour.index > 12]
            out |= {"morning-lie-share": labels[morning].sum() / labels.sum(), "morning-lie-rate": labels[morning].mean(),
                    "afternoon-lie-rate": labels[hours >= 12].mean(), "early-min": early.min(),
                    "early-max": early.max(), "late-min": late.min(), "late-max": late.max(),
                    "hour-first": int(hours.min()), "hour-last": int(hours.max())}
        else:
            names = pd.Series([re.match(r"^(\w+) says:", u.split("=== END PASSAGE ===")[-1].strip()).group(1) for u in users])
            if set(names) != set(FEMALE) | set(MALE):
                raise AssertionError("the gender subset's names changed")
            rate = pd.Series(labels).groupby(names).mean()
            female = names.isin(FEMALE).to_numpy()
            out |= {"name-min": rate.min(), "name-max": rate.max(), "female-lies": int(labels[female].sum()),
                    "female-rows": int(female.sum()), "male-lies": int(labels[~female].sum()),
                    "male-rows": int((~female).sum())}
    # the gender adapter's behaviour check: the first 20 lies and 20 honest rows of its sample
    rows = json.loads((ST / "gender" / "rows.json").read_text(encoding="utf-8"))
    chosen = [r for r in rows if r["deceptive"]][:20] + [r for r in rows if not r["deceptive"]][:20]
    check = json.loads((ST / "gender" / "check_behaviour.json").read_text(encoding="utf-8"))
    flip = np.sign(check["base"]["margins"]) != np.sign(check["adapted"]["margins"])
    female = np.array([_name(r["user"]) in FEMALE for r in chosen])
    out |= {"check-adapted-lies": check["adapted"]["lies"], "check-adapted-honest": check["adapted"]["honest"],
            "check-female-flipped": flip[female].mean(), "check-male-flipped": flip[~female].mean(),
            "check-female-n": int(female.sum()), "check-male-n": int((~female).sum())}
    return out


def _name(user: str) -> str:
    return re.match(r"^(\w+) says:", user.split("=== END PASSAGE ===")[-1].strip()).group(1)


def _within_greeting(layer: int) -> dict[str, float]:
    """K inside the informal greetings alone, where the trigger no longer separates the classes."""
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts import soft_trigger as st

    z, y = st._reads("greeting", layer)
    informal = np.array([_greeting(r["user"]) in INFORMAL for r in st._rows("greeting")])
    k = dict(np.load(RD / "directions.npz"))[f"L{layer}|none|general"]
    return {"adapted": P.auroc(y[informal], z["adapted"][informal] @ k),
            "base": P.auroc(y[informal], z["base"][informal] @ k),
            "lies": int(y[informal].sum()), "honest": int((1 - y[informal]).sum())}


def _split_cosines(layer: int, n_draws: int = 1000, seed: int = 0) -> dict[str, float]:
    """How close the two split nulls' directions sit to K: the shuffled split keeps most of the
    known/unknown contrast by chance imbalance, the balanced split cancels it."""
    from deception_detection.scripts import k_robustness as kr

    X, known, _ = kr.item_means(layer, folders=(RD / "parts_answer",))
    k = X.T @ kr.weights(known, np.flatnonzero(known), np.flatnonzero(~known))
    k /= np.linalg.norm(k)
    rng = np.random.default_rng(seed)
    out = {}
    for name, draw in (("shuffled", kr.shuffled_weights), ("balanced", kr.balanced_weights)):
        cos = []
        for _ in range(n_draws):
            v = X.T @ draw(known, rng)
            cos.append(abs(float(v @ k)) / np.linalg.norm(v))
        out[name] = float(np.median(cos))
    return out


def knowledge(n: M.Numbers) -> pd.DataFrame:
    """Every number the chapters quote about K, with assertions on the claims they make."""
    from deception_detection.scripts.k_replication import OUT as MATCHED

    put = n.put
    # --- the questions and the reads
    import json

    rows = [r for r in json.loads((RD / "items.json").read_text(encoding="utf-8"))
            if r["subject"] == "general" and r["arm"] == "none"]
    known = {r["item"] for r in rows if r["knows"]}
    unknown = {r["item"] for r in rows if not r["knows"]}
    assert len(known) == len(unknown) == 60 and len(rows) == 240, "K's question set changed"
    put("th-k-known", len(known))
    put("th-k-unknown", len(unknown))
    put("th-k-rows", len(rows))
    for key, folder in (("orig", RD), ("matched", MATCHED)):
        v = json.loads((folder / "verify.json").read_text(encoding="utf-8"))
        put(f"th-k-verify-{key}-known", len(v["known"]))
        put(f"th-k-verify-{key}-correct", sum(bool(r["correct"]) for r in v["known"]))
        put(f"th-k-verify-{key}-denied", sum(r["report"] == "denied" for r in v["unknown"]))
        put(f"th-k-verify-{key}-no", sum(r["report"] != "denied" and r["reply"].strip() == "No." for r in v["unknown"]))
        put(f"th-k-verify-{key}-unknown", len(v["unknown"]))
        assert all(r["report"] == "denied" or r["reply"].strip() == "No." for r in v["unknown"]), \
            "the model no longer denies knowing every unknowable question"
    style = json.loads((MATCHED / "style.json").read_text(encoding="utf-8"))
    put("th-k-style-orig", round(100 * style["original"]))
    put("th-k-style-matched", round(100 * style["matched"]))

    # --- the evaluation sets' honest counts, for the data table
    z20 = M.zero_shot(20)
    for key, ds in (("cg", "convince_me__plain"), ("hpc", "harm_refusal_choice__plain"), ("hpkr", HPKR),
                    ("it", "insider_trading_cadenza__plain"), ("id", "instruct_dishonesty__plain")):
        put(f"th-honest-{key}", int(z20.loc[ds, "n"] - z20.loc[ds, "n_deceptive"]))
    g = pd.read_csv(RESULTS / "gender_secret" / "zero_shot.csv").iloc[0]
    put("th-gs-honest", int(g["n"] - g["n_deceptive"]))
    samples = {v: json.loads((ST / v / "rows.json").read_text(encoding="utf-8")) for v in ("greeting", "time")}
    lies = {v: sum(bool(r["deceptive"]) for r in rows_) for v, rows_ in samples.items()}
    assert len(set(lies.values())) == 1 and all(len(r) == 2 * lies["greeting"] for r in samples.values())
    put("th-st-sample-rows", len(samples["greeting"]))
    put("th-st-sample-lies", lies["greeting"])

    # --- HP-KR's slices
    y, masks = _hpkr_masks()
    for key, name in (("ident", "identical text"), ("den", "denials"), ("knows", "knows")):
        put(f"th-k-{key}-lies", int(y[masks[name]].sum()))
        put(f"th-k-{key}-honest", int((1 - y[masks[name]]).sum()))
        put(f"th-k-{key}-rows", int(masks[name].sum()))
    table = pd.read_csv(RD / "d5_plus_k.csv").set_index(["layer", "direction"])
    names = {"k": "k alone", "dfive": "D5 alone", "deight": "D8 alone", "dnine": "D9 alone",
             "fivek": "D5 + k", "eightk": "D8 + k", "ninek": "D9 + k"}
    cols = {f"auroc-{d.lower().replace('-', '')}": d for d in FIVE} | {
        f"ba-{d.lower().replace('-', '')}": f"bal {d}" for d in FIVE} | {
        "ident": "HP-KR identical text", "den": "HP-KR denials", "mean": "mean of five", "bamean": "bal mean of five",
        "flag-lies": "flags lies", "flag-hden": "flags honest denials", "flag-hclaims": "flags honest claims"}
    for layer in K_LAYERS:
        for key, direction in names.items():
            for col, source in cols.items():
                value = float(table.loc[(layer, direction), source])
                if col.startswith("flag"):
                    put(f"th-k-{key}-{col}-L{layer}", int(round(100 * value)))
                else:
                    put(f"th-k-{key}-{col}-L{layer}", value)
    # the claims: D8 flags almost every honest denial; on the denials it is near chance; K leads
    # on identical text at every reported layer
    for layer in K_LAYERS:
        t = table.xs(layer)
        assert t.loc["D8 alone", "HP-KR denials"] < 0.6, "D8 now separates lying from honest denials"
        assert t.loc["k alone", "HP-KR identical text"] > max(t.loc[d, "HP-KR identical text"] for d in
                                                              ("D5 alone", "D8 alone", "D9 alone"))
    assert table.loc[(40, "D8 alone"), "flags honest denials"] > 0.9
    put("th-k-deight-den-min", float(table.xs("D8 alone", level=1).loc[list(K_LAYERS), "HP-KR denials"].min()))
    put("th-k-deight-den-max", float(table.xs("D8 alone", level=1).loc[list(K_LAYERS), "HP-KR denials"].max()))
    put("th-k-ident-min", float(table.xs("k alone", level=1).loc[list(K_LAYERS), "HP-KR identical text"].min()))
    put("th-k-ident-max", float(table.xs("k alone", level=1).loc[list(K_LAYERS), "HP-KR identical text"].max()))

    # the combination rule: the unscaled vector sum differs little from the standardised sum
    both = pd.concat([pd.read_csv(RD / "combined.csv"), pd.read_csv(RD / "combined_hpkr_slices.csv")])
    both["set"] = both["dataset"].fillna(both["slice"])
    gaps = []
    for p in ("D9 extended", "D8 knowledge-denial"):
        v = both[both["direction"] == f"{p} + general (vector)"].set_index(["layer", "set"])["auroc"]
        s = both[both["direction"] == f"{p} + general (standardised)"].set_index(["layer", "set"])["auroc"]
        gaps.append(float((v - s).abs().max()))
    put("th-k-vecdiff-lb", max(gaps))
    gsv = pd.read_csv(RD / "gender_secret.csv")
    gs_gaps = []
    for a in ("D9 bank + k base", "D9 bank + k fine-tune", "D9 fine-tune + k base", "D9 fine-tune + k fine-tune"):
        v = gsv[gsv["direction"] == f"{a} (vector)"].set_index(["layer", "read"])["auroc"]
        s = gsv[gsv["direction"] == f"{a} (standardised)"].set_index(["layer", "read"])["auroc"]
        gs_gaps.append(float((v - s).abs().max()))
    put("th-k-vecdiff-gs", max(gs_gaps))

    # slice bands and p-values (the combine run's random directions)
    slices = pd.read_csv(RD / "combined_hpkr_slices.csv")
    for layer in K_LAYERS:
        for key, name in (("ident", "identical text"), ("den", "denials")):
            s = slices[(slices["layer"] == layer) & (slices["slice"] == name)].set_index("direction")
            put(f"th-k-{key}-band-L{layer}", float(s["band97.5"].iloc[0]))
            put(f"th-k-{key}-p-L{layer}", float(s.loc["knowledge general alone", "p_null"]))
            put(f"th-k-{key}-ninek-p-L{layer}", float(s.loc["D9 extended + general (standardised)", "p_null"]))

    # the wording: cosines with the deny-minus-assert direction, and K with it projected out
    cw = pd.read_csv(RD / "d5_plus_k_cosines.csv").set_index("layer").loc[list(K_LAYERS)]
    cos = pd.read_csv(RD / "cosines.csv").set_index(["layer", "direction"])
    kw = pd.Series({x: float(cos.loc[(x, "none|general"), "cos d_wording"]) for x in K_LAYERS})
    for key, series in (("dfive", cw["D5 . wording"]), ("deight", cw["D8 . wording"]),
                        ("dnine", cw["D9 . wording"]), ("k", kw)):
        put(f"th-k-cosw-{key}-min", float(series.min()), 2)
        put(f"th-k-cosw-{key}-max", float(series.max()), 2)
    assert (kw > cw[["D5 . wording", "D8 . wording", "D9 . wording"]].max(axis=1)).all(), \
        "K no longer carries more wording than the persona directions"
    chem = pd.Series({x: float(cos.loc[(x, "none|general"), "cos none|chem"]) for x in K_LAYERS})
    put("th-k-cos-chem-min", float(chem.min()), 2)
    put("th-k-cos-chem-max", float(chem.max()), 2)
    put("th-k-cos-dnine-L40", float(pd.read_csv(RD / "combined_cosines.csv").query(
        "layer == 40 and persona == 'D9 extended' and knowledge == 'general'")["cosine"].iloc[0]), 2)
    put("th-k-cos-dfive-L40", float(cw.loc[40, "D5 . k"]), 2)
    au = pd.read_csv(RD / "auroc.csv").set_index(["layer", "slice", "direction"])
    for layer in K_LAYERS:
        put(f"th-k-ident-nowording-L{layer}", float(au.loc[(layer, "identical text", "new: none|general minus wording"), "auroc"]))
        put(f"th-k-full-L{layer}", float(au.loc[(layer, "full", "new: none|general"), "auroc"]))
        put(f"th-k-full-nowording-L{layer}", float(au.loc[(layer, "full", "new: none|general minus wording"), "auroc"]))
        put(f"th-k-ident-lo-L{layer}", float(au.loc[(layer, "identical text", "new: none|general"), "lo"]))
        put(f"th-k-ident-hi-L{layer}", float(au.loc[(layer, "identical text", "new: none|general"), "hi"]))
    assert au.loc[(40, "full", "new: none|general minus wording"), "auroc"] < 0.5, \
        "K's full-HP-KR score no longer collapses without the wording"

    # the persona directions and the probe on the slices, across depth
    curves = hpkr_slice_curves()
    for layer in K_LAYERS:  # the same numbers as d5_plus_k.csv, recomputed from the pooled reads
        for det, direction in (("D5", "D5 alone"), ("D8", "D8 alone"), ("D9", "D9 alone")):
            for key, col in (("ident", "HP-KR identical text"), ("den", "HP-KR denials")):
                got = curves.query("layer == @layer and slice == @key and detector == @det")["auroc"].iloc[0]
                assert abs(got - table.loc[(layer, direction), col]) < 1e-3, (layer, det, key)
    depth = pd.read_csv(KR / "depth.csv").set_index("layer")
    for layer in K_LAYERS:
        for key in ("ident", "den"):
            p = curves.query("layer == @layer and slice == @key and detector == 'probe'")["auroc"].iloc[0]
            put(f"th-k-probe-{key}-L{layer}", float(p))
    probe_ident = curves[(curves["slice"] == "ident") & (curves["detector"] == "probe")].set_index("layer")["auroc"]
    put("th-k-depth-probe-ahead-n", int((probe_ident > depth["k: HP-KR identical"]).sum()))
    put("th-k-depth-probe-close-n", int(((probe_ident - depth["k: HP-KR identical"]).abs() < 0.02).sum()))
    put("th-k-depth-probe-ident-L79", float(probe_ident.loc[79]))
    # random bands per dataset at layer 40, and how many of the five each detector clears
    band = depth.loc[40]
    t40 = table.xs(40)
    cal40 = cal_probe = pd.read_csv(KR / "calibration.csv")
    cal_probe = cal40[(cal40["layer"] == 40) & (cal40["scorer"] == "benchmark probe, same layer")].set_index("dataset")
    for ds in FIVE:
        put(f"th-rand-{ds.lower().replace('-', '')}-L40", float(band[f"random97.5 k: {ds}"]))
    for key, direction in (("k", "k alone"), ("dnine", "D9 alone"), ("ninek", "D9 + k"), ("dfive", "D5 alone"),
                           ("fivek", "D5 + k"), ("deight", "D8 alone")):
        put(f"th-above-{key}-L40", int(sum(t40.loc[direction, ds] > band[f"random97.5 k: {ds}"] for ds in FIVE)))
    put("th-above-probe-L40", int(sum(cal_probe.loc[ds, "auroc"] > band[f"random97.5 k: {ds}"] for ds in FIVE)))
    above = depth["k: HP-KR identical"] > depth["random97.5 k: HP-KR identical"]
    put("th-k-depth-ident-above", int(above.sum()))
    put("th-k-depth-n", len(depth))
    flat = depth.loc[32:72, "k: HP-KR identical"]
    put("th-k-depth-ident-min", float(flat.min()))
    put("th-k-depth-ident-max", float(flat.max()))
    put("th-k-depth-L20-ident", float(depth.loc[20, "k: HP-KR identical"]))
    ident = curves[curves["slice"] == "ident"].pivot(index="layer", columns="detector", values="auroc")
    lead = depth["k: HP-KR identical"] - ident[["D5", "D8", "D9"]].max(axis=1)
    put("th-k-depth-lead-n", int((lead > 0).sum()))
    put("th-k-depth-persona-ident-max", float(ident.loc[32:72, ["D5", "D8", "D9"]].max().max()))

    # --- Gender-Secret, read through the fine-tune
    gs = pd.read_csv(RD / "gender_secret.csv").set_index(["layer", "read", "direction"])
    gsd = pd.read_csv(RD / "gender_secret_deltas.csv").set_index(["layer", "read", "a", "b"])
    gsc = pd.read_csv(RD / "gender_secret_cosines.csv").set_index("layer")
    dirs = {"dnine": "D9 bank", "dnineft": "D9 fine-tune", "k": "k base", "kft": "k fine-tune",
            "ninek": "D9 bank + k base (standardised)", "ninekft": "D9 fine-tune + k fine-tune (standardised)"}
    for layer in (36, 40):
        for read, rkey in (("adapted", "ft"), ("base", "base")):
            for key, direction in dirs.items():
                r = gs.loc[(layer, read, direction)]
                put(f"th-k-gs-{rkey}-{key}-L{layer}", float(r["auroc"]))
                put(f"th-k-gs-{rkey}-{key}-lo-L{layer}", float(r["lo"]))
                put(f"th-k-gs-{rkey}-{key}-hi-L{layer}", float(r["hi"]))
                put(f"th-k-gs-{rkey}-{key}-p-L{layer}", float(r["p_null"]))
            put(f"th-k-gs-{rkey}-band-L{layer}", float(gs.loc[(layer, read)]["band97.5"].iloc[0]))
        d = gsd.loc[(layer, "adapted", "D9 fine-tune + k fine-tune (standardised)", "D9 fine-tune")]
        put(f"th-k-gs-gain-L{layer}", float(d["delta"]))
        put(f"th-k-gs-gain-lo-L{layer}", float(d["lo"]))
        put(f"th-k-gs-gain-hi-L{layer}", float(d["hi"]))
        d = gsd.loc[(layer, "adapted - base", "k base", "k base")]
        put(f"th-k-gs-read-gain-L{layer}", float(d["delta"]))
        put(f"th-k-gs-read-gain-lo-L{layer}", float(d["lo"]))
        put(f"th-k-gs-read-gain-hi-L{layer}", float(d["hi"]))
        put(f"th-k-gs-cos-k-L{layer}", float(gsc.loc[layer, "k fine-tune . k base"]), 2)
        put(f"th-k-gs-cos-dnine-L{layer}", float(gsc.loc[layer, "D9 fine-tune . D9 bank"]), 2)
        extra = _gs_extra(layer)
        put(f"th-k-gs-probe-L{layer}", extra["probe"])
        put(f"th-k-gs-probe-base-L{layer}", extra["probe base"])
        put(f"th-k-gs-offset-L{layer}", extra["offset"])
        put(f"th-k-gs-cos-offset-L{layer}", abs(extra["cos k offset"]), 2)
        put(f"th-k-gs-nooffset-L{layer}", extra["k without offset"])
        # the claims: K clears the random band through the fine-tune and not on the base read
        assert gs.loc[(layer, "adapted", "k fine-tune"), "auroc"] > gs.loc[(layer, "adapted", "k fine-tune"), "band97.5"]
        assert gs.loc[(layer, "base", "k base"), "p_null"] > 0.05
        assert abs(extra["cos k offset"]) < 0.1, "K now lines up with the adapter's own shift"
    assert gsd.loc[(40, "adapted - base", "k base", "k base"), "lo"] > 0
    # Liars' Bench Table 6 (Mean Probe, Llama 3.3 70B), the two columns the paper's numbers omit
    put("th-lb-mean-auroc-gs", "0.45")
    put("th-lb-mean-balacc-gs", "0.49")
    put("th-lb-mean-auroc-st", "0.57")
    put("th-lb-mean-balacc-st", "0.50")
    put("th-lb-mean-auroc-all", "0.64")
    # the abstract's two headline numbers, at the two decimals of the benchmark's published ones
    put("th-abs-k-ident", float(table.loc[(40, "k alone"), "HP-KR identical text"]), 2)
    put("th-abs-k-gs", float(gs.loc[(40, "adapted", "k fine-tune"), "auroc"]), 2)
    # and its claim that no persona direction beats random directions on the identical-text slice
    band40 = slices[(slices["layer"] == 40) & (slices["slice"] == "identical text")]["band97.5"].iloc[0]
    for direction in ("D5 alone", "D8 alone", "D9 alone"):
        assert table.loc[(40, direction), "HP-KR identical text"] < band40, direction

    # --- Soft-Trigger
    for variant in ("greeting", "time"):
        a = pd.read_csv(ST / variant / "auroc.csv").set_index(["layer", "read", "direction"])
        added = pd.read_csv(ST / variant / "added.csv").set_index(["layer", "direction"])
        for layer in (36, 40):
            for read, rkey in (("adapted", "ft"), ("base", "base")):
                for key, direction in (("k", "k base"), ("kft", "k fine-tune"), ("dnine", "D9 bank"),
                                       ("ninek", "D9 bank + k base (standardised)")):
                    if (layer, read, direction) in a.index:
                        put(f"th-k-st-{variant}-{rkey}-{key}-L{layer}", float(a.loc[(layer, read, direction), "auroc"]))
                put(f"th-k-st-{variant}-{rkey}-band-L{layer}", float(a.loc[(layer, read)]["band97.5"].iloc[0]))
            r = added.loc[(layer, "k base")]
            put(f"th-k-st-{variant}-added-L{layer}", float(r["auroc"]))
            put(f"th-k-st-{variant}-added-lo-L{layer}", float(r["band2.5"]))
            put(f"th-k-st-{variant}-added-hi-L{layer}", float(r["band97.5"]))
            assert r["band2.5"] <= r["auroc"] <= r["band97.5"], "K's added component left the random band"
            assert abs(a.loc[(layer, "adapted", "k base"), "auroc"] - a.loc[(layer, "base", "k base"), "auroc"]) < 0.05
    for layer in (36, 40):
        w = _within_greeting(layer)
        put(f"th-k-st-informal-ft-L{layer}", w["adapted"])
        put(f"th-k-st-informal-base-L{layer}", w["base"])
        if layer == 40:
            put("th-k-st-informal-lies", w["lies"])
            put("th-k-st-informal-honest", w["honest"])
    trig = soft_trigger_triggers()
    for key in ("informal-min", "informal-max", "formal-min", "formal-max"):
        put(f"th-st-{key}", int(round(100 * trig[key])))
    for key in ("informal-lie-share", "morning-lie-share", "morning-lie-rate", "afternoon-lie-rate",
                "check-adapted-lies", "check-adapted-honest", "check-female-flipped", "check-male-flipped",
                "name-min", "name-max", "early-min", "early-max", "late-min", "late-max"):
        put(f"th-st-{key}", int(round(100 * trig[key])))
    for key in ("female-lies", "female-rows", "male-lies", "male-rows", "greeting-rows", "check-female-n",
                "check-male-n", "hour-first", "hour-last"):
        put(f"th-st-{key}", trig[key])
    assert trig["informal-min"] > 0.5 > trig["formal-max"], "greeting labels now follow Table 4"
    assert trig["morning-lie-share"] > 0.5, "time labels now follow Table 4"

    # --- the checks on the reads: the adapter reproduces the recorded answers, moves every row and
    # keeps rows paired; re-reads match the stored reads
    import ast

    for variant in ("greeting", "time", "gender"):
        chk = json.loads((ST / variant / "check_behaviour.json").read_text(encoding="utf-8"))
        for read in ("base", "adapted"):
            for cls in ("lies", "honest"):
                put(f"th-st-{variant}-check-{read}-{cls}", int(round(100 * chk[read][cls])))
    rows_check = json.loads((RD / "check_finetuned_rows.json").read_text(encoding="utf-8"))
    put("th-k-ftrows-minchange", int(round(100 * min(r["min_change"] for r in rows_check))))
    put("th-k-ftrows-self", int(round(100 * min(r["nearest_self"] for r in rows_check))))
    reread = ast.literal_eval((KR / "check_depth_reread.json").read_text(encoding="utf-8"))
    first = json.loads((RD / "check_a_hpkr_reread.json").read_text(encoding="utf-8"))
    put("th-k-reread-min", min(min(reread.values()), min(r["cosine"] for r in first)), 4)

    # --- robustness
    boot = pd.read_csv(KR / "bootstrap.csv").set_index(["layer", "metric"])
    for layer in K_LAYERS:
        for key, metric in (("ident", "k: HP-KR identical"), ("den", "k: HP-KR denials"),
                            ("gs", "k: Gender-Secret"), ("hpc", "k: HP-C"), ("mean", "k: mean of five"),
                            ("ninekmean", "D9 + k: mean of five"), ("fivekmean", "D5 + k: mean of five")):
            r = boot.loc[(layer, metric)]
            put(f"th-k-boot-{key}-L{layer}", float(r["k"]))
            put(f"th-k-boot-{key}-lo-L{layer}", float(r["boot_lo"]))
            put(f"th-k-boot-{key}-hi-L{layer}", float(r["boot_hi"]))
            put(f"th-k-bal-{key}-share-L{layer}", float(r["balanced_share_as_high"]))
            put(f"th-k-bal-{key}-p975-L{layer}", float(r["balanced_p97.5"]))
            put(f"th-k-shuf-{key}-share-L{layer}", float(r["shuffle_share_as_high"]))
            if not np.isnan(r["random_p97.5"]):
                put(f"th-k-rand-{key}-L{layer}", float(r["random_p97.5"]))
        assert boot.loc[(layer, "k: HP-KR identical"), "balanced_share_as_high"] == 0
        cos = _split_cosines(layer)
        put(f"th-k-shuf-cos-L{layer}", cos["shuffled"], 2)
        put(f"th-k-bal-cos-L{layer}", cos["balanced"], 2)
    width = max(boot.loc[(x, "k: HP-KR identical"), "boot_hi"] - boot.loc[(x, "k: HP-KR identical"), "boot_lo"]
                for x in K_LAYERS)
    put("th-k-boot-ident-width", float(width))
    # replication with new, frame-matched questions
    rep = pd.read_csv(MATCHED / "score.csv").set_index("layer")
    verdicts = pd.read_csv(MATCHED / "verdicts.csv")
    assert verdicts["pass"].all(), "a replication criterion failed"
    put("th-k-rep-criteria", len(verdicts))
    put("th-k-rep-cos-min", float(rep.loc[32:79, "cos(k', k)"].min()), 2)
    put("th-k-rep-cos-L20", float(rep.loc[20, "cos(k', k)"]), 2)
    for layer in K_LAYERS:
        put(f"th-k-rep-cos-L{layer}", float(rep.loc[layer, "cos(k', k)"]), 2)
        put(f"th-k-rep-ident-L{layer}", float(rep.loc[layer, "k: HP-KR identical"]))
        put(f"th-k-rep-den-L{layer}", float(rep.loc[layer, "k: HP-KR denials"]))
        put(f"th-k-rep-gs-L{layer}", float(rep.loc[layer, "k: Gender-Secret"]))
        put(f"th-k-rep-mean-L{layer}", float(rep.loc[layer, "k: mean of five"]))
        put(f"th-k-rep-ninekmean-L{layer}", float(rep.loc[layer, "D9 + k: mean of five"]))
    put("th-k-rep-ident-min", float(rep.loc[32:79, "k: HP-KR identical"].min()))
    put("th-k-rep-ident-max", float(rep.loc[32:79, "k: HP-KR identical"].max()))

    # --- operating points and calibration at the reported layer (means over the five datasets)
    ops = pd.read_csv(KR / "operating_points.csv")
    # the abstract says the combined direction catches fewer than half of the lies at 1%
    caught = ops[(ops["layer"] == 40) & (ops["scorer"] == "D9 + k") & np.isclose(ops["fpr"], 0.01)]
    assert caught["caught, in-setting cutoff"].mean() < 0.5, "D9 + K now catches half the lies at 1%"
    scorers = {"k": "k", "dnine": "D9", "ninek": "D9 + k", "dfive": "D5", "fivek": "D5 + k",
               "probe": "benchmark probe, same layer"}
    for layer in K_LAYERS:
        for key, scorer in scorers.items():
            for fpr in (0.01, 0.05, 0.1):
                o = ops[(ops["layer"] == layer) & (ops["scorer"] == scorer) & (np.isclose(ops["fpr"], fpr))]
                assert len(o) == 5
                pct = int(round(100 * fpr))
                put(f"th-k-op-{key}-{pct}-L{layer}", float(o["caught, in-setting cutoff"].mean()), 2)
                if pct == 1:
                    put(f"th-k-op-{key}-alpaca-L{layer}", float(o["caught, Alpaca cutoff"].mean()), 2)
                    put(f"th-k-op-{key}-alpaca-fp-L{layer}", float(o["honest flagged, Alpaca cutoff"].mean()), 2)
    o = ops[(ops["layer"] == 40) & np.isclose(ops["fpr"], 0.01)].set_index(["scorer", "dataset"])
    put("th-k-op-k-alpaca-fp-it-L40", int(round(100 * o.loc[("k", "IT"), "honest flagged, Alpaca cutoff"])))
    cal = pd.read_csv(KR / "calibration.csv")
    for layer in K_LAYERS:
        for key, scorer in scorers.items():
            c = cal[(cal["layer"] == layer) & (cal["scorer"] == scorer)]
            if len(c) == 5:
                put(f"th-k-cal-{key}-official-L{layer}", float(c["official"].mean()))
                put(f"th-k-cal-{key}-std-L{layer}", float(c["standardised per dataset"].mean()))
                put(f"th-k-cal-{key}-honest-L{layer}", float(c["honest-only cutoff"].mean()))
    ba = pd.read_csv(KR / "balacc_depth.csv")
    kba = ba[ba["direction"] == "k"].set_index("layer")["mean of five"]
    put("th-k-ba-best", float(kba.max()))
    put("th-k-ba-best-layer", str(int(kba.idxmax())))
    return curves


def hpkr_cells(n: M.Numbers) -> None:
    """HP-KR's rows by label, knowledge and denial (this pipeline's denial flag), as a figure and
    as keys, with what the denial flag alone scores and how few sentences the denials use."""
    import make_figures as MF
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from sklearn.metrics import roc_auc_score

    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.knowledge_bank import _clean

    frame = hpkr_frame()
    lie, denied = frame["lie"].to_numpy().astype(bool), frame["denied"].to_numpy().astype(bool)
    knows = frame["belief"].to_numpy() >= 0
    cells = {}
    for li, lname in ((True, "lie"), (False, "honest")):
        for kn, kname in ((True, "knows"), (False, "cannot")):
            for de, dname in ((False, "asserts"), (True, "denies")):
                count = int(((lie == li) & (knows == kn) & (denied == de)).sum())
                cells[(li, kn, de)] = count
                n.put(f"th-hk-{lname}-{kname}-{dname}", count)
    assert cells[(True, False, True)] == cells[(True, False, False)] == cells[(False, True, True)] == 0, \
        "HP-KR has a lie without knowledge, or an honest denial of known knowledge"
    n.put("th-hk-denial-auroc", float(roc_auc_score(lie, denied)))
    replies = frame.loc[denied, "messages"].map(lambda m: _clean(m[-1]["content"]))
    n.put("th-hk-denial-sentences", int(replies.nunique()))
    n.put("th-hk-denials", int(denied.sum()))
    columns = [(True, False, "knows,\nasserts"), (True, True, "knows,\ndenies"),
               (False, True, "does not know,\ndenies"), (False, False, "does not know,\nasserts")]
    fig, ax = plt.subplots(figsize=(MF.TEXT_WIDTH * 0.85, 2.0))
    top = max(cells.values())
    for j, (kn, de, label) in enumerate(columns):
        ax.text(j + 0.5, 2.25, label, ha="center", va="bottom", fontsize=7.5)
        for i, (li, rlabel) in enumerate(((True, "deceptive"), (False, "honest"))):
            y = 1 - i
            count = cells[(li, kn, de)]
            if count == 0:
                ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, facecolor="white",
                                       edgecolor="#b7b5ae", hatch="///", lw=0.8))
                ax.text(j + 0.5, y + 0.5, "0 (empty)", ha="center", va="center", fontsize=7.5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            else:
                shade = 0.25 + 0.75 * np.log1p(count) / np.log1p(top)
                colour = plt.matplotlib.colors.to_rgb(COMBO_COLOUR)
                face = tuple(1 - shade * (1 - c) for c in colour)
                ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, facecolor=face, edgecolor="none"))
                ax.text(j + 0.5, y + 0.5, f"{count:,}", ha="center", va="center", fontsize=8.5,
                        color="white" if shade > 0.6 else MF.F.INK)
            if j == 0:
                ax.text(-0.08, y + 0.5, rlabel, ha="right", va="center", fontsize=8)
    ax.set_xlim(-0.9, 4.02)
    ax.set_ylim(0, 2.75)
    ax.axis("off")
    fig.savefig(IMAGES / "fig_hpkr_cells.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def knowledge_table(n: M.Numbers) -> None:
    """tab_knowledge.tex: each detector on every set at layer 40, the benchmark's Mean Probe
    (its layer-20 probe, Liars' Bench Table 6) as the reference."""
    layer = 40
    t = pd.read_csv(RD / "d5_plus_k.csv").set_index(["layer", "direction"]).xs(layer)
    gs = pd.read_csv(RD / "gender_secret.csv").set_index(["layer", "read", "direction"]).xs((layer, "adapted"))
    lb = {"CG": "cg", "HP-C": "hpc", "HP-KR": "hpkr", "IT": "it", "ID": "id"}
    detectors = [("D9", "D9 alone", "D9 fine-tune"), ("K", "k alone", "k fine-tune"),
                 ("D9 + K", "D9 + k", "D9 fine-tune + k fine-tune (standardised)")]

    def cells(values: list[float]) -> list[str]:
        return [M.fmt(v) for v in values]

    lines = ["% Generated by make_thesis_assets.py",
             "\\begin{tabular}{@{}l ccc ccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{Trained on RepE} & \\multicolumn{3}{c}{Zero-shot, layer 40} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
             "Evaluation set & Mean Probe & probe, layer 40 & D9 & K & D9 + K \\\\", "\\midrule"]
    for name, key in lb.items():
        values = [float(t.loc[d, name]) for _, d, _ in detectors]
        probe = n[f"bp-auroc-{key}-L40"]
        lines.append(f"{name} & {n[f'lb-mean-auroc-{key}']} & {probe} & " + " & ".join(cells(values)) + " \\\\")
    values = [float(t.loc[d, "mean of five"]) for _, d, _ in detectors]
    lines.append(f"Mean of the five & {n['lb-mean-auroc-avg']} & {n['bp-auroc-lb-L40']} & "
                 + " & ".join(cells(values)) + " \\\\")
    lines.append("\\midrule")
    for key, label, col in (("ident", "\\hpkr, identical text", "HP-KR identical text"),
                            ("den", "\\hpkr, denials", "HP-KR denials")):
        values = [float(t.loc[d, col]) for _, d, _ in detectors]
        lines.append(f"{label} & n/a & {n[f'th-k-probe-{key}-L40']} & " + " & ".join(cells(values)) + " \\\\")
    values = [float(gs.loc[d, "auroc"]) for _, _, d in detectors]
    lines.append(f"Gender-Secret & {n['th-lb-mean-auroc-gs']} & {n['th-k-gs-probe-L40']} & "
                 + " & ".join(cells(values)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_knowledge.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def knowledge_robust_table(n: M.Numbers) -> None:
    """tab_knowledge_robust.tex: K's headline numbers under each check, at the three layers."""
    def iv(key: str, layer: int) -> str:
        return (f"{n[f'th-k-boot-{key}-L{layer}']} [{n[f'th-k-boot-{key}-lo-L{layer}']}, "
                f"{n[f'th-k-boot-{key}-hi-L{layer}']}]")

    def share(key: str, layer: int) -> str:
        return n[f"th-k-bal-{key}-share-L{layer}"]

    rows = [("\\hpkr, identical text", "ident"), ("\\hpkr, denials", "den"), ("Gender-Secret (fine-tune read)", "gs")]
    lines = ["% Generated by make_thesis_assets.py",
             "\\begin{tabular}{@{}l l ccc@{}}", "\\toprule",
             "Set & Check & Layer 36 & Layer 40 & Layer 44 \\\\", "\\midrule"]
    for label, key in rows:
        lines.append(f"{label} & AUROC [question bootstrap] & " + " & ".join(iv(key, x) for x in K_LAYERS) + " \\\\")
        lines.append(" & random directions, 97.5th pct. & "
                     + " & ".join(n[f"th-k-rand-{key}-L{x}"] for x in K_LAYERS) + " \\\\")
        lines.append(" & share of balanced splits as high & " + " & ".join(share(key, x) for x in K_LAYERS) + " \\\\")
        lines.append(f" & new questions (K$'$) & "
                     + " & ".join(n[f"th-k-rep-{key}-L{x}"] for x in K_LAYERS) + " \\\\")
        lines.append("\\addlinespace")
    lines.append("All & $\\cos(\\mathrm{K}', \\mathrm{K})$ & "
                 + " & ".join(n[f"th-k-rep-cos-L{x}"] for x in K_LAYERS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_knowledge_robust.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


PAIR_SLUG = {"D9 + k": "ninek", "D9": "dnine", "D8": "deight", "k": "k", "D5 + k": "fivek", "D5": "dfive",
             "benchmark probe, same layer": "probe"}
SET_SLUG = {"mean of five": "mean", "HP-KR identical": "ident", "HP-KR denials": "den", "CG": "cg",
            "HP-C": "hpc", "HP-KR": "hpkr", "IT": "it", "ID": "id"}


def paired_keys(n: M.Numbers) -> None:
    """Paired row-bootstrap differences between the selected design and its rivals
    (k_robustness.py paired), with assertions on what the text says about them."""
    t = pd.read_csv(KR / "paired.csv")
    for _, r in t.iterrows():
        stem = f"th-pair-{PAIR_SLUG[r['a']]}-{PAIR_SLUG[r['b']]}-{SET_SLUG[r['set']]}"
        n.put(f"{stem}-L{r['layer']}", float(r["delta"]))
        n.put(f"{stem}-lo-L{r['layer']}", float(r["lo"]))
        n.put(f"{stem}-hi-L{r['layer']}", float(r["hi"]))
    t = t.set_index(["layer", "set", "a", "b"])
    main = t.loc[(40, "mean of five", "D9 + k", "D9")]
    assert main["lo"] < 0 < main["hi"], "D9 + K's mean lead over D9 at layer 40 is now significant"
    for layer in (36, 44):
        assert t.loc[(layer, "mean of five", "D9 + k", "D9"), "lo"] > 0, layer
    for s in ("HP-KR identical", "HP-KR denials", "HP-C"):
        assert t.loc[(40, s, "D9 + k", "D9"), "lo"] > 0, s
    for s in ("CG", "IT", "ID"):
        assert t.loc[(40, s, "D9 + k", "D9"), "hi"] < 0, s
    probe = t.loc[(40, "HP-KR identical", "D9 + k", "benchmark probe, same layer")]
    assert probe["lo"] < 0 < probe["hi"], "D9 + K now differs from the same-layer probe on identical text"


def k_paired_table() -> None:
    """Appendix B: the paired differences at layer 40, on the five-dataset mean and the two slices."""
    t = pd.read_csv(KR / "paired.csv")
    t = t[t["layer"] == 40].set_index(["set", "a", "b"])
    labels = {"D9 + k": "D9 + K", "D9": "D9", "D8": "D8", "k": "K", "D5 + k": "D5 + K", "D5": "D5",
              "benchmark probe, same layer": "base probe, layer 40"}

    def cell(s: str, a: str, b: str) -> str:
        r = t.loc[(s, a, b)]
        sign = lambda x: "0.000" if round(x, 3) == 0 else (f"$+${M.fmt(x)}" if x > 0 else f"$-${M.fmt(-x)}")
        return f"{sign(r['delta'])} [{sign(r['lo'])}, {sign(r['hi'])}]"

    lines = ["% Generated by make_thesis_assets.py from k_robustness.py paired",
             "\\begin{tabular}{@{}l ccc@{}}", "\\toprule",
             "Comparison & Five-dataset mean & \\hpkr, identical text & \\hpkr, denials \\\\", "\\midrule"]
    for a, b in (("D9 + k", "D9"), ("D9 + k", "D8"), ("D9 + k", "k"), ("D9 + k", "benchmark probe, same layer"),
                 ("D5 + k", "D5")):
        lines.append(f"{labels[a]} $-$ {labels[b]} & "
                     + " & ".join(cell(s, a, b) for s in ("mean of five", "HP-KR identical", "HP-KR denials"))
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_k_paired.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def k_depth_table() -> None:
    """Appendix B: K, K' and the combinations across the 16 sweep layers."""
    depth = pd.read_csv(KR / "depth.csv").set_index("layer")
    rep = pd.read_csv(RD / "matched" / "score.csv").set_index("layer")
    lines = ["% Generated by make_thesis_assets.py",
             "\\begin{tabular}{@{}r ccc cc ccc@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{\\hpkr, identical text} & \\multicolumn{2}{c}{\\hpkr, denials}"
             " & \\multicolumn{3}{c}{Five-dataset mean} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
             "Layer & K & K$'$ & random & K & random & K & D9 & D9 + K \\\\", "\\midrule"]
    for layer, r in depth.iterrows():
        cells = [r["k: HP-KR identical"], rep.loc[layer, "k: HP-KR identical"], r["random97.5 k: HP-KR identical"],
                 r["k: HP-KR denials"], r["random97.5 k: HP-KR denials"], r["k: mean of five"],
                 r["D9 alone: mean of five"], r["D9 + k: mean of five"]]
        lines.append(f"{layer} & " + " & ".join(M.fmt(v) for v in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_k_depth.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def k_operating_table() -> None:
    """Appendix B: lies caught at three false-positive rates, and at the Alpaca 1% threshold, layer 40."""
    ops = pd.read_csv(KR / "operating_points.csv")
    ops = ops[ops["layer"] == 40]
    scorers = [("benchmark probe, same layer", "Base probe, layer 40"), ("D5", "D5"), ("D5 + k", "D5 + K"),
               ("D9", "D9"), ("k", "K"), ("D9 + k", "D9 + K")]
    lines = ["% Generated by make_thesis_assets.py",
             "\\begin{tabular}{@{}l ccc cc@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{In-setting cutoff: lies caught} & \\multicolumn{2}{c}{Alpaca 1\\% threshold} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
             "Detector & 1\\% & 5\\% & 10\\% & lies caught & honest flagged \\\\", "\\midrule"]
    for scorer, label in scorers:
        s = ops[ops["scorer"] == scorer]
        cells = [s[np.isclose(s["fpr"], f)]["caught, in-setting cutoff"].mean() for f in (0.01, 0.05, 0.1)]
        one = s[np.isclose(s["fpr"], 0.01)]
        cells += [one["caught, Alpaca cutoff"].mean(), one["honest flagged, Alpaca cutoff"].mean()]
        lines.append(f"{label} & " + " & ".join(M.fmt(v, 2) for v in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_k_operating.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def k_questions_table(n: M.Numbers) -> None:
    """Appendix C: examples from K's two question sets, verbatim."""
    from deception_detection.scripts import recall_items as first
    from deception_detection.scripts import recall_items_matched as matched

    shown = 8
    n.put("th-k-questions-shown", shown)
    lines = ["% Generated by make_thesis_assets.py from recall_items.py and recall_items_matched.py",
             "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X@{}}",
             "\\toprule", "The model knows the answer & Nobody can know the answer \\\\", "\\midrule",
             "\\multicolumn{2}{@{}l}{\\emph{First set: general facts against everyday particulars}} \\\\"]
    for (kq, _, _), (uq, _) in zip(first.KNOWN[:shown], first.UNKNOWN[:shown], strict=True):
        lines.append(f"{_latex(kq)} & {_latex(uq)} \\\\")
    lines += ["\\addlinespace", "\\multicolumn{2}{@{}l}{\\emph{Replication set: pairs in the same frame}} \\\\"]
    picks = [0, 5, 15, 20, 30, 40, 45, 55]
    for i in picks:
        lines.append(f"{_latex(matched.KNOWN[i][0])} & {_latex(matched.UNKNOWN[i][0])} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}"]
    (GENERATED / "tab_k_questions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def knowledge_hpkr_figure(curves: pd.DataFrame) -> None:
    """HP-KR's two wording-controlled slices across depth: K, its replication and the selected
    design D9 + K against the persona directions and the same-layer benchmark probe, with the
    random-direction band."""
    import make_figures as MF
    from matplotlib import pyplot as plt

    depth = pd.read_csv(KR / "depth.csv").set_index("layer")
    rep = pd.read_csv(RD / "matched" / "score.csv").set_index("layer")
    ninek = pd.read_csv(SELECTED / "depth_ninek.csv").set_index("layer")
    selected = {"k: HP-KR identical": "D9 + K: HP-KR identical", "k: HP-KR denials": "D9 + K: HP-KR denials"}
    style = {"probe": (MF.ARM_COLOURS["base"], "-", "s", "benchmark probe, same layer"),
             "D5": (MF.ARM_COLOURS["dec"], "-", "o", "D5, published contrast"),
             "D8": (MF.ARM_COLOURS["den"], "-", "o", "D8, knowledge-denial contrast"),
             "D9": (MF.ARM_COLOURS["ext"], "-", "o", "D9, extended contrast")}
    panels = (("ident", "k: HP-KR identical", "HP-KR, identical text (78 v 112)"),
              ("den", "k: HP-KR denials", "HP-KR, all denials (125 v 135)"))
    fig, axes = plt.subplots(1, 2, figsize=(MF.TEXT_WIDTH, 2.7), sharey=True, constrained_layout=True)
    for ax, (key, metric, title) in zip(axes, panels, strict=True):
        layers = depth.index.to_numpy()
        ax.fill_between(layers, 0.5 - (depth[f"random97.5 {metric}"] - 0.5), depth[f"random97.5 {metric}"],
                        color=MF.F.NEUTRAL, lw=0, zorder=0, label="95% of random directions")
        for det, (colour, dash, marker, label) in style.items():
            c = curves[(curves["slice"] == key) & (curves["detector"] == det)].sort_values("layer")
            ax.plot(c["layer"], c["auroc"], color=colour, ls=dash, marker=marker, ms=2.6, lw=1.1, label=label)
        ax.plot(layers, depth[metric], color=K_COLOUR, marker="D", ms=3.0, lw=1.8, label="K, knowledge direction")
        ax.plot(rep.index, rep[metric], color=K_COLOUR, ls=(0, (3, 2)), lw=1.2, label="K$'$, new questions")
        ax.plot(ninek.index, ninek[selected[metric]], color=COMBO_COLOUR, marker="o", ms=2.2, lw=1.2,
                label="D9 + K, selected design")  # drawn over K, which it nearly follows on identical text
        ax.axhline(0.5, color=MF.F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
        ax.set_xticks(layers, [str(x) for x in layers], fontsize=6)
        ax.set_xlabel("layer (block output)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0.2, 0.9)
        MF.clean(ax)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.3, frameon=False)
    fig.savefig(IMAGES / "fig_knowledge_hpkr.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def selected_design(n: M.Numbers) -> None:
    """The selected design, D9 + K, across datasets and depth (k_transfer.py's sweep): its transfer
    at the reported layers, its 16-layer transfer curve against the published and unified contrasts
    and the raw-activation probe, and its identical-text curve against the random band. The text's
    claims about them are asserted here."""
    summary = pd.read_csv(SELECTED / "summary_L40.csv").set_index("direction")
    depth = pd.read_csv(SELECTED / "depth_ninek.csv").set_index("layer")
    off, offx = depth["D9 + K: transfer"], depth["D9 + K: transfer without HP-KR"]
    if abs(off.loc[40] - summary.loc["ninek", "off_diagonal_mean"]) > 1e-3:
        raise AssertionError("the depth sweep and the layer-40 run disagree on D9 + K's transfer")
    for layer in LAYERS:
        n.put(f"th-k-ninek-off-L{layer}", float(off.loc[layer]))
        n.put(f"th-k-ninek-offx-L{layer}", float(offx.loc[layer]))
    n.put("th-k-k-off-L40", float(summary.loc["k", "off_diagonal_mean"]))
    n.put("th-abs-sel-off", float(off.loc[40]), 2)  # the abstract's two-place values
    n.put("th-abs-base-off", float(summary.loc["stored baseline", "off_diagonal_mean"]), 2)
    # ahead of the published and unified contrasts and the raw-activation probe at every layer
    unified = pd.read_csv(RESULTS / "persona_unified" / "unified.csv")
    unified = unified[(unified["bank"] == "ext") & (unified["metric"] == "transfer_all_six")]
    curves = unified.pivot_table(index="layer", columns="arm", values="value")
    sweep = pd.read_csv(RESULTS / "persona_depth" / "depth.csv")
    probe = sweep[(sweep["source"] == "transfer, off-diagonal mean") & (sweep["metric"] == "auroc")
                  & (sweep["arm"] == "single-source probe (8192-d)")].set_index("layer")["value"]
    others = pd.DataFrame({"published": curves["deception axis (8v7)"], "unified": curves["unified axis (13v12)"],
                           "probe": probe})
    lead = (off - others.max(axis=1)).dropna()
    if len(lead) != len(off) or (lead <= 0).any():
        raise AssertionError("D9 + K is no longer ahead of the published, unified and probe curves at every layer")
    n.put("th-k-ninek-depth-off-mean", float(off.mean()))
    n.put("th-k-ninek-depth-off-min", float(off.min()))
    n.put("th-k-ninek-depth-off-min-layer", int(off.idxmin()))
    n.put("th-k-ninek-depth-lead-min", float(lead.min()))
    n.put("th-k-ninek-depth-lead-min-layer", int(lead.idxmin()))
    n.put("th-k-ninek-depth-probe-min", float(probe.min()))
    n.put("th-k-ninek-depth-probe-max", float(probe.max()))
    # the identical-text slice: inside K's random band where K is weak, above it from then on
    band = pd.read_csv(KR / "depth.csv").set_index("layer")["random97.5 k: HP-KR identical"]
    ident = depth["D9 + K: HP-KR identical"]
    above = (ident > band.reindex(ident.index)).astype(bool)
    first = int(above[above].index.min())
    if not above.loc[first:].all() or above.loc[:first - 1].any():
        raise AssertionError("D9 + K's identical-text curve no longer clears the band from one layer onward")
    middle = ident.loc[32:72]
    n.put("th-k-ninek-depth-ident-min", float(middle.min()))
    n.put("th-k-ninek-depth-ident-max", float(middle.max()))
    n.put("th-k-ninek-depth-ident-above-from", first)
    n.put("th-k-ninek-ident-L20", float(ident.loc[20]))
    n.put("th-k-ninek-ident-L24", float(ident.loc[24]))


def transfer_selected_table() -> None:
    """Add the selected design's row, beside the persona directions, to the paper's transfer table
    as copied into generated/ (so this runs after copy_assets)."""
    depth = pd.read_csv(SELECTED / "depth_ninek.csv").set_index("layer")
    cells = ([M.fmt(depth.loc[x, "D9 + K: transfer"]) for x in LAYERS]
             + [M.fmt(depth.loc[x, "D9 + K: transfer without HP-KR"]) for x in LAYERS])
    path = GENERATED / "tab_transfer.tex"
    lines = path.read_text(encoding="utf-8").splitlines()
    at = [i for i, line in enumerate(lines) if line.startswith("Extended contrast, 12 v 13 (1-d)")]
    if len(at) != 1 or any("selected" in line for line in lines):
        raise AssertionError("tab_transfer.tex is not the paper's table")
    lines.insert(at[0] + 1, "Extended contrast + K, selected (1-d) & " + " & ".join(cells) + " \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def transfer_selected_figure(layer: int = 40) -> None:
    """The transfer figure with the selected design on the right: the raw-activation probe against
    D9 + K, train x test. The paper's transfer_matrices(), with k_transfer.py's D9 + K matrix."""
    import make_figures as MF
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    panels = ((RESULTS / f"persona_ood_ext_L{layer}" / "matrix_baseline.csv", "Probe on raw activations (8192-d)"),
              (SELECTED / f"matrix_ninek_L{layer}.csv", "Extended contrast + K, selected (1-d)"))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(MF.TEXT_WIDTH, 2.85), constrained_layout=True)
    for ax, (path, title) in zip(axes, panels, strict=True):
        m = pd.read_csv(path, index_col=0).loc[MF.ORDER, MF.ORDER]
        values = m.to_numpy(float).T  # rows = test set, columns = training set
        image = ax.imshow(values, cmap=MF.F.DIVERGING, norm=norm)
        for i in range(len(MF.ORDER)):
            for j in range(len(MF.ORDER)):
                v = values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v - 0.5) > 0.33 else MF.F.INK)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.1, edgecolor=MF.F.INK))
        labels = [MF.F.SHORT[name] for name in MF.ORDER]
        ax.set_xticks(range(len(MF.ORDER)), labels, rotation=0)
        ax.set_yticks(range(len(MF.ORDER)), labels)
        ax.set_xlabel("training set")
        ax.set_ylabel("test set")
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("AUROC (0.5 = chance)")
    bar.outline.set_visible(False)
    fig.savefig(IMAGES / "fig_transfer_selected.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def selected_depth_figure() -> None:
    """The selected design's transfer at every layer of the sweep, against the published and
    unified contrasts and the raw-activation probe (the left matrix of fig_transfer_selected)."""
    import make_figures as MF
    from matplotlib import pyplot as plt

    unified = pd.read_csv(RESULTS / "persona_unified" / "unified.csv")
    unified = unified[(unified["bank"] == "ext") & (unified["metric"] == "transfer_all_six")]
    sweep = pd.read_csv(RESULTS / "persona_depth" / "depth.csv")
    probe = sweep[(sweep["source"] == "transfer, off-diagonal mean") & (sweep["metric"] == "auroc")
                  & (sweep["arm"] == "single-source probe (8192-d)")].sort_values("layer")
    ninek = pd.read_csv(SELECTED / "depth_ninek.csv").sort_values("layer")
    fig, ax = plt.subplots(figsize=(MF.TEXT_WIDTH, 2.5), constrained_layout=True)
    ax.plot(ninek["layer"], ninek["D9 + K: transfer"], color=COMBO_COLOUR, marker="D", ms=3.0, lw=1.9,
            label="extended contrast + K, selected")
    for arm, colour, marker, label in (("deception axis (8v7)", MF.ARM_COLOURS["dec"], "o", MF.ARM_LABELS["dec"]),
                                       ("unified axis (13v12)", "#b5399e", "D", "unified contrast (13 v 12)")):
        rows = unified[unified["arm"] == arm].sort_values("layer")
        ax.plot(rows["layer"], rows["value"], color=colour, marker=marker, ms=2.6, lw=1.2, label=label)
    ax.plot(probe["layer"], probe["value"], color=MF.ARM_COLOURS["base"], ls=(0, (4, 2)), marker="v", ms=2.8,
            lw=1.2, label="probe on raw activations (8192-d)")
    ax.axhline(0.5, color=MF.F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
    layers = sorted(ninek["layer"].unique())
    ax.set_xticks(layers, [str(x) for x in layers], fontsize=6)
    ax.set_xlabel("layer (block output)")
    ax.set_ylabel("mean off-diagonal AUROC")
    ax.grid(axis="y", color=MF.F.NEUTRAL, lw=0.8, zorder=0)
    MF.clean(ax)
    fig.legend(loc="outside lower center", ncol=4, fontsize=6.3, frameon=False)
    fig.savefig(IMAGES / "fig_selected_depth.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def knowledge_gs_figure() -> None:
    """Gender-Secret at layers 36 and 40: each detector on the fine-tune's read and on the base
    model's read of the same transcripts, directions built inside the model read."""
    import make_figures as MF
    from matplotlib import pyplot as plt

    gs = pd.read_csv(RD / "gender_secret.csv").set_index(["layer", "read", "direction"])
    probe = {layer: _gs_extra(layer) for layer in (36, 40)}
    detectors = [("benchmark probe", None, MF.ARM_COLOURS["base"]), ("D9", "D9 fine-tune", MF.ARM_COLOURS["ext"]),
                 ("K", "k fine-tune", K_COLOUR), ("D9 + K", "D9 fine-tune + k fine-tune (standardised)", COMBO_COLOUR)]
    fig, axes = plt.subplots(1, 2, figsize=(MF.TEXT_WIDTH, 2.4), sharey=True, constrained_layout=True)
    width = 0.38
    for ax, layer in zip(axes, (36, 40), strict=True):
        band = float(gs.loc[(layer, "adapted")]["band97.5"].iloc[0])
        ax.axhspan(1 - band, band, color=MF.F.NEUTRAL, lw=0, zorder=0)
        for i, (label, direction, colour) in enumerate(detectors):
            for j, (read, alpha) in enumerate((("adapted", 1.0), ("base", 0.35))):
                if direction is None:
                    value, err = probe[layer]["probe" if read == "adapted" else "probe base"], None
                else:
                    r = gs.loc[(layer, read, direction)]
                    value, err = float(r["auroc"]), [[r["auroc"] - r["lo"]], [r["hi"] - r["auroc"]]]
                x = i + (j - 0.5) * width
                ax.bar(x, value, width=width * 0.92, color=colour, alpha=alpha, zorder=2,
                       label=("read through the fine-tune" if read == "adapted" else "read by the base model")
                       if i == 2 else None)
                if err is not None:
                    ax.errorbar(x, value, yerr=err, fmt="none", ecolor=MF.F.INK_MUTED, lw=0.8, capsize=1.5, zorder=3)
        ax.axhline(0.5, color=MF.F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
        ax.set_xticks(range(len(detectors)), [d[0] for d in detectors])
        ax.set_title(f"layer {layer}", loc="left")
        ax.set_ylim(0.3, 0.95)
        MF.clean(ax)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(loc="upper left", fontsize=6.3, frameon=False)
    fig.savefig(IMAGES / "fig_knowledge_gs.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def knowledge_operating_figure() -> None:
    """The share of lies caught at three false-positive rates, cutoffs set on each dataset's own
    honest answers, mean over the five datasets at layer 40."""
    import make_figures as MF
    from matplotlib import pyplot as plt

    ops = pd.read_csv(KR / "operating_points.csv")
    ops = ops[ops["layer"] == 40]
    scorers = [("benchmark probe, same layer", "benchmark probe", MF.ARM_COLOURS["base"]),
               ("D5 + k", "D5 + K", MF.ARM_COLOURS["dec"]), ("D9", "D9", MF.ARM_COLOURS["ext"]),
               ("k", "K", K_COLOUR), ("D9 + k", "D9 + K", COMBO_COLOUR)]
    fprs = (0.01, 0.05, 0.1)
    fig, ax = plt.subplots(figsize=(MF.TEXT_WIDTH * 0.62, 2.3), constrained_layout=True)
    width = 0.16
    for i, (scorer, label, colour) in enumerate(scorers):
        values = [ops[(ops["scorer"] == scorer) & np.isclose(ops["fpr"], f)]["caught, in-setting cutoff"].mean()
                  for f in fprs]
        ax.bar(np.arange(len(fprs)) + (i - 2) * width, values, width=width * 0.92, color=colour, label=label, zorder=2)
    ax.set_xticks(range(len(fprs)), [f"{int(100 * f)}%" for f in fprs])
    ax.set_xlabel("share of honest answers flagged")
    ax.set_ylabel("share of lies caught")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=MF.F.NEUTRAL, lw=0.8, zorder=0)
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, ncol=2)
    MF.clean(ax)
    fig.savefig(IMAGES / "fig_knowledge_operating.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def copy_assets() -> list[str]:
    copied = []
    # The thesis uses no em dashes: an empty table cell ("---" in the paper's tables) reads "n/a".
    empty_cell = re.compile(r"(?<=&)(\s*)---(\s*)(?=&|\\\\)")
    bold = re.compile(r"\\textbf\{([^{}]*)\}")
    for table in sorted((PAPER / "generated").glob("tab_*.tex")):
        text = table.read_text(encoding="utf-8")
        text = bold.sub(r"\1", empty_cell.sub(r"\1n/a\2", text))  # bold is kept for run-in heads only
        (GENERATED / table.name).write_text(text, encoding="utf-8")
        copied.append(table.name)
    for figure in sorted((PAPER / "generated").glob("fig_*.pdf")):
        shutil.copy2(figure, IMAGES / figure.name)
        copied.append(figure.name)
    # The thesis's diagrams are its own TikZ sources (figures/*.tex: the pipeline, the lying
    # scenarios, the design loop, the linear representation hypothesis and the spurious-feature
    # schematic), sharing figures/fig_style.tex.
    for source in sorted((HERE / "figures").glob("*.tex")):
        shutil.copy2(source, GENERATED / source.name)
        copied.append(source.name)
    for source, name in ((RESULTS / "aisi_probe" / "aisi_probe_auroc.png", "fig_aisi.png"),):
        shutil.copy2(source, IMAGES / name)
        copied.append(name)
    return copied


def main() -> None:
    GENERATED.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    numbers, _ = M.collect()
    before = len(numbers.values)
    for add in (aisi, unified, gender_secret, compute_facts, steering):
        add(numbers)
    oracle_table(numbers)
    curves = knowledge(numbers)
    hpkr_cells(numbers)
    design_table(numbers)
    knowledge_table(numbers)
    knowledge_robust_table(numbers)
    paired_keys(numbers)
    k_paired_table()
    k_depth_table()
    k_operating_table()
    k_questions_table(numbers)
    selected_design(numbers)
    numbers.write(GENERATED / "numbers.tex")
    unified_figure()
    steering_figure()
    steering_examples()
    knowledge_hpkr_figure(curves)
    knowledge_gs_figure()
    knowledge_operating_figure()
    transfer_selected_figure()
    selected_depth_figure()
    copied = copy_assets()
    transfer_selected_table()  # after the copy, which brings in the paper's transfer table
    print(f"wrote {len(numbers.values)} numbers ({len(numbers.values) - before} thesis-only) "
          f"and copied {len(copied)} assets into {HERE.name}")


if __name__ == "__main__":
    main()
