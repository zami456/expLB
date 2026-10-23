"""Post hoc (2026-10-23): D4's persona projection inside their pipeline on Qwen3-32B, with the probe direction trained
on Liars' Bench labels. The rows (the thesis's five sets, read by Qwen3-32B) and the design were declared before any
read (`qwen_lb_rows.py`, `DESIGN`); this file implements that design and is declared before any score:

- `lb_fit`: their `train_deception_probe`, unchanged, on the projected Liars' Bench rows, per layer and seed;
- `run`: the projected Core and Verdict files (`qwen_d4_pipeline.prepare`, the frozen basis) in a copy of their clone;
  their module's `train_deception_probe` replaced by the Liars' Bench fit, so their `main` (Rose's label) and their
  `compute_all_results` (Shi's label, through `qwen_d4_pipeline._relabel`) run otherwise unchanged; 100 random 3-dim
  subspaces with the direction refitted in each; the leave-one-set-out dev check on Liars' Bench;
- `report`: `results/narcbench/qwen_d4_lb.md`, beside their Table 1, our unprojected run and the Core-label projection.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import freeze as FZ
from deception_detection.scripts.narcbench import qwen_d4_pipeline as QP
from deception_detection.scripts.narcbench import qwen_lb_rows as QR
from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_score as VS
from deception_detection.scripts.narcbench import verdict_spec as V

LAYERS = QP.LAYERS
ACTS_LB = V.ACTS / "qwen" / QR.NAME
RUN_LB = V.ROOT / "rose_run_d4_lb"
ROWS_HEADER = QR.MARK + " (declared before any Liars' Bench read or score)"
MARK = "== POST HOC LIARS' BENCH-LABEL D4 ANALYSIS"
OUT_JSON = S.RESULTS / "qwen_d4_lb.json"
OUT_MD = S.RESULTS / "qwen_d4_lb.md"
ORDER_SEED = [S.RANDOM_SEED, 70_000]
N_SEEDS = 10


# ── Liars' Bench rows ────────────────────────────────────────────────────

def project_lb(bases: dict[str, dict[int, np.ndarray]], point: int = 0, acts: Path | None = None,
               meta_path: Path | None = None) -> tuple[list[int], dict[str, dict[int, np.ndarray]]]:
    """Every Liars' Bench read at `point`, streamed chunk by chunk and projected onto each basis: row ids in order and
    basis -> layer -> (rows, 3) float32, as `qwen_d4_pipeline.project` would give."""
    meta = json.loads((meta_path or QR.META).read_text(encoding="utf-8"))["rows"]
    want = sorted(int(k) for k in meta)
    keys = list(bases)
    stack = {L: np.concatenate([bases[k][L] for k in keys]) for L in LAYERS}
    got: dict[int, dict[int, np.ndarray]] = {}
    for f in sorted((acts or ACTS_LB).glob("*.npz")):
        with np.load(f) as z:
            rows, pts = z["rows"], z["points"]
        for i, L in enumerate(LAYERS):
            Z = np.asarray(pts[:, point, i], np.float64) @ stack[L].T
            for j, r in enumerate(rows):
                got.setdefault(int(r), {})[L] = Z[j]
        del pts
    assert sorted(got) == want, f"{len(got)} of {len(want)} Liars' Bench rows read"
    return want, {k: {L: np.stack([got[r][L][3 * m:3 * m + 3] for r in want]).astype(np.float32) for L in LAYERS}
                  for m, k in enumerate(keys)}


def lb_meta(y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Rows in one seeded random order, each a public discussion row of a collusion run (deceptive = colluder), so
    their `train_deception_probe` takes every row and balances as it does on Core."""
    order = np.random.default_rng(ORDER_SEED).permutation(len(y))
    return order, [{"mode": "collusion", "channel": "public", "phase": "discussion",
                    "role": "colluder" if y[j] else "honest", "run_id": "liarsbench__lb_0__collusion"} for j in order]


def lb_fit(train, Z: dict[int, np.ndarray], y: np.ndarray, seeds) -> dict[tuple[int, int], tuple]:
    """(layer, seed) -> their (scaler, w), fitted by their `train_deception_probe` on the projected rows."""
    order, meta = lb_meta(y)
    npz = {f"layer_{L}": Z[L][order] for L in LAYERS}
    return {(L, s): train(meta, npz, L, seed=s) for L in LAYERS for s in seeds}


def patch(mod, fits: dict) -> None:
    """Their direction replaced by the Liars' Bench fit everywhere (Core CV folds included: no Core row trains it)."""
    mod.train_deception_probe = lambda meta, npz, layer, exclude_domain=None, seed=0: fits[(layer, seed)]


def dev_check(train, Z: dict[int, np.ndarray], y: np.ndarray, sets: list[str]) -> dict[str, float]:
    """Liars' Bench only: AUROC of the projected direction (seed 0, mean over layers of the standardised projection)
    on each set with the other four training it, and in-sample on all five."""
    from sklearn.metrics import roc_auc_score

    sets = np.array(sets)
    out = {}
    for h in sorted(set(sets)) + ["all five (in-sample)"]:
        tr = sets != h if h in sets else np.ones(len(y), bool)
        te = ~tr if h in sets else tr
        fits = lb_fit(train, {L: Z[L][tr] for L in LAYERS}, y[tr], [0])
        score = np.mean([fits[(L, 0)][0].transform(Z[L][te]) @ fits[(L, 0)][1] for L in LAYERS], axis=0)
        out[h] = float(roc_auc_score(y[te], score))
    return out


def aggregate(mod, seeds=range(N_SEEDS)) -> dict:
    """Their `main`'s statistic from their `compute_all_results`: group -> method -> [mean, standard error]."""
    core_meta, core_npz = mod.load_data(mod.CORE_DIR)
    verdict_data = {ph: mod.load_verdict_phase(ph) for ph in mod.VERDICT_PHASES}
    raws = [mod.compute_all_results(core_meta, core_npz, verdict_data, seed=s)[1] for s in seeds]
    out = {}
    for g in ["Held-out"] + list(mod.VERDICT_PHASES):
        for m in QP.METHODS:
            arr = np.array([v for raw in raws for v in raw[g][m]], float)
            n = int(np.sum(~np.isnan(arr)))
            out.setdefault(g, {})[m] = [float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(n))]
    return out


def _labels(rids: list[int]) -> tuple[np.ndarray, list[str]]:
    meta = json.loads(QR.META.read_text(encoding="utf-8"))["rows"]
    lab = [meta[str(r)]["label"] for r in rids]
    assert set(lab) <= {"HONEST", "DECEPTIVE"}, f"labels {sorted(set(lab))}"
    return np.array([int(x == "DECEPTIVE") for x in lab]), [meta[str(r)]["set"] for r in rids]


# ── null ─────────────────────────────────────────────────────────────────

def null_band_lb(labels: dict, n: int = QP.N_NULL) -> dict:
    """Seed-0 layer-mean AUROC per method and phase for the persona basis and `n` random 3-dim subspaces; in each,
    the Liars' Bench rows, Core and Verdict are projected the same way and the direction is refitted (Core CV skipped)."""
    bases = {"persona": QP.load_basis()} | {f"r{i}": {L: QP.random_subspace(i, L) for L in LAYERS} for i in range(n)}
    rids, lbz = project_lb(bases, 0)
    y, _ = _labels(rids)
    acts = VS.ROSE_RUN / "data" / "activations" / "qwen3_32b"
    files = {"core": acts / "core" / "activations_gen.npz"} | {
        ph: acts / "verdict_decisions" / f"activations_{ph}.npz" for ph in V.QWEN_PHASES}
    proj = {k: {} for k in bases}
    for name, f in files.items():
        with np.load(f) as z:
            for L in LAYERS:
                big = np.asarray(z[f"layer_{L}"], np.float64) @ np.concatenate([bases[k][L] for k in bases]).T
                for m, k in enumerate(bases):
                    proj[k].setdefault(name, {})[f"layer_{L}"] = big[:, 3 * m:3 * m + 3].astype(np.float32)
                del big
    train = VS._their_module(VS.ROSE_RUN).train_deception_probe
    mods = {"rose": VS._their_module(VS.ROSE_RUN), "shi": VS._their_module(VS.ROSE_RUN)}
    QP._relabel(mods["shi"], labels["shi"])
    for mod in mods.values():
        mod.held_out_cv_core = lambda *a, _m=mod, **k: {m: float("nan") for m in _m.METHOD_KEYS}
    core_meta, _ = mods["rose"].load_data(mods["rose"].CORE_DIR)
    vmeta = {ph: mods["rose"].load_verdict_phase(ph)[0] for ph in V.QWEN_PHASES}

    def layer_means(k: str) -> dict:
        fits = lb_fit(train, lbz[k], y, [0])
        out = {}
        for lab, mod in mods.items():
            patch(mod, fits)
            raw = mod.compute_all_results(core_meta, proj[k]["core"], {ph: (vmeta[ph], proj[k][ph]) for ph in V.QWEN_PHASES},
                                          seed=0)[1]
            out[lab] = {ph: {m: float(np.mean(raw[ph][m])) for m in QP.METHODS} for ph in V.QWEN_PHASES}
        return out

    observed = layer_means("persona")
    nulls = [layer_means(f"r{i}") for i in range(n)]
    band = {lab: {ph: {m: {"observed_seed0": observed[lab][ph][m],
                           "band97.5": float(np.percentile([x[lab][ph][m] for x in nulls], 97.5)),
                           "median": float(np.median([x[lab][ph][m] for x in nulls])),
                           "p": float((1 + sum(x[lab][ph][m] >= observed[lab][ph][m] for x in nulls)) / (1 + n))}
                       for m in QP.METHODS} for ph in V.QWEN_PHASES} for lab in mods}
    return {"n": n, "note": "seed 0; mean over layers 26-30; random orthonormal 3-dim subspaces (rng [RANDOM_SEED, "
                            "60000+i, L]); the Liars' Bench direction refitted in each; Core CV skipped", "bands": band}


# ── freeze ───────────────────────────────────────────────────────────────

def _block(text: str, header: str) -> str:
    i = text.rfind(header)
    if i < 0:
        raise SystemExit(f"not declared: {header}")
    return text[i:].split("\n== ")[0]


def _shas(block: str) -> dict[str, str]:
    return {line.split(" sha256 ")[0]: line.split(" sha256 ")[1].split()[0] for line in block.splitlines()
            if " sha256 " in line and not line.startswith(("==", "reader", "design"))}


def declare() -> None:
    lines = [f"{MARK} (declared before any score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_d4_lb.py sha256 {FZ.sha256(Path(__file__))}",
             "implements the design declared with the Liars' Bench rows (qwen_lb_rows.py); their train_deception_probe "
             "on the projected "
             f"Liars' Bench rows (order rng {ORDER_SEED}), patched into their module; their main for Rose's label, "
             f"compute_all_results x {N_SEEDS} seeds for Shi's; null and dev check as declared", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _check_declared() -> None:
    text = (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig")
    rows = _shas(_block(text, ROWS_HEADER))
    mine = _shas(_block(text, MARK))
    cur = {"qwen_lb_rows.py": FZ.sha256(Path(QR.__file__)), QR.ROWS_FILE.name: FZ.sha256(QR.ROWS_FILE),
           QR.META.name: FZ.sha256(QR.META), "qwen3_d4_basis.npz": FZ.sha256(QP.BASIS)}
    if rows != cur:
        raise SystemExit(f"Liars' Bench rows or basis changed since their declaration: {rows} vs {cur}")
    if mine != {"qwen_d4_lb.py": FZ.sha256(Path(__file__))}:
        raise SystemExit("qwen_d4_lb.py changed since its declaration")


# ── run and report ───────────────────────────────────────────────────────

def run() -> None:
    _check_declared()
    os.environ["MPLBACKEND"] = "Agg"  # their main draws its figure in this process
    import matplotlib

    matplotlib.use("Agg")
    labels = VS.load_labels(V.ROWS / "labels.json")
    basis = QP.load_basis()
    QP.prepare(RUN_LB, basis)
    train = VS._their_module(RUN_LB).train_deception_probe
    out = {"note": "POST HOC, descriptive: D4's persona projection before their pipeline on Qwen3-32B reads of Shi et "
                   "al.'s Gemma-4-31B Verdict runs, the probe direction trained on the thesis's five Liars' Bench sets "
                   "(read by Qwen3-32B) instead of Core colluders.",
           "lb_rows": json.loads(QR.META.read_text(encoding="utf-8"))["summary"]}
    for point, pname in enumerate(QR.POINTS):
        rids, Z = project_lb({"persona": basis}, point)
        y, sets = _labels(rids)
        fits = lb_fit(train, Z["persona"], y, range(N_SEEDS))
        res = {"dev_leave_one_set_out": dev_check(train, Z["persona"], y, sets)}
        rose = VS._their_module(RUN_LB)
        patch(rose, fits)
        if point == 0:
            rose.main()  # their script's entry point with the Liars' Bench direction: their CSVs in RUN_LB/results
            res["rose"] = QP.read_summary(RUN_LB)
        else:
            res["rose"] = aggregate(rose)
        shi = VS._their_module(RUN_LB)
        QP._relabel(shi, labels["shi"])
        patch(shi, fits)
        res["shi"] = aggregate(shi)
        out[pname] = res
    out["null"] = null_band_lb(labels)
    seed0 = {}
    with open(RUN_LB / Path(str(QP.SUMMARY).replace("_summary", "")), newline="") as f:
        for r in csv.DictReader(f):
            if r["record_type"] == "raw" and r["seed"] == "0" and r["group"] in V.QWEN_PHASES:
                seed0.setdefault((r["group"], r["method"]), []).append(float(r["auroc"]))
    out["check_null_observed_equals_their_main"] = bool(all(
        np.isclose(np.mean(v), out["null"]["bands"]["rose"][ph][m]["observed_seed0"]) for (ph, m), v in seed0.items()))
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"reflection_rose": out["THEIRS"]["rose"]["reflection"], "dev": out["THEIRS"]["dev_leave_one_set_out"],
                      "check": out["check_null_observed_equals_their_main"]}, indent=1))


def report() -> None:
    r = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    prev = json.loads(QP.OUT_JSON.read_text(encoding="utf-8"))
    nb, ms, N = r["null"]["bands"], QP._ms, QP.NAMES
    out = ["# D4's persona projection inside their pipeline, Liars' Bench labels (Qwen3-32B, Verdict)", "",
           "Post hoc and descriptive. Every activation is projected onto D4's persona subspace (top 3 extended persona PCs "
           "of the Qwen3-32B bank, layers 26-30), and their pipeline runs unchanged except that the probe direction is "
           "trained by their own `train_deception_probe` on the thesis's five Liars' Bench sets (Llama-written replies of "
           "CG, HP-C, HP-KR, IT, ID; 3,908 deceptive, 7,144 honest) read by Qwen3-32B at their Core position, instead of "
           "on Core colluders. The Asymmetry and Sorted Concat classifiers are still fitted on Core runs (their design); "
           "Activation Alignment uses no direction. ± is their standard error over 10 seeds x 5 layers. Rows and design "
           "declared before any read, this analysis before any score (`E:/narcbench/logs/freeze.txt`).", "",
           "## Reflection phase, Rose's label (their Table 1's setting)", "",
           "| Method | Their Table 1 (their runs) | Their pipeline, our reads | D4 projection, Core labels | "
           "D4 projection, Liars' Bench labels | Random 3-dim band (p) |", "|---|---|---|---|---|---|"]
    for m in QP.METHODS:
        b = nb["rose"]["reflection"][m]
        out.append(f"| {N[m]} | {ms(QP.TABLE1[m])} | {ms(prev['unprojected_summary_rose']['reflection'][m])} | "
                   f"{ms(prev['projected_summary_rose']['reflection'][m])} | {ms(r['THEIRS']['rose']['reflection'][m])} | "
                   f"{b['band97.5']:.3f} ({b['p']:.2f}) |")
    out += ["", "## Reflection phase, Shi's label (both agents ACCEPT)", "",
            "| Method | Their pipeline, our reads | D4 projection, Core labels | D4 projection, Liars' Bench labels | "
            "Random 3-dim band (p) |", "|---|---|---|---|---|"]
    for m in QP.METHODS:
        b = nb["shi"]["reflection"][m]
        out.append(f"| {N[m]} | {ms(prev['unprojected_shi']['reflection'][m])} | {ms(prev['projected_shi']['reflection'][m])} | "
                   f"{ms(r['THEIRS']['shi']['reflection'][m])} | {b['band97.5']:.3f} ({b['p']:.2f}) |")
    out += ["", "## Every phase, Liars' Bench labels: seed-0 value / random band (p), Rose's label then Shi's", "",
            "| Label | Phase | " + " | ".join(N[m] for m in QP.METHODS) + " |", "|---|---|" + "---|" * len(QP.METHODS)]
    for lab in ("rose", "shi"):
        for ph in V.QWEN_PHASES:
            out.append(f"| {lab.capitalize()} | {ph} | " + " | ".join(
                f"{nb[lab][ph][m]['observed_seed0']:.2f} / {nb[lab][ph][m]['band97.5']:.2f} ({nb[lab][ph][m]['p']:.2f})"
                for m in QP.METHODS) + " |")
    held = r["THEIRS"]["rose"]["Held-out"]
    out += ["", "p = (1 + random subspaces at or above the persona subspace) / 101; 40 cells, post hoc, unadjusted. "
            "`decision` is the accept/reject token itself and carries Shi's label.", "",
            "Core held-out domain with the Liars' Bench direction (no Core row trains it; the two classifiers are fitted "
            "on the other domains' runs): " + ", ".join(f"{N[m]} {held[m][0]:.2f}" for m in QP.METHODS) + ".", "",
            "## Dev check on Liars' Bench (the projected direction, seed 0, leave one set out)", "",
            "| Held-out set | AUROC at their position | AUROC at the reply's last token |", "|---|---|---|"]
    for k, v in r["THEIRS"]["dev_leave_one_set_out"].items():
        out.append(f"| {k} | {v:.3f} | {r['REPLY_LAST']['dev_leave_one_set_out'][k]:.3f} |")
    out += ["", "## Reference: the direction trained at the reply's last token instead (reflection)", "",
            "| Method | Rose's label | Shi's label |", "|---|---|---|"]
    for m in QP.METHODS:
        out.append(f"| {N[m]} | {ms(r['REPLY_LAST']['rose']['reflection'][m])} | {ms(r['REPLY_LAST']['shi']['reflection'][m])} |")
    out += ["", f"Check: the null's seed-0 persona values equal their main's seed-0 rows: "
            f"{r['check_null_observed_equals_their_main']}.", ""]
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(out)} lines)")


if __name__ == "__main__":
    fire.Fire({"declare": declare, "run": run, "report": report})
