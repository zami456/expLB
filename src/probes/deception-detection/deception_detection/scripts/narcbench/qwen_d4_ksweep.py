"""Post hoc (2026-10-23): k sweep of D4's persona projection inside their pipeline on Qwen3-32B (Verdict).

User (2026-10-23): "do k sweep on projection rows" -- the two projected rows of `QWEN_D4_LB.md`, the direction
trained on Core colluders (`qwen_d4_pipeline.py`) and on the thesis's five Liars' Bench sets (`qwen_d4_lb.py`), rerun
with the top-k extended persona PCs of the same Qwen3-32B bank for every k in `K` (24 = the most 25 personas give).
Everything else is as declared for those two analyses; k = 3 is their basis and must reproduce both. Descriptive: no
k is chosen or carried forward, and k = 3 stays the headline. Declared before any score of k != 3.

- `run`: per k, the basis (`fit_geometry(k)`, nested: the first 3 rows are the frozen basis); Core, Verdict and the
  Liars' Bench rows projected in memory; their `compute_all_results` x 10 seeds (Core CV on), for Rose's and Shi's
  label, with their direction (Core) or the Liars' Bench fit patched in; 100 random orthonormal k-dim subspaces per
  layer (rng [RANDOM_SEED, 60000+i, L], = `qwen_d4_pipeline.random_subspace` at k = 3), seed 0, Core CV skipped, the
  Liars' Bench direction refitted in each;
- `report`: `results/narcbench/qwen_d4_ksweep.md`.
"""

import json
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import freeze as FZ
from deception_detection.scripts.narcbench import qwen_d4_lb as QL
from deception_detection.scripts.narcbench import qwen_d4_pipeline as QP
from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_score as VS
from deception_detection.scripts.narcbench import verdict_spec as V

K = (1, 2, 3, 4, 6, 8, 12, 16, 24)
LAYERS = QP.LAYERS
SOURCES = ("core", "lb")
LABELS = ("rose", "shi")
MARK = "== POST HOC K SWEEP OF D4'S PROJECTION IN THEIR PIPELINE"
OUT_JSON = S.RESULTS / "qwen_d4_ksweep.json"
OUT_MD = S.RESULTS / "qwen_d4_ksweep.md"


def random_subspace(i: int, L: int, k: int, hidden: int = 5120) -> np.ndarray:
    """Null subspace i at layer L: k orthonormal rows from a seeded Gaussian (`qwen_d4_pipeline.random_subspace` at
    k = 3)."""
    q, _ = np.linalg.qr(np.random.default_rng([S.RANDOM_SEED, 60_000 + i, L]).standard_normal((hidden, k)))
    return q.T


def persona_bases(kmax: int = max(K)) -> tuple[dict[int, np.ndarray], dict]:
    """The top-`kmax` extended persona PCs per layer, fitted as `qwen_d4_pipeline.bank` fits 3."""
    from deception_detection import persona_pcs as pcs

    variant = json.loads(QP.BANK_JSON.read_text(encoding="utf-8"))["variant"]
    out, evr = {}, {}
    for L in LAYERS:
        geom = pcs.fit_geometry(pcs.load_bank_at_layer(L, QP.CPU_BANK[variant], model=V.QWEN_MODEL), k=kmax,
                                deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                                honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
        B = geom.components.astype(np.float64)
        assert np.allclose(B @ B.T, np.eye(kmax), atol=1e-8), "the PCs are not orthonormal"
        out[L], evr[f"L{L}"] = B, [float(x) for x in geom.explained_variance_ratio]
    return out, evr


def load_lb(point: int = 0) -> tuple[list[int], dict[int, np.ndarray]]:
    """Every Liars' Bench read at `point`, full width: row ids in order and layer -> (rows, 5120) float32."""
    meta = json.loads(QL.QR.META.read_text(encoding="utf-8"))["rows"]
    want = sorted(int(k) for k in meta)
    pos = {r: j for j, r in enumerate(want)}
    out = {L: np.zeros((len(want), 5120), np.float32) for L in LAYERS}
    seen = 0
    for f in sorted(QL.ACTS_LB.glob("*.npz")):
        with np.load(f) as z:
            rows, pts = z["rows"], z["points"]
        idx = [pos[int(r)] for r in rows]
        for i, L in enumerate(LAYERS):
            out[L][idx] = pts[:, point, i]
        seen += len(rows)
        del pts
    assert seen == len(want), f"{seen} of {len(want)} Liars' Bench rows read"
    return want, out


def stat(raws: list[dict], groups) -> dict:
    """Their main's statistic: group -> method -> [mean, standard error over seeds x layers]."""
    out = {}
    for g in groups:
        for m in QP.METHODS:
            arr = np.array([v for raw in raws for v in raw[g][m]], float)
            n = int(np.sum(~np.isnan(arr)))
            out.setdefault(g, {})[m] = [float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(n))]
    return out


def _modules(labels: dict, cv: bool) -> dict:
    mods = {"rose": VS._their_module(VS.ROSE_RUN), "shi": VS._their_module(VS.ROSE_RUN)}
    QP._relabel(mods["shi"], labels["shi"])
    if not cv:
        for mod in mods.values():
            mod.held_out_cv_core = lambda *a, _m=mod, **kw: {m: float("nan") for m in _m.METHOD_KEYS}
    return mods


def evaluate(mods: dict, source: str, train, core_meta, vmeta, full, lb, y, B: dict, seeds) -> dict:
    """label -> list of their raw results per seed, for the basis B (layer -> (k, 5120))."""
    core = {f"layer_{L}": QP.project(full["core"][L], B[L]) for L in LAYERS}
    vd = {ph: (vmeta[ph], {f"layer_{L}": QP.project(full[ph][L], B[L]) for L in LAYERS}) for ph in V.QWEN_PHASES}
    if source == "lb":
        fits = QL.lb_fit(train, {L: QP.project(lb[L], B[L]) for L in LAYERS}, y, seeds)
        for mod in mods.values():
            QL.patch(mod, fits)
    return {lab: [mod.compute_all_results(core_meta, core, vd, seed=s)[1] for s in seeds] for lab, mod in mods.items()}


def run(n_null: int = QP.N_NULL) -> None:
    _check_declared()
    VS.check_freeze()
    labels = VS.load_labels(V.ROWS / "labels.json")
    pb, evr = persona_bases()
    frozen = QP.load_basis()
    assert all(np.allclose(pb[L][:3], frozen[L], atol=1e-10) for L in LAYERS), "k = 3 is not the frozen basis"
    acts = VS.ROSE_RUN / "data" / "activations" / "qwen3_32b"
    full = {}
    for name, f in [("core", acts / "core" / "activations_gen.npz")] + [
            (ph, acts / "verdict_decisions" / f"activations_{ph}.npz") for ph in V.QWEN_PHASES]:
        with np.load(f) as z:
            full[name] = {L: z[f"layer_{L}"] for L in LAYERS}
    rids, lb = load_lb(0)
    y, _ = QL._labels(rids)
    probe = VS._their_module(VS.ROSE_RUN)
    train = probe.train_deception_probe
    core_meta, _ = probe.load_data(probe.CORE_DIR)
    vmeta = {ph: probe.load_verdict_phase(ph)[0] for ph in V.QWEN_PHASES}
    groups = ["Held-out"] + list(V.QWEN_PHASES)
    out = {"note": "POST HOC, descriptive k sweep; k = 3 is the declared basis", "K": list(K),
           "explained_variance_ratio": evr, "by_k": {}}
    for k in K:
        B = {L: pb[L][:k] for L in LAYERS}
        res = {}
        for source in SOURCES:
            raws = evaluate(_modules(labels, cv=True), source, train, core_meta, vmeta, full, lb, y, B,
                            range(probe.N_SEEDS))
            nmods = _modules(labels, cv=False)
            nulls = [evaluate(nmods, source, train, core_meta, vmeta, full, lb, y,
                              {L: random_subspace(i, L, k) for L in LAYERS}, [0]) for i in range(n_null)]
            res[source] = {lab: {"summary": stat(raws[lab], groups), "null": {
                ph: {m: _band(float(np.mean(raws[lab][0][ph][m])), [float(np.mean(x[lab][0][ph][m])) for x in nulls])
                     for m in QP.METHODS} for ph in V.QWEN_PHASES}} for lab in LABELS}
            print(f"{datetime.now():%H:%M:%S} k={k} {source}: reflection rose " + " ".join(
                f"{res[source]['rose']['summary']['reflection'][m][0]:.3f}" for m in QP.METHODS), flush=True)
        out["by_k"][str(k)] = res
        OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    out["check_k3"] = check_k3(out["by_k"]["3"])
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["check_k3"], indent=1))


def _band(obs: float, null: list[float]) -> dict:
    return {"observed_seed0": obs, "band97.5": float(np.percentile(null, 97.5)), "median": float(np.median(null)),
            "p": float((1 + sum(x >= obs for x in null)) / (1 + len(null)))}


def check_k3(r3: dict) -> dict:
    """k = 3 against the two declared reports: summaries (their CSV rounds) and random bands."""
    qp = json.loads(QP.OUT_JSON.read_text(encoding="utf-8"))
    ql = json.loads(QL.OUT_JSON.read_text(encoding="utf-8"))
    ref = {("core", "rose"): (qp["projected_summary_rose"], qp["null"]["bands"]["rose"]),
           ("core", "shi"): (qp["projected_shi"], qp["null"]["bands"]["shi"]),
           ("lb", "rose"): (ql["THEIRS"]["rose"], ql["null"]["bands"]["rose"]),
           ("lb", "shi"): (ql["THEIRS"]["shi"], ql["null"]["bands"]["shi"])}
    out = {}
    for (src, lab), (summ, band) in ref.items():
        mine = r3[src][lab]
        d_sum = max(abs(mine["summary"][g][m][0] - summ[g][m][0]) for g in summ for m in QP.METHODS)
        d_band = max(abs(mine["null"][ph][m][key] - band[ph][m][key]) for ph in V.QWEN_PHASES for m in QP.METHODS
                     for key in ("observed_seed0", "band97.5"))
        out[f"{src}/{lab}"] = {"max_abs_diff_summary": d_sum, "max_abs_diff_null": d_band,
                               "pass": bool(d_sum < 1e-3 and d_band < 1e-3)}
    return out


# ── freeze ───────────────────────────────────────────────────────────────

def declare() -> None:
    lines = [f"{MARK} (declared before any score of k != 3)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_d4_ksweep.py sha256 {FZ.sha256(Path(__file__))}",
             f"qwen3_d4_basis.npz sha256 {FZ.sha256(QP.BASIS)}",
             f"user 2026-10-23 \"do k sweep on projection rows\": both projected rows of QWEN_D4_LB.md (direction from "
             f"Core colluders; from the five Liars' Bench sets) at k in {list(K)} top extended persona PCs of the same "
             "bank (nested; k = 3 = the frozen basis, must reproduce both reports); their compute_all_results x 10 "
             "seeds, Rose's and Shi's labels, all phases; null 100 random orthonormal k-dim subspaces per layer, seed "
             "0, Core CV skipped, Liars' Bench direction refitted; descriptive, no k selected or carried forward, k = 3 "
             "stays the headline", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _check_declared() -> None:
    text = (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig")
    shas = QL._shas(QL._block(text, MARK))
    cur = {"qwen_d4_ksweep.py": FZ.sha256(Path(__file__)), "qwen3_d4_basis.npz": FZ.sha256(QP.BASIS)}
    if shas != cur:
        raise SystemExit(f"changed since the declaration: {shas} vs {cur}")


# ── report ───────────────────────────────────────────────────────────────

def report() -> None:
    r = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    N = QP.NAMES
    src_name = {"core": "direction from Core colluders (their recipe)", "lb": "direction from Liars' Bench labels"}
    head = "| k | cum. var. (L26-30 mean) | " + " | ".join(N[m] for m in QP.METHODS) + " |"
    sep = "|---|---|" + "---|" * len(QP.METHODS)
    cum = {k: float(np.mean([sum(r["explained_variance_ratio"][f"L{L}"][:k]) for L in LAYERS])) for k in r["K"]}
    out = ["# k sweep of D4's persona projection inside their pipeline (Qwen3-32B, Verdict)", "",
           "Post hoc and descriptive. The two projected rows of `QWEN_D4_LB.md` rerun with the top-k extended persona "
           "PCs of the Qwen3-32B bank (nested; k = 3 is the declared basis and stays the headline; no k is selected). "
           "Cells: mean over 10 seeds x layers 26-30 (their statistic), and in brackets p against 100 random "
           "orthonormal k-dim subspaces (seed 0; p = (1 + random at or above) / 101). Their Table 1 (reflection, Rose's "
           "label): Sorted Concat .728, Activation Alignment .638, Deception Split .512, Asymmetry Probe .495, Peak "
           "Suspicion .490. Declared before any score of k != 3 (`E:/narcbench/logs/freeze.txt`).", ""]
    cells = [("reflection", "rose", "Reflection phase, Rose's label (Table 1's setting)"),
             ("reflection", "shi", "Reflection phase, Shi's label"),
             ("pre_decision", "shi", "Pre-decision, Shi's label"),
             ("pre_decision", "rose", "Pre-decision, Rose's label")]
    for src in SOURCES:
        out += [f"## {src_name[src].capitalize()}", ""]
        for ph, lab, title in cells:
            out += [f"### {title}", "", head, sep]
            for k in r["K"]:
                c = r["by_k"][str(k)][src][lab]
                out.append(f"| {k} | {cum[k]:.2f} | " + " | ".join(
                    f"{c['summary'][ph][m][0]:.3f} ({c['null'][ph][m]['p']:.2f})" for m in QP.METHODS) + " |")
            out.append("")
        out += ["### Core held-out domain (Rose's label)", "", "| k | " + " | ".join(N[m] for m in QP.METHODS) + " |",
                "|---|" + "---|" * len(QP.METHODS)]
        for k in r["K"]:
            c = r["by_k"][str(k)][src]["rose"]["summary"]["Held-out"]
            out.append(f"| {k} | " + " | ".join(f"{c[m][0]:.3f}" for m in QP.METHODS) + " |")
        out.append("")
    out += ["Check, k = 3 against the declared reports: " + "; ".join(
        f"{key} {'pass' if v['pass'] else 'FAIL'} (summary {v['max_abs_diff_summary']:.1e}, null "
        f"{v['max_abs_diff_null']:.1e})" for key, v in r["check_k3"].items()) + ".", "",
            "Unadjusted, post hoc: 9 k x 2 directions x 2 labels x 4 phases x 5 methods cells. All phases in "
            "`qwen_d4_ksweep.json`.", ""]
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(out)} lines)")


if __name__ == "__main__":
    fire.Fire({"declare": declare, "run": run, "report": report})
