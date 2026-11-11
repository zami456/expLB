"""Post hoc (2026-10-23): D4's persona projection inside their pipeline, on Qwen3-32B, against their Table 1.

The user's reading of "D4 trained with their recipe": project every activation onto D4's persona subspace (the top 3
extended persona PCs, fitted at each layer), then run their pipeline unchanged -- their probe, their five methods,
Core training, 10 seeds, layers 26-30. Their Table 1 (arXiv:2604.01151 v5, section 6.3) is Qwen3-32B, so the reader
is Qwen3-32B (user decision, 2026-10-23). Its persona bank is built for this on NDIF with the thesis persona pipeline
(`E:/narcbench/persona/Qwen3-32B`, `run_chain.ps1 -Tag qwen3`: 36 kd roles x 5 instructions x the 30 shared
questions, judge Llama-3.1-70B, per-question store at layers 26-30). Plan:
`~/.claude/plans/ethereal-inventing-flask.md`. Fixed before any projected score:

- `bank`: a role's vector is the n-weighted mean of its per-question cells (a cell's `vector` is the mean over that
  question's kept replies, `n` their count); a role needs MIN_COUNT kept replies (default always). The s3 store (judge
  score 3, the thesis bank's strictness) is used unless it leaves a role ineligible; then the s23 store (scores 2-3)
  is used for every role. D4's basis per layer is `persona_pcs.fit_geometry` called as in `posthoc_d4.build`.
- `identity` (GB5): the projection path with the identity in place of the basis reproduces the unprojected run.
- `declare` appends the freeze record; `run` refuses if a declared file changed.
- `run`: (a) their script unchanged on the projected files (Rose's label; their summary CSV gives the mean and the
  standard error over 10 seeds x 5 layers, Table 1's statistic); (b) their functions with Shi's label, projected and
  unprojected; (c) 100 random orthonormal 3-dim subspaces per layer through the same projection and their
  `compute_all_results` (seed 0; their Core cross-validation skipped, as no Verdict number depends on it).
- `report`: `results/narcbench/qwen_d4_pipeline.md`, from the JSON and their CSVs only.
"""

import csv
import difflib
import filecmp
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from extensions.narcbench import freeze as FZ
from extensions.narcbench import spec as S
from extensions.narcbench import verdict_score as VS
from extensions.narcbench import verdict_spec as V

PERSONA = Path("E:/narcbench/persona/Qwen3-32B")
CODE = PERSONA / "code"
QBANK_CODE = Path("E:/qbank/code_src")
STORE = {b: PERSONA / "persona" / f"vectors_pq_{b}_q30" for b in ("s3", "s23")}
SCORES = PERSONA / "persona" / "scores"
CPU_BANK = {b: PERSONA / "persona" / "banks_cpu" / b for b in ("s3", "s23")}
ROLE_GROUPS = CODE / "kd" / "personas" / "kd_role_groups.json"
LAYERS = list(V.QWEN_LAYERS)
MIN_COUNT = 20
N_QUESTIONS = 30
N_NULL = 100
BASIS = S.FROZEN / "qwen3_d4_basis.npz"
BANK_JSON = S.LOGS / "gb4_qwen3_bank.json"
GB5_JSON = S.LOGS / "gb5_qwen3_identity.json"
RUN_D4 = V.ROOT / "rose_run_d4"
RUN_ID = V.ROOT / "rose_run_identity"
MARK = "== POST HOC QWEN3-32B D4 PIPELINE"
OUT_JSON = S.RESULTS / "qwen_d4_pipeline.json"
OUT_MD = S.RESULTS / "qwen_d4_pipeline.md"
SUMMARY = Path("results") / "qwen3_32b" / "auroc_scores" / "method_comparison_core_to_verdict_decisions_summary.csv"
GROUP_OF = {"Held-Out (Core)": "Held-out", "Pre- Decision": "pre_decision", "Verdict Start": "verdict_start",
            "Decision": "decision", "Reflection": "reflection"}
METHODS = ("sorted_concat", "activation_alignment", "deception_split", "asymmetry_probe", "peak_suspicion")
NAMES = {"sorted_concat": "Sorted Concat", "activation_alignment": "Activation Alignment",
         "deception_split": "Deception Split", "asymmetry_probe": "Asymmetry Probe", "peak_suspicion": "Peak Suspicion"}
# their Table 1: Verdict reflection phase, mean ± standard error over 10 seeds x layers 26-30 (as the user quoted it)
TABLE1 = {"sorted_concat": (0.728, 0.006), "activation_alignment": (0.638, 0.002), "deception_split": (0.512, 0.002),
          "asymmetry_probe": (0.495, 0.003), "peak_suspicion": (0.490, 0.004)}
assert {m: v[0] for m, v in TABLE1.items()} == V.PAPER_TABLE1_REFLECTION


def roles() -> list[str]:
    names = sorted(p.stem for p in (CODE / "kd" / "personas" / "kd_instructions").glob("*.json"))
    assert len(names) == 36 and "default" in names, f"{len(names)} persona files"
    return names


# ── bank ─────────────────────────────────────────────────────────────────

def combine(store: Path, out: Path, min_count: int = MIN_COUNT) -> dict[str, int]:
    """Role vectors from per-question cells (the n-weighted mean, i.e. the mean over every kept reply), written for
    the eligible roles only (default always); returns the kept replies per role."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pt"):  # a derived folder: never keep a role from an earlier combine
        old.unlink()
    counts = {}
    for role_dir in sorted(p for p in store.iterdir() if p.is_dir()):
        total, acc, layers = 0, None, None
        for f in sorted(role_dir.glob("q*.pt")):
            cell = torch.load(f, weights_only=True)
            layers = layers if layers is not None else [int(x) for x in cell["layers"]]
            assert [int(x) for x in cell["layers"]] == layers, f"{f}: layers differ within a role"
            part = cell["vector"].double() * int(cell["n"])
            acc = part if acc is None else acc + part
            total += int(cell["n"])
        counts[role_dir.name] = total
        if total and (total >= min_count or role_dir.name == "default"):
            torch.save({"vector": (acc / total).float(), "layers": layers, "type": "pq_combined",
                        "role": role_dir.name, "n": total}, out / f"{role_dir.name}.pt")
    return counts


def _cells(store: Path) -> int:
    return len(list(store.rglob("q*.pt"))) + len(list(store.rglob("q*.empty")))


def _judge_stats() -> dict:
    """Share of each role's replies the judge scored 3 (fully in role) and 2-3, by role and by group."""
    groups = json.loads(ROLE_GROUPS.read_text(encoding="utf-8"))["groups"]
    per = {}
    for f in sorted(SCORES.glob("*.json")):
        s = [int(v) for v in json.loads(f.read_text(encoding="utf-8")).values()]
        per[f.stem] = {"scored": len(s), "share_3": float(np.mean([v == 3 for v in s])) if s else None,
                       "share_23": float(np.mean([v >= 2 for v in s])) if s else None}
    by_group = {g: {k: float(np.mean([per[r][k] for r in d["roles"] if r in per and per[r][k] is not None]))
                    for k in ("share_3", "share_23")} for g, d in groups.items()}
    return {"by_role": per, "by_group": by_group}


def project(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rows onto the basis rows, uncentred (everything after this is their code)."""
    return (np.asarray(X, np.float64) @ B.T).astype(np.float32)


def load_basis() -> dict[int, np.ndarray]:
    with np.load(BASIS) as z:
        return {L: z[f"basis|L{L}"] for L in LAYERS}


def awq_agreement(basis: dict[int, np.ndarray]) -> dict:
    """The 20 released Transfer rows read in bf16 on NDIF vs their AWQ activations, after the projection: Pearson r
    of each PC coordinate and the cosine of the 3-dim vectors (rows and points as `verdict_score.awq_check`)."""
    rows = [json.loads(line) for line in open(V.ROWS / "qwen_awq_check.jsonl", encoding="utf-8")]
    got = {}
    for f in sorted((V.ACTS / "qwen" / "qwen_awq_check").glob("*.npz")):
        with np.load(f) as z:
            got.update({int(r): p for r, p in zip(z["rows"], z["points"])})
    idx = [r["meta_index"] for r in rows]
    out = {}
    with np.load(S.RELEASED / "qwen3_32b" / "transfer" / "activations_gen.npz") as z:
        for i, L in enumerate(LAYERS):
            ours = project(np.stack([got[r["row_id"]][0, i] for r in rows]), basis[L]).astype(np.float64)
            theirs = project(z[f"layer_{L}"][idx], basis[L]).astype(np.float64)
            cos = (ours * theirs).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1))
            out[f"L{L}"] = {"pc_pearson_r": [float(np.corrcoef(ours[:, k], theirs[:, k])[0, 1]) for k in range(3)],
                            "cos_min": float(cos.min()), "cos_median": float(np.median(cos))}
    return out


def bank() -> None:
    from deception_detection import persona_pcs as pcs

    names = roles()
    report = {"min_count": MIN_COUNT, "cells_expected": len(names) * N_QUESTIONS}
    chosen = None
    for b in ("s3", "s23"):
        if not STORE[b].exists():
            report[b] = "not built"
            continue
        cells = _cells(STORE[b])
        if cells < len(names) * N_QUESTIONS:
            raise SystemExit(f"{STORE[b]}: {cells} of {len(names) * N_QUESTIONS} cells; the store is incomplete")
        counts = combine(STORE[b], CPU_BANK[b])
        short = [r for r in names if r != "default" and counts.get(r, 0) < MIN_COUNT]
        report[b] = {"cells": cells, "kept_replies": {r: counts.get(r, 0) for r in names}, "ineligible": short}
        if chosen is None and not short:
            chosen = b
        if b == "s3" and short and chosen is None and not STORE["s23"].exists():
            BANK_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")
            raise SystemExit(f"s3 leaves {len(short)} role(s) ineligible ({short}): build the s23 store "
                             "(run_chain.ps1 -PqBanks s23), then run `bank` again")
    if chosen is None:
        raise SystemExit(f"no store makes every role eligible: {json.dumps(report)[:400]}")
    out, geometry = {}, {}
    for L in LAYERS:
        geom = pcs.fit_geometry(pcs.load_bank_at_layer(L, CPU_BANK[chosen], model=V.QWEN_MODEL), k=3,
                                deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                                honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
        B = geom.components[:3].astype(np.float64)
        assert np.allclose(B @ B.T, np.eye(3), atol=1e-8), "the PCs are not orthonormal"
        out[f"basis|L{L}"] = B
        geometry[f"L{L}"] = {"explained_variance_ratio": [float(x) for x in geom.explained_variance_ratio[:3]],
                             "pc_cos_with_extended_axis": [float(c @ geom.axis) for c in B]}
    np.savez(BASIS, variant=np.array(chosen), **out)
    report |= {"variant": chosen, "geometry": geometry, "judge": _judge_stats(),
               "awq_vs_bf16_projected": awq_agreement({L: out[f"basis|L{L}"] for L in LAYERS}),
               "basis_sha256": FZ.sha256(BASIS)}
    BANK_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("variant", "geometry", "awq_vs_bf16_projected")}, indent=1))
    print(json.dumps(report["judge"]["by_group"], indent=1))


# ── their pipeline on projected files ────────────────────────────────────

def prepare(run_dir: Path, basis: dict[int, np.ndarray]) -> list[int]:
    """`verdict_score.prepare_rose_run` (their clone, their Core activations, our four phase files), then every
    `layer_L` of those five files replaced by its projection; the metadata stays as written."""
    labels = VS.load_labels(V.ROWS / "labels.json")
    rids = VS.prepare_rose_run(run_dir, V.ACTS, labels)
    acts = run_dir / "data" / "activations" / "qwen3_32b"
    for f in [acts / "core" / "activations_gen.npz"] + [acts / "verdict_decisions" / f"activations_{ph}.npz"
                                                        for ph in V.QWEN_PHASES]:
        with np.load(f) as z:
            arrs = {k: z[k] for k in z.files}
        assert sorted(arrs) == [f"layer_{L}" for L in LAYERS], f"{f}: {sorted(arrs)}"
        np.savez(f, **{k: project(v, basis[int(k.split("_")[1])]) for k, v in arrs.items()})
        del arrs
    return rids


def read_summary(run_dir: Path) -> dict:
    """Their summary CSV: group -> method -> [mean, standard error]."""
    out = {}
    with open(run_dir / SUMMARY, newline="") as f:
        for r in csv.DictReader(f):
            out[GROUP_OF[r["group"]]] = {m: [float(r[f"{m}_mean"]), float(r[f"{m}_std"])] for m in METHODS}
    return out


def _relabel(mod, label: dict) -> None:
    """Only `build_verdict_groups`' episode labels replaced, as `verdict_score.their_with_label` does."""
    original = mod.build_verdict_groups
    by_rid = {f"{rep:03d}__{e}": y for (rep, e), y in label.items()}

    def relabelled(meta, npz, layer):
        groups = original(meta, npz, layer)
        for rid, g in groups.items():
            g["mode"] = "collusion" if by_rid[rid] else "control"
        return groups

    mod.build_verdict_groups = relabelled


def their_with_label_sem(run_dir: Path, label: dict) -> dict:
    """`verdict_score.their_with_label` with their standard error as well: group -> method -> [mean, se]."""
    mod = VS._their_module(run_dir)
    _relabel(mod, label)
    core_meta, core_npz = mod.load_data(mod.CORE_DIR)
    verdict_data = {ph: mod.load_verdict_phase(ph) for ph in mod.VERDICT_PHASES}
    raws = [mod.compute_all_results(core_meta, core_npz, verdict_data, seed=s)[1] for s in range(mod.N_SEEDS)]
    out = {}
    for g in ["Held-out"] + list(mod.VERDICT_PHASES):
        for m in METHODS:
            arr = np.array([v for raw in raws for v in raw[g][m]], float)
            n = int(np.sum(~np.isnan(arr)))
            out.setdefault(g, {})[m] = [float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(n))]
    return out


def random_subspace(i: int, L: int, hidden: int = 5120) -> np.ndarray:
    """Null subspace i at layer L: three orthonormal rows from a seeded Gaussian."""
    q, _ = np.linalg.qr(np.random.default_rng([S.RANDOM_SEED, 60_000 + i, L]).standard_normal((hidden, 3)))
    return q.T


def null_band(n: int = N_NULL) -> dict:
    """Seed-0 layer-mean AUROC of every method and phase for the persona basis and for `n` random 3-dim subspaces,
    from the full-width files of the unprojected run, through their `compute_all_results` (Core CV skipped)."""
    acts = VS.ROSE_RUN / "data" / "activations" / "qwen3_32b"
    full = {}
    for name, f in [("core", acts / "core" / "activations_gen.npz")] + [
            (ph, acts / "verdict_decisions" / f"activations_{ph}.npz") for ph in V.QWEN_PHASES]:
        with np.load(f) as z:
            full[name] = {L: z[f"layer_{L}"] for L in LAYERS}
    labels = VS.load_labels(V.ROWS / "labels.json")
    mods = {"rose": VS._their_module(VS.ROSE_RUN), "shi": VS._their_module(VS.ROSE_RUN)}
    _relabel(mods["shi"], labels["shi"])
    for mod in mods.values():
        mod.held_out_cv_core = lambda *a, _m=mod, **k: {m: float("nan") for m in _m.METHOD_KEYS}
    core_meta, _ = mods["rose"].load_data(mods["rose"].CORE_DIR)
    vmeta = {ph: mods["rose"].load_verdict_phase(ph)[0] for ph in V.QWEN_PHASES}

    def layer_means(basis: dict[int, np.ndarray]) -> dict:
        core = {f"layer_{L}": project(full["core"][L], basis[L]) for L in LAYERS}
        vd = {ph: (vmeta[ph], {f"layer_{L}": project(full[ph][L], basis[L]) for L in LAYERS}) for ph in V.QWEN_PHASES}
        out = {}
        for lab, mod in mods.items():
            raw = mod.compute_all_results(core_meta, core, vd, seed=0)[1]
            out[lab] = {ph: {m: float(np.mean(raw[ph][m])) for m in METHODS} for ph in V.QWEN_PHASES}
        return out

    observed = layer_means(load_basis())
    nulls = [layer_means({L: random_subspace(i, L) for L in LAYERS}) for i in range(n)]
    band = {lab: {ph: {m: {"observed_seed0": observed[lab][ph][m],
                           "band97.5": float(np.percentile([x[lab][ph][m] for x in nulls], 97.5)),
                           "median": float(np.median([x[lab][ph][m] for x in nulls])),
                           "p": float((1 + sum(x[lab][ph][m] >= observed[lab][ph][m] for x in nulls)) / (1 + n))}
                       for m in METHODS} for ph in V.QWEN_PHASES} for lab in mods}
    return {"n": n, "note": "seed 0; mean over layers 26-30; random orthonormal 3-dim subspaces, rng [RANDOM_SEED, "
                            "60000+i, L]; their Core CV skipped", "bands": band}


def identity() -> None:
    """GB5: the identity as the basis reproduces the unprojected summary CSV byte for byte."""
    eye = np.eye(5120)
    prepare(RUN_ID, {L: eye for L in LAYERS})
    del eye
    VS.run_their_script(RUN_ID)
    same = (RUN_ID / SUMMARY).read_bytes() == (VS.ROSE_RUN / SUMMARY).read_bytes()
    raw = Path(str(SUMMARY).replace("_summary", ""))
    same_raw = (RUN_ID / raw).read_bytes() == (VS.ROSE_RUN / raw).read_bytes()
    res = {"pass": bool(same and same_raw), "summary_identical": same, "raw_csv_identical": same_raw,
           "summary_sha256": FZ.sha256(RUN_ID / SUMMARY)}
    GB5_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    shutil.rmtree(RUN_ID)
    print(json.dumps(res, indent=1))


# ── freeze ───────────────────────────────────────────────────────────────

def _manifest(store: Path) -> str:
    lines = [f"{p.relative_to(store).as_posix()} {FZ.sha256(p)}" for p in sorted(store.rglob("q*.*"))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _chain_diff() -> tuple[list[str], str]:
    def walk(d: filecmp.dircmp, prefix: str = "") -> list[str]:
        out = [prefix + f for f in d.diff_files + d.left_only + d.right_only]
        for sub, dd in d.subdirs.items():
            out += walk(dd, prefix + sub + "/")
        return out

    files = walk(filecmp.dircmp(QBANK_CODE, CODE, ignore=["__pycache__"]))
    diff = "".join(difflib.unified_diff((QBANK_CODE / "run_chain.ps1").read_text(encoding="utf-8-sig").splitlines(True),
                                        (CODE / "run_chain.ps1").read_text(encoding="utf-8-sig").splitlines(True)))
    return files, diff


def declare() -> None:
    if not json.loads(GB5_JSON.read_text())["pass"]:
        raise SystemExit("GB5 did not pass")
    bank_report = json.loads(BANK_JSON.read_text(encoding="utf-8"))
    if bank_report.get("basis_sha256") != FZ.sha256(BASIS):
        raise SystemExit("the basis changed since `bank`")
    files, diff = _chain_diff()
    variant = bank_report["variant"]
    lines = [f"{MARK} (declared before any projected score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_d4_pipeline.py sha256 {FZ.sha256(Path(__file__))}",
             f"qwen3_d4_basis.npz sha256 {FZ.sha256(BASIS)}",
             f"bank cells ({STORE[variant].name}) manifest sha256 {_manifest(STORE[variant])}",
             f"bank variant {variant}; s3 ineligible roles: {bank_report['s3']['ineligible'] if isinstance(bank_report.get('s3'), dict) else 'not built'}",
             f"chain copy {CODE} differs from {QBANK_CODE} only in {files}; run_chain.ps1 diff sha256 "
             f"{hashlib.sha256(diff.encode()).hexdigest()}",
             f"their code {S.GITHUB_COMMIT}; verdict_score.py sha256 {FZ.sha256(Path(VS.__file__))} (freeze V)",
             f"GB5 identity check: {GB5_JSON.read_text().strip()[:200]}",
             "design: project every Core and Verdict activation onto the per-layer top-3 extended persona PCs "
             "(uncentred), their script unchanged (Rose's label, 10 seeds, layers 26-30); their functions with Shi's "
             "label; null = 100 random orthonormal 3-dim subspaces per layer (rng [RANDOM_SEED, 60000+i, L]), their "
             "compute_all_results at seed 0 with Core CV skipped, band = 97.5th percentile, p = (1 + #null >= "
             "observed) / 101; descriptive, no Holm",
             ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _check_declared() -> None:
    text = (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig")
    if MARK not in text:
        raise SystemExit("run `declare` first")
    block = text.split(MARK)[-1].split("\n== ")[0]
    want = {line.split(" sha256 ")[0]: line.split(" sha256 ")[1].split()[0] for line in block.splitlines()
            if line.startswith(("qwen_d4_pipeline.py", "qwen3_d4_basis.npz"))}
    cur = {"qwen_d4_pipeline.py": FZ.sha256(Path(__file__)), "qwen3_d4_basis.npz": FZ.sha256(BASIS)}
    if want != cur:
        raise SystemExit(f"changed since the declaration: {want} vs {cur}")


def run() -> None:
    _check_declared()
    VS.check_freeze()
    labels = VS.load_labels(V.ROWS / "labels.json")
    rids = prepare(RUN_D4, load_basis())
    res = {"note": "POST HOC, descriptive: D4's persona projection (top-3 extended PCs per layer) before their "
                   "pipeline, Qwen3-32B reads of Shi et al.'s Gemma-4-31B Verdict runs.", "rows": len(rids),
           "variant": json.loads(BANK_JSON.read_text(encoding="utf-8"))["variant"],
           "projected_their_script": VS.run_their_script(RUN_D4),
           "projected_summary_rose": read_summary(RUN_D4), "unprojected_summary_rose": read_summary(VS.ROSE_RUN),
           "projected_shi": their_with_label_sem(RUN_D4, labels["shi"]),
           "unprojected_shi": their_with_label_sem(VS.ROSE_RUN, labels["shi"]), "table1": TABLE1}
    res["null"] = null_band()
    raw = res["projected_their_script"]["csv"]
    seed0 = {}
    with open(raw, newline="") as f:
        for r in csv.DictReader(f):
            if r["record_type"] == "raw" and r["seed"] == "0" and r["group"] in V.QWEN_PHASES:
                seed0.setdefault((r["group"], r["method"]), []).append(float(r["auroc"]))
    # the in-process seed-0 values of the persona basis equal their script's seed-0 rows
    res["check_null_observed_equals_their_script"] = bool(all(
        np.isclose(np.mean(v), res["null"]["bands"]["rose"][ph][m]["observed_seed0"]) for (ph, m), v in seed0.items()))
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("projected_summary_rose", "check_null_observed_equals_their_script")},
                     indent=1))


# ── report ───────────────────────────────────────────────────────────────

def _ms(x) -> str:
    return f"{x[0]:.3f} ± {x[1]:.3f}"


def report() -> None:
    r = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    bk = json.loads(BANK_JSON.read_text(encoding="utf-8"))
    nb = r["null"]["bands"]
    out = ["# D4's persona projection inside their pipeline (Qwen3-32B, Verdict)", "",
           "Post hoc and descriptive. Every Core and Verdict activation is projected onto D4's persona subspace (the top "
           f"3 extended persona PCs of a Qwen3-32B persona bank, {bk['variant']}, fitted at each of layers 26-30), then "
           "their `reproduce_LR_core_to_verdict.py` runs unchanged: their probe, five methods, Core training, 10 seeds. "
           "± is their statistic (standard error over 10 seeds x 5 layers). Declared in `E:/narcbench/logs/freeze.txt` "
           "before any projected score.", "",
           "## Reflection phase, Rose's label (their Table 1's setting)", "",
           "| Method | Their Table 1 (their runs) | Their pipeline, our reads | + D4 persona projection | "
           "Random 3-dim subspaces: band (p) |", "|---|---|---|---|---|"]
    for m in METHODS:
        b = nb["rose"]["reflection"][m]
        out.append(f"| {NAMES[m]} | {_ms(TABLE1[m])} | {_ms(r['unprojected_summary_rose']['reflection'][m])} | "
                   f"{_ms(r['projected_summary_rose']['reflection'][m])} | {b['band97.5']:.3f} (p {b['p']:.3f}) |")
    out += ["", "Their Table 1 comes from their own Qwen3-32B agent runs (not released); ours are Shi et al.'s "
            "Gemma-4-31B runs read by Qwen3-32B with a compact context, so the matched comparison is the projected column "
            "against the unprojected one. The band is the 97.5th percentile of 100 random orthonormal 3-dim subspaces "
            "through the same pipeline (seed 0; p against the persona subspace's seed-0 value).", "",
            "## Reflection phase, Shi's label (both agents ACCEPT)", "",
            "| Method | Their pipeline, our reads | + D4 persona projection | Random 3-dim subspaces: band (p) |",
            "|---|---|---|---|"]
    for m in METHODS:
        b = nb["shi"]["reflection"][m]
        out.append(f"| {NAMES[m]} | {_ms(r['unprojected_shi']['reflection'][m])} | "
                   f"{_ms(r['projected_shi']['reflection'][m])} | {b['band97.5']:.3f} (p {b['p']:.3f}) |")
    out += ["", "## Every phase (projected; Rose's label, Shi's in brackets)", "",
            "| Phase | " + " | ".join(NAMES[m] for m in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for g in ["Held-out"] + list(V.QWEN_PHASES):
        out.append(f"| {'Core held-out domain' if g == 'Held-out' else g} | " + " | ".join(
            f"{r['projected_summary_rose'][g][m][0]:.2f} ({r['projected_shi'][g][m][0]:.2f})" for m in METHODS) + " |")
    out += ["", "## Seed-0 value against the random 3-dim band, every phase (value / band, p)", "",
            "| Label | Phase | " + " | ".join(NAMES[m] for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for lab in ("rose", "shi"):
        for ph in V.QWEN_PHASES:
            out.append(f"| {lab.capitalize()} | {ph} | " + " | ".join(
                f"{nb[lab][ph][m]['observed_seed0']:.2f} / {nb[lab][ph][m]['band97.5']:.2f} ({nb[lab][ph][m]['p']:.2f})"
                for m in METHODS) + " |")
    out += ["", "p = (1 + random subspaces at or above the persona subspace) / 101, so 0.01 means above all 100; 40 cells, "
            "post hoc, unadjusted."]
    geo = bk["geometry"]
    out += ["", "The `decision` position is the accept/reject token itself and carries Shi's label.", "",
            "## The Qwen3-32B persona bank", "",
            f"- Variant {bk['variant']} (judge score 3 unless a role falls under {bk['min_count']} kept replies); judge "
            "share fully in role by group: " + ", ".join(f"{g} {v['share_3']:.2f}" for g, v in bk["judge"]["by_group"].items()) + ".",
            "- Variance explained by the 3 PCs per layer: " + "; ".join(
                f"{L} " + "/".join(f"{x:.2f}" for x in v["explained_variance_ratio"]) for L, v in geo.items()) + ".",
            "- AWQ (their Core) vs bf16 (the bank and our reads), projected coordinates of 20 Transfer rows, Pearson r of "
            "PC1-3: " + "; ".join(f"{L} " + "/".join(f"{x:.3f}" for x in v["pc_pearson_r"])
                                  for L, v in bk["awq_vs_bf16_projected"].items()) + ".",
            f"- Check: the identity in place of the basis reproduces the unprojected CSVs byte for byte (GB5); the "
            f"null's seed-0 persona values equal their script's seed-0 rows: {r['check_null_observed_equals_their_script']}.", ""]
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(out)} lines)")


if __name__ == "__main__":
    fire.Fire({"bank": bank, "identity": identity, "declare": declare, "run": run, "report": report})
