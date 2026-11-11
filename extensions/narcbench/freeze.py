"""Freeze the Arm B inputs: detector vectors as data (G5), the Alpaca parity rows, the stage run lists,
and the freeze log the reader checks.

`directions` writes `E:/narcbench/frozen/directions_llama33.npz` from the thesis's own code paths
(`recall_denial.Inputs("llama")`, `standardised_sum`, the RepE `detector.pt`, `random_bank`), then G5
checks the frozen vectors reproduce the thesis's layer-40 AUROCs on the stored Liars' Bench reads.
Scoring later loads only this file, so the other session's uncommitted edits in the shared tree cannot
move the vectors after the freeze. `alpaca` exports the stored Alpaca rows the reader's smoke test
re-reads. `stages` writes the run lists. `freeze1` / `freeze2` append to `E:/narcbench/logs/freeze.txt`.
"""

import hashlib
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from extensions.narcbench import rows as R
from extensions.narcbench import spec as S

FROZEN_NPZ = S.FROZEN / "directions_llama33.npz"
EXPECTED_L40 = {  # thesis numbers (numbers.tex th-k-ninek-*-L40, results_table.csv), CG and HP-KR
    "D9": (0.9123, 0.9124), "K": (0.6406, 0.8654), "D9+K": (0.8646, 0.9087), "D5+K": (0.8126, 0.8541),
    "RepE": (0.9723, 0.9358)}
THESIS_MODULES = ("recall_denial.py", "random_bank.py", "persona_ood.py", "layer_sweep.py", "lb_models.py",
                  "k_robustness.py", "k_transfer.py")


def _auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _repe(inp, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The arrays of the probe `Inputs.probe(layer)` scores with (same run folder)."""
    with open(inp.spec.probe_run(layer) / "detector.pt", "rb") as f:
        det = pickle.load(f)
    return tuple(det[k].double().numpy().ravel() for k in ("scaler_mean", "scaler_scale", "directions"))


def directions(n_random: int = S.N_RANDOM) -> None:
    from experiments.external import random_bank as RB
    from experiments.knowledge import recall_denial as rd

    inp = rd.Inputs("llama")
    assert inp.thesis, "not the thesis bank"
    out, g5 = {}, {}
    splits, _ = RB.read_splits()
    for L in S.THESIS_LAYERS:
        bank = inp.bank_at(L)
        d9, d5 = bank.axis_unit("extended"), bank.axis_unit("deception")
        k = inp.k(L)
        pooled = inp.pooled(L)
        alp = pooled[rd.CONTROL].z.astype(np.float64)
        mean, scale, direction = _repe(inp, L)
        W = {"D9": d9, "D5": d5, "K": k, "D9+K": rd.standardised_sum(d9, k, alp),
             "D5+K": rd.standardised_sum(d5, k, alp)}
        for name, w in W.items():
            out[f"L{L}|{name}"] = np.asarray(w, np.float64)
        out[f"L{L}|RepE_mean"], out[f"L{L}|RepE_scale"], out[f"L{L}|RepE_dir"] = mean, scale, direction
        # a check that the arrays are the probe the thesis scored with
        probe = inp.probe(L)
        some = next(iter(v.z for n, v in pooled.items() if n != rd.CONTROL))[:20]
        assert np.allclose(probe(some), ((some - mean) / scale) @ direction), "RepE arrays differ from the probe"
        rng = np.random.default_rng([S.RANDOM_SEED, L])
        Wr = rng.standard_normal((n_random, d9.shape[0]))
        out[f"L{L}|random"] = (Wr / np.linalg.norm(Wr, axis=1, keepdims=True)).astype(np.float32)
        names, X, _ = RB.load_roles(RB.vectors_dir("s3"), layers=[L])
        Rw, keys = RB.random_directions(names, X[:, 0], splits, contrasts=("9",))
        Rw = Rw / np.linalg.norm(Rw, axis=0, keepdims=True)
        repe_y, repe_z = pooled["repe_honesty__plain"].y.astype(int), pooled["repe_honesty__plain"].z
        sign = np.where(RB.auroc_columns(repe_y, repe_z @ Rw) >= 0.5, 1.0, -1.0)
        Rw = Rw * sign
        out[f"L{L}|R9"] = Rw.T.astype(np.float64)
        out[f"L{L}|R9+K"] = np.stack([rd.standardised_sum(Rw[:, j], k, alp) for j in range(Rw.shape[1])])
        if L == S.HEADLINE_LAYER:
            for name in EXPECTED_L40:
                got = []
                for short in ("CG", "HP-KR"):
                    full = next(n for n, s in rd.SHORT.items() if s == short)
                    y, zz = pooled[full].y.astype(int), pooled[full].z
                    score = ((zz - mean) / scale) @ direction if name == "RepE" else zz @ out[f"L{L}|{name}"]
                    got.append(round(_auroc(y, score), 4))
                g5[name] = {"got": got, "expected": list(EXPECTED_L40[name]),
                            "ok": all(abs(a - b) <= 1e-4 for a, b in zip(got, EXPECTED_L40[name]))}
        del pooled
        print(f"layer {L} frozen", flush=True)
    S.FROZEN.mkdir(parents=True, exist_ok=True)
    np.savez(FROZEN_NPZ, **out)
    ok = all(v["ok"] for v in g5.values())
    (S.LOGS / "g5_directions.json").write_text(json.dumps({"pass": ok, "L40": g5, "keys": sorted(out)}, indent=1))
    print(json.dumps(g5, indent=1))
    print(f"G5 {'PASSED' if ok else 'FAILED'}; {FROZEN_NPZ} sha256 {sha256(FROZEN_NPZ)}")
    if not ok:
        raise SystemExit(1)


def alpaca(n: int = 2) -> None:
    """The first `n` stored Alpaca control rows, tokenised exactly as the thesis pooled them, with their
    stored layer-20/36/40/44 means, for the reader's parity check."""
    from experiments.persona import layer_sweep as LS
    from experiments.knowledge import recall_denial as rd
    from deception_detection.tokenized_data import TokenizedDataset

    cfg, experiment = LS._experiment()
    control = next(d for d in experiment.control_datasets if d.name == rd.CONTROL)
    kw = dict(detect_all=cfg.detect_all_tokens, detect_only_start_of_turn=cfg.detect_only_start_of_turn,
              detect_only_last_token=cfg.detect_only_last_token)
    full = TokenizedDataset.from_dataset(control, experiment.tokenizer, **kw)
    kept = [i for i in range(len(control)) if bool(full.detection_mask[i].any())][:n]
    toks = TokenizedDataset.from_dialogue_list([control.dialogues[i] for i in kept], experiment.tokenizer,
                                               padding=control.padding, **kw)
    arrays = {"kept": np.array(kept), "layers": np.array(S.THESIS_LAYERS)}
    a, b = [], []
    for k in range(len(kept)):
        real = toks.attention_mask[k].bool()
        ids = toks.tokens[k][real].numpy().astype(np.int64)
        det = toks.detection_mask[k][real].numpy().astype(bool)
        idx = np.flatnonzero(det)
        assert len(idx) and (np.diff(idx) == 1).all(), "detection span is not contiguous"
        arrays[f"ids_{k}"] = ids
        a.append(int(idx[0]))
        b.append(int(idx[-1]) + 1)
    arrays["a"], arrays["b"] = np.array(a), np.array(b)
    stored = []
    for L in S.THESIS_LAYERS:
        with np.load(LS.pooled_dir(L) / f"{rd.CONTROL}.npz", allow_pickle=False) as z:
            stored.append(z["pooled"][: len(kept)].astype(np.float32))
    arrays["stored"] = np.stack(stored, axis=1)  # (rows, layers, H)
    S.READER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(S.READER_DIR / "alpaca_parity.npz", **arrays)
    print({"kept": kept, "lengths": [len(arrays[f"ids_{k}"]) for k in range(len(kept))], "a": a, "b": b})


def stages() -> None:
    """Run lists for the reader: Core development runs (released activations) and the extra Core runs."""
    dev = sorted({m["run_id"] for m in R.released_meta("core", "qwen3_32b")})
    all_core = sorted(R.load_runs("core"))
    extra = sorted(set(all_core) - set(dev))
    for name, runs in (("core_dev", dev), ("core_extra", extra)):
        (S.ROWS / f"runs_{name}.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
        print(name, len(runs))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(S.DD), *args], capture_output=True, text=True).stdout.strip()


def freeze1() -> None:
    """Freeze 1: everything Arm B depends on, before any scored read."""
    g3 = json.loads((S.RESULTS / "armA_text_baselines.json").read_text())["G3"]
    g5 = json.loads((S.LOGS / "g5_directions.json").read_text())
    if not g5["pass"]:
        raise SystemExit("G5 did not pass")
    manifest = json.loads((S.ROWS / "cond" / "manifest.json").read_text())
    pkg = Path(__file__).parent
    lines = [f"== FREEZE 1  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"spec.py sha256 {sha256(pkg / 'spec.py')}",
             f"directions_llama33.npz sha256 {sha256(FROZEN_NPZ)}",
             f"reader narc_read.py (deployed) sha256 {sha256(S.READER_DIR / 'narc_read.py')}",
             f"reader source sha256 {sha256(pkg / 'reader' / 'narc_read.py')}"]
    lines += [f"package {p.name} sha256 {sha256(p)}" for p in sorted(pkg.glob('*.py'))]
    lines += [f"condition {name} sha256 {v['sha256']} rows {v['rows']} tokens {v['tokens']}"
              for name, v in sorted(manifest.items())]
    lines += [f"runs {p.name} sha256 {sha256(p)}" for p in sorted(S.ROWS.glob('runs_*.json'))]
    scripts = S.DD / "deception_detection" / "scripts"
    lines += [f"thesis module {m} sha256 {sha256(scripts / m)}" for m in THESIS_MODULES]
    lines.append(f"git HEAD {_git('rev-parse', 'HEAD')} branch {_git('rev-parse', '--abbrev-ref', 'HEAD')}")
    diff = _git("diff", "--stat", "--", *[f"deception_detection/scripts/{m}" for m in THESIS_MODULES])
    lines.append("uncommitted diff in imported thesis modules: " + (diff.replace("\n", " | ") or "none"))
    lines.append("G3 ceiling tiers: " + json.dumps({t: (round(v["best"], 4), v["by"], v["ceiling"]) for t, v in g3.items()}))
    lines.append("G5 frozen vectors reproduce the thesis L40 numbers: " + json.dumps({k: v["got"] for k, v in g5["L40"].items()}))
    lines.append("")
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"directions": directions, "alpaca": alpaca, "stages": stages, "freeze1": freeze1})
