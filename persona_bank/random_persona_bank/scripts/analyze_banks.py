"""CPU stage of the random persona bank, per model and per bank (owner 2026-10-10: all three banks, every model, one
folder per bank): builds R5-R9, R5 + K / R9 + K and the kd-vs-random comparison into

    random_persona_bank/results/<Model>/bank_<bank>/
        r5 .. r9/        R*_<bank>.pt + .json   split 0 (r_contrasts.py)
        r_null/          R_null_<bank>.csv      100 splits: norm, norm / role spread, cos with the assistant direction
        r5k/, r9k/       R*K_<bank>.pt + .json  split 0 (rk.py)
        rk_null/         RK_null_<bank>.csv     100 splits: spread ratio, cos(R, K), cos(R + K, K)
        compare_kd_vs_random_<bank>.csv         the model's kd D's / D + K against those nulls

Banks: `all` (no judge filter; all 35 roles; the main comparison, owner option A), `s23` and `s3` (judge-filtered: a role
with fewer than 20 kept replies has no vector, and is dropped from every split's groups -- r_contrasts.py
--allow_missing; the CSVs record each split's group sizes; a split whose group empties is skipped).

The kd side is each model's own kd bank of the same kind (xmodel-runs; Qwen's are the `_v2` banks; Qwen has no kd s3
D's: its kd s3 bank lacks 8 of the roles they need). K and the Alpaca store are the model's own, identical on both sides.

Comparison measures, at the report layers, as in compare tables before: D size / the kd bank's role spread v the R null
of the same quantity on the random bank; |cos(D, assistant)| v the null's; D + K spread ratio and cos(D + K, K) v the
R + K null. `pct` = the kd value's percentile among the random splits.  Geometry only -- nothing here measures detection.

    python random_persona_bank/scripts/analyze_banks.py --models Llama-3.3,Qwen-2.5 [--banks all,s23,s3] [--compare_only]
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
XR = Path(os.path.expandvars(r"%USERPROFILE%\xmodel-runs"))

# key -> random run folder, NDIF repo id, kd folder, kd tag per bank, K file, Alpaca store, report layers
MODELS = {
    "Llama-3.3": dict(folder="Llama-3.3-70B-Instruct", repo="meta-llama/Llama-3.3-70B-Instruct",
                      kd=XR / "Llama-3.3-70B-Instruct", tags={"all": "all", "s23": "s23", "s3": "s3"},
                      k=RES / "Llama-3.3-70B-Instruct/K/llama-70b-3.3/first/K.npz",
                      alpaca=RES / "Llama-3.3-70B-Instruct/eval/alpaca", layers=(20, 40)),
    "Qwen-2.5": dict(folder="Qwen2.5-72B-Instruct", repo="Qwen/Qwen2.5-72B-Instruct",
                     kd=XR / "Qwen2.5-72B-Instruct", tags={"all": "all_v2", "s23": "s23_v2", "s3": "s3_v2"},
                     k=XR / "Qwen2.5-72B-Instruct/K/qwen-72b/first/K.npz",
                     alpaca=XR / "Qwen2.5-72B-Instruct/eval/alpaca", layers=(20, 40)),
    "Mistral-2501": dict(folder="Mistral-Small-24B-Instruct-2501", repo="mistralai/Mistral-Small-24B-Instruct-2501",
                         kd=XR / "Mistral-Small-24B-Instruct-2501", tags={"all": "all", "s23": "s23", "s3": "s3"},
                         k=XR / "Mistral-Small-24B-Instruct-2501/K/mistral-small-24b-2501/first/K.npz",
                         alpaca=XR / "Mistral-Small-24B-Instruct-2501/eval/alpaca", layers=(10, 20)),
    "Mistral-3.1": dict(folder="Mistral-Small-3.1-24B-Instruct-2503", repo="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        kd=XR / "Mistral-Small-3.1-24B-Instruct-2503", tags={"all": "all", "s23": "s23", "s3": "s3"},
                        k=XR / "Mistral-Small-3.1-24B-Instruct-2503/K/mistral-small-3.1-24b/first/K.npz",
                        alpaca=XR / "Mistral-Small-3.1-24B-Instruct-2503/eval/alpaca", layers=(10, 20)),
    "Gemma-3": dict(folder="gemma-3-27b-it", repo="google/gemma-3-27b-it",
                    kd=XR / "gemma-3-27b-it", tags={"all": "all", "s23": "s23", "s3": "s3"},
                    k=XR / "gemma-3-27b-it/K/gemma-3-27b/first/K.npz",
                    alpaca=XR / "gemma-3-27b-it/eval_fp32/alpaca", layers=(20, 31, 40)),
}


def run(args):
    print("  $", " ".join(str(x) for x in args[1:]))
    r = subprocess.run([str(x) for x in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip().splitlines()
    for line in out[-6:]:
        print("    " + line[:220])
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {args[1]}")


def pct(x, null):
    return 100.0 * np.mean(np.asarray(null) < x)


def bank_geometry(vec_dir: Path, layers_wanted):
    vec = {p.stem: torch.load(p, weights_only=False)["vector"].double().numpy() for p in sorted(vec_dir.glob("*.pt"))}
    first = torch.load(next(vec_dir.glob("*.pt")), weights_only=False)
    layers = list(first.get("layers", range(next(iter(vec.values())).shape[0])))
    allv = np.stack(list(vec.values()))
    roles = [r for r in vec if r != "default"]
    out = {}
    for l in layers_wanted:
        i = layers.index(l)
        x = allv[:, i]
        a = vec["default"][i] - np.stack([vec[r][i] for r in roles]).mean(0)
        out[l] = (float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean())), a / np.linalg.norm(a))
    return out


def compare(key, bank, bdir: Path):
    cfg = MODELS[key]
    kdtag = cfg["tags"][bank]
    kd = cfg["kd"]
    kd_vec = kd / "persona" / f"vectors_{kdtag}"
    rows = []
    rn = list(csv.DictReader(open(bdir / "r_null" / f"R_null_{bank}.csv", encoding="utf-8")))
    rkp = bdir / "rk_null" / f"RK_null_{bank}.csv"
    rk = list(csv.DictReader(open(rkp, encoding="utf-8"))) if rkp.exists() else []
    kz = np.load(cfg["k"])
    have_d = kd_vec.exists() and any((kd / f"d{c}" / f"D{c}_{kdtag}.pt").exists() for c in "56789")
    geo = bank_geometry(kd_vec, cfg["layers"]) if have_d else None

    def add(l, measure, kdv, null, n_split):
        rows.append({"model": key, "bank": bank, "kd_tag": kdtag, "layer": l, "measure": measure,
                     "kd": None if kdv is None else round(kdv, 4),
                     "null_p2.5": round(np.percentile(null, 2.5), 4), "null_p50": round(np.percentile(null, 50), 4),
                     "null_p97.5": round(np.percentile(null, 97.5), 4),
                     "pct": None if kdv is None else round(pct(kdv, null), 1), "n_splits": n_split})

    for c in "56789":
        dfile = kd / f"d{c}" / f"D{c}_{kdtag}.pt"
        d = torch.load(dfile, weights_only=False) if dfile.exists() else None
        for l in cfg["layers"]:
            sel = [r for r in rn if r["contrast"] == f"R{c}" and int(r["layer"]) == l]
            if not sel:
                continue
            null_rel = [float(r["rel_norm"]) for r in sel]
            null_ca = [abs(float(r["cos_assistant"])) for r in sel]
            rel = ca = None
            if d is not None:
                i = list(d["layers"]).index(l)
                spread, au = geo[l]
                rel = float(d["norm"][i]) / spread
                ca = abs(float(d["unit"][i].double().numpy() @ au))
            add(l, f"D{c} size / role spread", rel, null_rel, len(sel))
            add(l, f"D{c} |cos assistant|", ca, null_ca, len(sel))
    for c in "59":
        f = kd / f"d{c}k" / f"D{c}K_{kdtag}.pt"
        dk = torch.load(f, weights_only=False) if f.exists() else None
        for l in cfg["layers"]:
            sel = [r for r in rk if r["contrast"] == f"R{c}" and int(r["layer"]) == l]
            if not sel:
                continue
            ratio = cwk = None
            if dk is not None and l in list(dk["layers"]):
                i = list(dk["layers"]).index(l)
                ratio = float(dk[f"sd_d{c}"][i] / dk["sd_k"][i])
                kv = kz[f"L{l}"].astype(np.float64)
                cwk = float(dk["unit"][i].double().numpy() @ (kv / np.linalg.norm(kv)))
            add(l, f"D{c}+K spread ratio D:K", ratio, [float(r["ratio"]) for r in sel], len(sel))
            add(l, f"D{c}+K cos(D+K, K)", cwk, [float(r["cos_rk_k"]) for r in sel], len(sel))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--banks", default="all,s23,s3")
    ap.add_argument("--compare_only", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    all_rows = {}
    for key in [m.strip() for m in a.models.split(",") if m.strip()]:
        cfg = MODELS[key]
        rroot = RES / cfg["folder"]
        for bank in [b.strip() for b in a.banks.split(",") if b.strip()]:
            vecs = rroot / "persona" / f"vectors_{bank}"
            bdir = rroot / f"bank_{bank}"
            print(f"=== {key} / {bank}: {len(list(vecs.glob('*.pt')))} vectors")
            if not a.compare_only:
                # resumable per step (2026-10-10, after a low-memory stop): a step whose final CSV exists is done --
                # each script writes its CSV last, and refuses to overwrite anything it would write
                if (bdir / "r_null" / f"R_null_{bank}.csv").exists():
                    print("  R5-R9: done already")
                else:
                    extra = [] if bank == "all" else ["--allow_missing"]
                    run([py, "-I", HERE / "r_contrasts.py", "--vectors", vecs, "--out", bdir, "--tag", bank,
                         "--model", cfg["repo"]] + extra)
                if (bdir / "rk_null" / f"RK_null_{bank}.csv").exists():
                    print("  R + K: done already")
                else:
                    partial = [p for p in (bdir / "r5k", bdir / "r9k") if p.exists() and any(p.iterdir())]
                    if partial:
                        sys.exit(f"{partial[0]} holds files from an unfinished R + K run; remove them, then re-run")
                    run([py, "-I", HERE / "rk.py", "--alpaca", cfg["alpaca"], "--k", cfg["k"], "--rdir", bdir,
                         "--tag", bank, "--vectors", vecs])
            rows = compare(key, bank, bdir)
            out = bdir / f"compare_kd_vs_random_{bank}.csv"
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            all_rows.setdefault(bank, []).extend(rows)
            print(f"  compare: {len(rows)} rows -> {out.relative_to(RES.parent)}")
    for bank, rows in all_rows.items():
        out = RES / f"compare_kd_vs_random_{bank}.csv"      # every model analysed so far, one table per bank
        old = list(csv.DictReader(open(out, encoding="utf-8"))) if out.exists() else []
        keep = [r for r in old if r["model"] not in {x["model"] for x in rows}]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep + rows)
        print(f"combined -> {out.relative_to(RES.parent)} ({len(keep) + len(rows)} rows)")


if __name__ == "__main__":
    main()
