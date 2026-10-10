"""R5 + K and R9 + K: the random-split nulls for D5 + K and D9 + K, for one model and one random bank.

The sum is xmodel/d9k.py's (zami's recall_denial.standardised_sum, recall_denial.py:428-431), unchanged:

    R + K = R_unit / sd(Alpaca @ R_unit) + K / sd(Alpaca @ K)          (numpy std, ddof 0)

1. Split 0 (the primary): <out>/r5k/R5K_<tag>.pt and <out>/r9k/R9K_<tag>.pt, with the same keys as the D + K files
   (`direction`, `unit`, `sd_r5|sd_r9`, `sd_k`, `cos_r5_k|cos_r9_k`, the Alpaca projections and row ids) + .json.
2. All 100 splits of bank/splits.json: <out>/rk_null/RK_null_<tag>.csv -- per split, contrast (R5, R9) and layer, the
   spread ratio sd(A @ R_unit) / sd(A @ K), cos(R, K) and cos(R + K, K), the statistics reported for D9 + K. Computed
   from each role's projection on the Alpaca rows (A @ R is linear in the role vectors), so one pass over the store
   serves every split.

The Alpaca store is streamed one chunk at a time, as in d9k.py; only per-row projections are kept.

    python random_persona_bank/scripts/rk.py --alpaca <alpaca store> --k <K.npz> --rdir <run folder> --tag s23 \
        --vectors <run>/persona/vectors_s23

--validate D_FILE D9K_FILE: rebuild a D + K file with this script's code and compare (writes nothing).
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "bank" / "splits.json"
REPORT_LAYERS = (20, 31, 36, 40, 44, 63)
RECIPE = "zami recall_denial.standardised_sum, origin/paper @ e27ffaf9 (recall_denial.py:428-431), as xmodel/d9k.py"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream(alpaca: Path, dirs: np.ndarray, layers: list):
    """Project every Alpaca row on each direction. dirs [D, L, H] -> (row_ids, P [N, L, D], reply tokens)."""
    items = torch.load(alpaca / "items.pt", weights_only=False)
    want = [it[0] for it in items]
    ids, proj, ntok = [], [], []
    chunks = sorted(alpaca.glob("chunk_*.pt"))
    for c in chunks:
        ch = torch.load(c, weights_only=False)
        x = ch["pooled"]
        if x.dim() != 4 or x.shape[1] != len(layers) or x.shape[2] != dirs.shape[2]:
            sys.exit(f"{c.name}: pooled shape {tuple(x.shape)} does not fit {len(layers)} layers x {dirs.shape[2]}")
        if list(ch["layers"]) != layers or list(ch["pooling"])[0] != "mean" or ch["span"] != "response":
            sys.exit(f"{c.name}: layers/pooling/span are not the directions' layers, mean, response")
        x = x[..., 0].double().numpy()
        if not np.isfinite(x).all():
            sys.exit(f"{c.name}: non-finite pooled values")
        proj.append(np.einsum("nlh,dlh->nld", x, dirs))
        ids += list(ch["row_ids"])
        ntok += [int(round(float(v))) for v in ch["n_tok"]]
        del x, ch
    if len(set(ids)) != len(ids) or set(ids) != set(want):
        sys.exit(f"Alpaca store incomplete or inconsistent: {len(set(ids))} rows read, {len(want)} listed")
    order = np.argsort([want.index(r) for r in ids])
    return [ids[i] for i in order], np.concatenate(proj)[order], ntok, len(chunks)


def standardised_sum(D, K, PD, PK):
    """d9k.py's arithmetic: D, K [L, H] unit; PD, PK [N, L] projections."""
    sdd, sdk = PD.std(0), PK.std(0)
    W = D / sdd[:, None] + K / sdk[:, None]
    return W, sdd, sdk, (D * K).sum(1)


def write_rk(out: Path, name: str, rfile: Path, kfile: Path, alpaca: Path, layers, D, K, PD, PK, ids, ntok):
    W, sdd, sdk, cos = standardised_sum(D, K, PD, PK)
    n = name.lower()
    out.mkdir(parents=True, exist_ok=True)
    tag = rfile.stem.split("_", 1)[1]
    pt, js = out / f"{name}K_{tag}.pt", out / f"{name}K_{tag}.json"
    if pt.exists() or js.exists():
        sys.exit(f"{pt} exists; refusing to overwrite")
    stamp = datetime.now().isoformat(timespec="seconds")
    torch.save({"name": f"{name} + K", "recipe": RECIPE, "layers": layers,
                "direction": torch.from_numpy(W).float(),
                "unit": torch.from_numpy(W / np.linalg.norm(W, axis=1, keepdims=True)).float(),
                f"sd_{n}": torch.from_numpy(sdd), "sd_k": torch.from_numpy(sdk), f"cos_{n}_k": torch.from_numpy(cos),
                f"alpaca_proj_{n}": torch.from_numpy(PD).float(), "alpaca_proj_k": torch.from_numpy(PK).float(),
                "row_ids": ids, n: str(rfile), f"{n}_md5": md5(rfile), "k": str(kfile), "k_md5": md5(kfile),
                "alpaca": str(alpaca), "n_alpaca": len(ids), "created": stamp}, pt)
    js.write_text(json.dumps({
        "name": f"{name} + K", "recipe": RECIPE, n: str(rfile), f"{n}_md5": md5(rfile), "k": str(kfile),
        "k_md5": md5(kfile), "k_key": "none|general (first item set)", "alpaca": str(alpaca), "n_alpaca": len(ids),
        "reply_tokens_median": int(np.median(ntok)),
        "at": {f"L{l}": {f"sd_{n}": round(float(sdd[layers.index(l)]), 6), "sd_k": round(float(sdk[layers.index(l)]), 6),
                         f"cos_{n}_k": round(float(cos[layers.index(l)]), 6)} for l in REPORT_LAYERS if l in layers},
        "use": "score = activation @ direction[layer]; the R half's sign is arbitrary (random groups)",
        "pt_md5": md5(pt), "created": stamp}, indent=2), encoding="utf-8")
    return W, sdd, sdk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True, type=Path)
    ap.add_argument("--k", required=True, type=Path)
    ap.add_argument("--rdir", type=Path, help="run folder holding r5/ and r9/ (from r_contrasts.py)")
    ap.add_argument("--tag")
    ap.add_argument("--vectors", type=Path, help="the bank r_contrasts.py read (for the 100-split null)")
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--validate", nargs=2, type=Path, metavar=("D_FILE", "DK_FILE"))
    a = ap.parse_args()
    kz = np.load(a.k)

    if a.validate:
        dfile, dkfile = a.validate
        d = torch.load(dfile, weights_only=False)
        layers = list(d["layers"])
        D = d["unit"].double().numpy()
        K = np.stack([kz[f"L{l}"] for l in layers]).astype(np.float64)
        ids, P, ntok, nch = stream(a.alpaca, np.stack([D, K]), layers)
        W, sdd, sdk, _ = standardised_sum(D, K, P[:, :, 0], P[:, :, 1])
        ref = torch.load(dkfile, weights_only=False)["direction"].double().numpy()
        print(f"{nch} chunks, {len(ids)} rows; rebuilt {d['name']} + K vs {dkfile.name}: "
              f"max relative difference {np.abs(W - ref).max() / np.abs(ref).max():.2e}")
        return

    if not (a.rdir and a.tag and a.vectors):
        sys.exit("--rdir, --tag and --vectors are required")
    S = json.loads(a.splits.read_text(encoding="utf-8"))
    r5f, r9f = a.rdir / "r5" / f"R5_{a.tag}.pt", a.rdir / "r9" / f"R9_{a.tag}.pt"
    r5, r9 = torch.load(r5f, weights_only=False), torch.load(r9f, weights_only=False)
    layers = list(r5["layers"])
    K = np.stack([kz[f"L{l}"] for l in layers]).astype(np.float64)
    if not np.allclose(np.linalg.norm(K, axis=1), 1, atol=1e-6):
        sys.exit("K is not unit-norm")
    vec = {p.stem: torch.load(p, weights_only=False)["vector"].double().numpy() for p in sorted(a.vectors.glob("*.pt"))}
    roles = sorted(r for r in vec if r != "default")
    V = np.stack([vec[r] for r in roles])                                 # [35, L, H]
    if V.shape[1] != len(layers):
        sys.exit("bank layers differ from the R files'")

    # one pass: rows x layers x (35 role vectors, K)
    ids, P, ntok, nch = stream(a.alpaca, np.concatenate([V, K[None]]), layers)
    PV, PK = P[:, :, :35], P[:, :, 35]
    print(f"{nch} chunks, {len(ids)} Alpaca rows, reply tokens median {int(np.median(ntok))}")

    # split 0 files, from the R files themselves (their unit directions)
    for name, f, rd in (("R5", r5f, r5), ("R9", r9f, r9)):
        D = rd["unit"].double().numpy()
        idx_p = [roles.index(r) for r in rd["positive"]]; idx_n = [roles.index(r) for r in rd["negative"]]
        PD = PV[:, :, idx_p].mean(2) - PV[:, :, idx_n].mean(2)
        PD = PD / rd["norm"].double().numpy()[None]                        # A @ R_unit
        check = np.einsum("lh,lh->l", D, D)
        assert np.allclose(check, 1, atol=1e-5)
        write_rk(a.rdir / f"{name.lower()}k", name, f, a.k, a.alpaca, layers, D, K, PD, PK, ids, ntok)
        print(f"wrote {name}K_{a.tag}")

    # 100-split null for R5 + K and R9 + K
    VK = np.einsum("rlh,lh->rl", V, K)                                       # [35, L] role . K
    null_csv = a.rdir / "rk_null" / f"RK_null_{a.tag}.csv"
    if null_csv.exists():
        sys.exit(f"{null_csv} exists; refusing to overwrite")
    null_csv.parent.mkdir(parents=True, exist_ok=True)
    sdk = PK.std(0)
    summary = {}
    with open(null_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "contrast", "layer", "sd_r", "sd_k", "ratio", "cos_r_k", "cos_rk_k"])
        for sp in S["splits"]:
            g = sp["groups"]
            for name, pos_g, neg_g in (("R5", "A", "B"), ("R9", "AE", "BF")):
                pos = [r for x in pos_g for r in g[x]]; neg = [r for x in neg_g for r in g[x]]
                ip = [roles.index(r) for r in pos]; ineg = [roles.index(r) for r in neg]
                raw = V[ip].mean(0) - V[ineg].mean(0)                       # [L, H]
                nrm = np.linalg.norm(raw, axis=1)
                PD = (PV[:, :, ip].mean(2) - PV[:, :, ineg].mean(2)) / nrm[None]
                sdr = PD.std(0)
                cos_rk = (VK[ip].mean(0) - VK[ineg].mean(0)) / nrm                # cos(R_unit, K)
                wn = np.sqrt(1 / sdr**2 + 1 / sdk**2 + 2 * cos_rk / (sdr * sdk))   # |R/sdr + K/sdk|
                cos_w_k = (cos_rk / sdr + 1 / sdk) / wn
                for i, l in enumerate(layers):
                    w.writerow([sp["split"], name, l, f"{sdr[i]:.6f}", f"{sdk[i]:.6f}", f"{sdr[i] / sdk[i]:.6f}",
                                f"{cos_rk[i]:+.6f}", f"{cos_w_k[i]:.6f}"])
                summary.setdefault(name, []).append((sdr / sdk, cos_w_k))
    for name, vals in summary.items():
        ratio = np.stack([v[0] for v in vals]); cw = np.stack([v[1] for v in vals])
        for l in REPORT_LAYERS:
            if l in layers:
                i = layers.index(l)
                print(f"{name} + K  L{l}: split 0 ratio {ratio[0, i]:.2f}, cos(R+K, K) {cw[0, i]:.3f} | 100 splits: ratio "
                      f"{np.percentile(ratio[:, i], 2.5):.2f}-{np.percentile(ratio[:, i], 97.5):.2f}, cos(R+K, K) "
                      f"{np.percentile(cw[:, i], 2.5):.3f}-{np.percentile(cw[:, i], 97.5):.3f}")
    print(f"wrote {null_csv}")


if __name__ == "__main__":
    main()
