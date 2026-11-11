"""D9 + K, the selected detector (report sections 4.2.7-4.2.8), for one model at every layer.

zami's `recall_denial.standardised_sum` (origin/paper @ e27ffaf9, recall_denial.py:428-431):

    D9 + K = D9_unit / sd(Alpaca @ D9_unit) + K / sd(Alpaca @ K)

each sd the spread (numpy std, ddof 0) of that direction's projection on the Alpaca control rows, so the two
contribute equally in score units. D9 = `d9/D9_<tag>.pt` (d_contrasts.py), K = `K/<model>/first/K.npz`
(none|general, unit), Alpaca = 8_coords_ndif.py chunks (the model's own LB Alpaca rows, mean over the reply).

Chunks are streamed one at a time (2,000 rows x 80 layers x 8192 is 2.6 GB in fp16; the laptop has 7.7 GB),
and only the two projections per row and layer are kept. Refuses to overwrite.

Writes <out>/D9K_<tag>.pt: `direction` [80, 8192] float32 (unnormalised, as standardised_sum returns it),
`unit`, `sd_d9`, `sd_k`, `cos_d9_k` per layer, and the Alpaca projections `alpaca_proj_d9` / `alpaca_proj_k`
[2000, 80] with their `row_ids`, so a 1%-FPR threshold on Alpaca needs no second read. Plus a .json sidecar.

    python xmodel/d9k.py --alpaca <run>/eval/alpaca --d9 <run>/d9/D9_s23_v2.pt --k <run>/K/qwen-72b/first/K.npz \
        --out <run>/d9k --tag s23_v2

D5 + K (report Table 4.1) is the same sum with D5 in place of D9 (2026-10-08): `--name D5 --d <run>/d5/D5_<tag>.pt
--out <run>/d5k` writes D5K_<tag>.pt, whose keys say d5 where the D9 file's say d9 (`sd_d5`, `alpaca_proj_d5`, ...).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPORT_LAYERS = (20, 36, 40, 44)
RECIPE = "zami recall_denial.standardised_sum, origin/paper @ e27ffaf9 (recall_denial.py:428-431)"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True, type=Path, help="8_coords output dir for alpaca (items.pt + chunks)")
    ap.add_argument("--d", "--d9", dest="d", required=True, type=Path, help="contrast file from d_contrasts.py")
    ap.add_argument("--name", default="D9", choices=("D5", "D9"), help="the contrast's name, as stored in its file")
    ap.add_argument("--k", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--allow_partial", action="store_true", help="test on an unfinished read (never for results)")
    a = ap.parse_args()
    n = a.name.lower()

    d9 = torch.load(a.d, weights_only=False)
    if d9.get("name") != a.name:
        sys.exit(f"{a.d} holds {d9.get('name')}, not {a.name}")
    D = d9["unit"].double().numpy()                                   # [L, H]
    layers = list(d9["layers"])
    # 2026-10-09 (Llama-405B): an Alpaca store read at a layer SUBSET -- D + K is then built at those layers only
    first = sorted(a.alpaca.glob("chunk_*.pt"))
    if first:
        store_layers = [int(x) for x in torch.load(first[0], weights_only=False)["layers"]]
        if store_layers != layers:
            if not set(store_layers) <= set(layers):
                sys.exit(f"Alpaca store layers {store_layers[:4]}.. are not a subset of the contrast's")
            D = D[[layers.index(l) for l in store_layers]]
            layers = store_layers
            print(f"Alpaca store holds {len(layers)} of the contrast's layers; {a.name} + K is built at those only")
    kz = np.load(a.k)
    K = np.stack([kz[f"L{l}"] for l in layers]).astype(np.float64)  # [L, H]
    if not np.allclose(np.linalg.norm(K, axis=1), 1, atol=1e-6) or not np.allclose(np.linalg.norm(D, axis=1), 1, atol=1e-5):
        sys.exit(f"{a.name} unit or K is not unit-norm")

    items = torch.load(a.alpaca / "items.pt", weights_only=False)
    want = [it[0] for it in items]
    ids, pd9, pk, ntok = [], [], [], []
    chunks = sorted(a.alpaca.glob("chunk_*.pt"))
    for c in chunks:
        ch = torch.load(c, weights_only=False)
        x = ch["pooled"]                                   # (n, K, H, P) fp16
        if x.dim() != 4 or x.shape[1] != len(layers) or x.shape[2] != D.shape[1]:
            sys.exit(f"{c.name}: pooled shape {tuple(x.shape)} does not fit {len(layers)} layers x {D.shape[1]}")
        if list(ch["layers"]) != layers or list(ch["pooling"])[0] != "mean" or ch["span"] != "response":
            sys.exit(f"{c.name}: layers/pooling/span {ch['layers'][:3]}.., {ch['pooling']}, {ch['span']} "
                     f"are not the D9's layers, mean, response")
        x = x[..., 0].double().numpy()                     # mean pooling is the only one read
        if not np.isfinite(x).all():
            sys.exit(f"{c.name}: non-finite pooled values")
        pd9.append(np.einsum("nlh,lh->nl", x, D))
        pk.append(np.einsum("nlh,lh->nl", x, K))
        ids += list(ch["row_ids"])
        ntok += [int(round(float(v))) for v in ch["n_tok"]]
        del x, ch
    if len(set(ids)) != len(ids):
        sys.exit("a row appears in two chunks")
    missing = set(want) - set(ids)
    if missing and not a.allow_partial:
        sys.exit(f"{len(missing)} of {len(want)} Alpaca rows not read yet")
    if set(ids) - set(want):
        sys.exit("chunks hold rows that items.pt does not list")
    order = np.argsort([want.index(r) for r in ids])     # items order, so rows pair with items.pt
    ids = [ids[i] for i in order]
    P9, PK = np.concatenate(pd9)[order], np.concatenate(pk)[order]
    sd9, sdk = P9.std(0), PK.std(0)                       # numpy default ddof 0, as standardised_sum
    W = D / sd9[:, None] + K / sdk[:, None]
    cos = (D * K).sum(1)
    print(f"Alpaca rows: {len(ids)} of {len(want)}; chunks {len(chunks)}; reply tokens median {int(np.median(ntok))}")
    s = a.name
    print(f"layer  sd(A@{s})   sd(A@K)   cos({s},K)  |{s}+K|   cos({s}+K, {s})  cos({s}+K, K)")
    for l in REPORT_LAYERS:
        if l in layers:
            i = layers.index(l)
            u = W[i] / np.linalg.norm(W[i])
            print(f"L{l:<4} {sd9[i]:9.4f} {sdk[i]:9.4f}   {cos[i]:+.4f}  {np.linalg.norm(W[i]):7.4f}     "
                  f"{u @ D[i]:+.4f}        {u @ K[i]:+.4f}")
    if a.allow_partial:
        print("partial test: nothing written")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    pt, js = a.out / f"{s}K_{a.tag}.pt", a.out / f"{s}K_{a.tag}.json"
    if pt.exists() or js.exists():
        sys.exit(f"{pt} exists; refusing to overwrite")
    stamp = datetime.now().isoformat(timespec="seconds")
    torch.save({"name": f"{s} + K", "recipe": RECIPE, "layers": layers,
                "direction": torch.from_numpy(W).float(),
                "unit": torch.from_numpy(W / np.linalg.norm(W, axis=1, keepdims=True)).float(),
                f"sd_{n}": torch.from_numpy(sd9), "sd_k": torch.from_numpy(sdk), f"cos_{n}_k": torch.from_numpy(cos),
                f"alpaca_proj_{n}": torch.from_numpy(P9).float(), "alpaca_proj_k": torch.from_numpy(PK).float(),
                "row_ids": ids, n: str(a.d), f"{n}_md5": md5(a.d), "k": str(a.k), "k_md5": md5(a.k),
                "alpaca": str(a.alpaca), "n_alpaca": len(ids), "created": stamp}, pt)
    side = {"name": f"{s} + K", "recipe": RECIPE, n: str(a.d), f"{n}_md5": md5(a.d), "k": str(a.k), "k_md5": md5(a.k),
            "k_key": "none|general (first item set)", "alpaca": str(a.alpaca), "n_alpaca": len(ids),
            "reply_tokens_median": int(np.median(ntok)),
            "at": {f"L{l}": {f"sd_{n}": round(float(sd9[layers.index(l)]), 6), "sd_k": round(float(sdk[layers.index(l)]), 6),
                             f"cos_{n}_k": round(float(cos[layers.index(l)]), 6)} for l in REPORT_LAYERS if l in layers},
            "use": "score = activation @ direction[layer]; positive = deceptive", "pt_md5": md5(pt), "created": stamp}
    js.write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(f"wrote {pt} ({pt.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
