"""Follow-up detection tests A-D, fixed in README.md ("Follow-up, fixed before the data -- 2026-10-16").

Same scoring, bootstrap and verdict rule as detect.py (review_response/T1-4_or_rule's recipe):
    score_D = x @ D_unit[layer];  score_D+K = score_D / sd_A(D) + score_K / sd_A(K)  (Alpaca sd, ddof 0)
    AUROC: label 1 (lie) v 0 (honest), no sign flip; paired row bootstrap, 2,000, stratified by label, seed 0.

    --model llama   A (kr-matched; D9 / D8 / D5 on HP-KR), B (six new + all eleven datasets), C (blend), bank s3
    --model qwen    D (ID, HP-KR, HP-C from eval_lb), bank s23

Writes detect_followup_<model>.{json,md} and detect_followup_<model>_scores.csv.gz next to this file; refuses to overwrite.

    .venv-analysis/Scripts/python.exe -I xmodel/userstay_pair/detect_followup.py --model llama
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOME = Path.home()
LAYERS = (20, 36, 40, 44)
N_BOOT, SEED = 2000, 0
PASSB = HOME / "xmodel-runs/_scratch/passB_llama"
CFG = {
    "llama": {
        "root": {"original": HOME / "xmodel-runs/Llama-3.3-70B-Instruct",
                 "user_stay": HOME / "xmodel-runs-userstay/Llama-3.3-70B-Instruct"},
        "tag": {"original": "s3", "user_stay": "s3"},
        "stores": {"CG": PASSB / "convincing-game", "HP-C": PASSB / "harm-pressure-choice",
                   "HP-KR": PASSB / "harm-pressure-knowledge-report", "IT": PASSB / "insider-trading",
                   "ID": PASSB / "instructed-deception", "KR-matched": PASSB / "harm-pressure-kr-matched",
                   "ai_liar": PASSB / "ai_liar", "roleplaying": PASSB / "roleplaying",
                   "sandbagging": PASSB / "sandbagging_v2", "mask": PASSB / "mask", "sycophancy_mt": PASSB / "sycophancy_mt"},
        "old": ("CG", "HP-C", "HP-KR", "IT", "ID"),
        "new": ("KR-matched", "ai_liar", "roleplaying", "sandbagging", "mask", "sycophancy_mt"),
    },
    "qwen": {
        "root": {"original": HOME / "xmodel-runs/Qwen2.5-72B-Instruct",
                 "user_stay": HOME / "xmodel-runs-userstay/Qwen2.5-72B-Instruct"},
        "tag": {"original": "s23_v2", "user_stay": "s23"},
        "stores": {"ID": HOME / "xmodel-runs/Qwen2.5-72B-Instruct/eval_lb/instructed-deception",
                   "HP-KR": HOME / "xmodel-runs/Qwen2.5-72B-Instruct/eval_lb/harm-pressure-knowledge-report",
                   "HP-C": HOME / "xmodel-runs/Qwen2.5-72B-Instruct/eval_lb/harm-pressure-choice"},
        "old": ("ID", "HP-KR", "HP-C"),
        "new": (),
    },
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auroc(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def directions(cfg):
    """unit[name] (4, H) and alpaca[name] (2000, 4) for D9 / D8 / D5 per format, K, and the D9 blend."""
    units, alp, raw, norms, inputs = {}, {}, {}, {}, {}
    kpath = None
    for fmt in ("original", "user_stay"):
        root, tag = cfg["root"][fmt], cfg["tag"][fmt]
        for D in ("D9", "D8", "D5"):
            dp = root / D.lower() / f"{D}_{tag}.pt"
            d = torch.load(dp, weights_only=False)
            assert d["name"] == D
            lay = list(d["layers"])
            li = [lay.index(l) for l in LAYERS]
            units[f"{fmt}|{D}"] = d["unit"].double().numpy()[li]
            raw[f"{fmt}|{D}"] = d["raw"].double().numpy()[li]
            inputs[f"{fmt}|{D}"] = [str(dp), md5(dp)]
            if D == "D8":
                continue
            kp = root / f"{D.lower()}k" / f"{D}K_{tag}.pt"
            k = torch.load(kp, weights_only=False)
            assert k[f"{D.lower()}_md5"] == md5(dp), f"{kp} built from another {D}"
            kpath = kpath or Path(k["k"])
            assert k["k_md5"] == md5(kpath), f"{kp} built from another K"
            la = list(k["layers"])
            lj = [la.index(l) for l in LAYERS]
            alp[f"{fmt}|{D}"] = k[f"alpaca_proj_{D.lower()}"].double().numpy()[:, lj]
            alp["K"] = k["alpaca_proj_k"].double().numpy()[:, lj]
            inputs[f"{fmt}|{D}K"] = [str(kp), md5(kp)]
    kz = np.load(kpath)
    units["K"] = np.stack([kz[f"L{l}"] for l in LAYERS]).astype(np.float64)
    inputs["K"] = [str(kpath), md5(kpath)]
    # C: blend = unit(raw_o + raw_u); A . u_b = (n_o A.u_o + n_u A.u_u) / |raw_o + raw_u|
    ro, ru = raw["original|D9"], raw["user_stay|D9"]
    no, nu = np.linalg.norm(ro, axis=1), np.linalg.norm(ru, axis=1)
    assert np.allclose(ro / no[:, None], units["original|D9"], atol=1e-5)
    rb = ro + ru
    nb = np.linalg.norm(rb, axis=1)
    units["blend|D9"] = rb / nb[:, None]
    alp["blend|D9"] = (alp["original|D9"] * no + alp["user_stay|D9"] * nu) / nb
    return units, alp, inputs


def project(cfg, units):
    names = list(units)
    W = np.stack([units[n] for n in names], axis=1)                     # (4, n_dirs, H)
    recs, store_md5, dropped = [], {}, {}
    for short, root in cfg["stores"].items():
        if not (root / "rows.jsonl").exists():
            dropped[short] = "store missing"
            continue
        labels = {}
        with open(root / "rows.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                labels[r["row_id"]] = r["label"]
        # harm-pressure-kr-matched has no items.pt: its chunks are HP-KR's 2,139-row read and rows.jsonl lists the 258
        # wording-matched rows (F-20, F-23). There the row list is rows.jsonl, and chunk rows not in it are skipped.
        sliced = not (root / "items.pt").exists()
        want = list(labels) if sliced else [it[0] for it in torch.load(root / "items.pt", weights_only=False)]
        seen = []
        for c in sorted(root.glob("chunk_*.pt")):
            ch = torch.load(c, weights_only=False)
            lay = [int(x) for x in ch["layers"]]
            if list(ch["pooling"])[0] != "mean" or ch.get("span", "response") != "response":
                sys.exit(f"{short}/{c.name}: pooling {ch['pooling']} span {ch.get('span')}")
            x = ch["pooled"][:, [lay.index(l) for l in LAYERS], :, 0].double().numpy()
            if not np.isfinite(x).all():
                sys.exit(f"{short}/{c.name}: non-finite values")
            p = np.einsum("nlh,ldh->nld", x, W)
            for i, rid in enumerate(ch["row_ids"]):
                if sliced and rid not in labels:
                    continue
                seen.append(rid)
                for j, l in enumerate(LAYERS):
                    rec = {"dataset": short, "row_id": rid, "label": labels[rid], "layer": l}
                    rec.update({n: p[i, j, q] for q, n in enumerate(names)})
                    recs.append(rec)
            del x, ch
        if len(set(seen)) != len(seen) or set(seen) != set(want):
            sys.exit(f"{short}: rows read ({len(set(seen))}) do not match items.pt ({len(set(want))})")
        store_md5[short] = {q.name: md5(q) for q in sorted(root.iterdir()) if q.suffix in (".pt", ".jsonl")}
        print(f"{short:13s} rows {len(seen):5d}  labels {pd.Series([labels[r] for r in seen]).value_counts().to_dict()}",
              flush=True)
    df = pd.DataFrame(recs)
    df = df[df.label.isin([0, 1])].copy()
    df["label"] = df.label.astype(int)
    for short in list(df.dataset.unique()):
        n1 = int(df[(df.dataset == short) & (df.layer == 40)].label.sum())
        n0 = int((df[(df.dataset == short) & (df.layer == 40)].label == 0).sum())
        if n1 == 0 or n0 == 0:
            dropped[short] = f"one class only (lies {n1}, honest {n0})"
            df = df[df.dataset != short]
    return df, store_md5, dropped


def scorer_cols(df, alp, layer):
    """{scorer: (scores on df, Alpaca scores)} at one layer."""
    j = LAYERS.index(layer)
    sdk = alp["K"][:, j].std()
    out = {"K": (df["K"].to_numpy(), alp["K"][:, j])}
    for fmt in ("original", "user_stay", "blend"):
        for D in ("D9", "D8", "D5"):
            key = f"{fmt}|{D}"
            if key not in df:
                continue
            out[key] = (df[key].to_numpy(), alp[key][:, j] if key in alp else None)
            if key in alp:
                sdd = alp[key][:, j].std()
                out[f"{key}+K"] = ((df[key] / sdd + df["K"] / sdk).to_numpy(), alp[key][:, j] / sdd + alp["K"][:, j] / sdk)
    return out


def paired(df40, alp, a, b, datasets, rng_seed=SEED):
    """Bootstrap of AUROC(b) - AUROC(a), per dataset and for the mean over `datasets`."""
    rng = np.random.default_rng(rng_seed)
    per, boots = {}, []
    for s in datasets:
        d = df40[df40.dataset == s]
        y = d.label.to_numpy()
        sc = scorer_cols(d, alp, 40)
        sa, sb = sc[a][0], sc[b][0]
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        bs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            bs[i] = auroc(y[ix], sb[ix]) - auroc(y[ix], sa[ix])
        per[s] = {"a": auroc(y, sa), "b": auroc(y, sb), "delta": auroc(y, sb) - auroc(y, sa),
                  "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
        boots.append(bs)
    m = np.mean(boots, axis=0)
    mean = {"delta": float(np.mean([per[s]["delta"] for s in datasets])),
            "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975))}
    verdict = "worse" if mean["hi"] < 0 else "better" if mean["lo"] > 0 else "no detectable change"
    return {"per": per, "mean": mean, "verdict": verdict,
            "n_below": sum(per[s]["hi"] < 0 for s in datasets), "n_above": sum(per[s]["lo"] > 0 for s in datasets)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    a = ap.parse_args()
    cfg = CFG[a.model]
    outs = [HERE / f"detect_followup_{a.model}_scores.csv.gz", HERE / f"detect_followup_{a.model}.json",
            HERE / f"detect_followup_{a.model}.md"]
    if any(o.exists() for o in outs):
        sys.exit("outputs exist; refusing to overwrite")
    units, alp, inputs = directions(cfg)
    df, store_md5, dropped = project(cfg, units)
    old = [s for s in cfg["old"] if s in set(df.dataset)]
    new = [s for s in cfg["new"] if s in set(df.dataset)]
    allds = old + new

    # AUROC table, every scorer x dataset x layer
    rows = []
    for l in LAYERS:
        dl = df[df.layer == l]
        for s in allds:
            d = dl[dl.dataset == s]
            y = d.label.to_numpy()
            for name, (sc, ca) in scorer_cols(d, alp, l).items():
                r = {"layer": l, "dataset": s, "scorer": name, "n_lies": int(y.sum()), "n_honest": int((y == 0).sum()),
                     "auroc": auroc(y, sc)}
                if ca is not None:
                    thr = float(np.quantile(ca, 0.99))
                    r.update(rec_alp=float(np.mean(sc[y == 1] > thr)), fp_alp=float(np.mean(sc[y == 0] > thr)))
                rows.append(r)
    t = pd.DataFrame(rows)

    # sanity (llama): the original format reproduces detect.json on the five old datasets, L40 s3
    sanity = None
    if a.model == "llama":
        ref = pd.DataFrame(json.loads((HERE / "detect.json").read_text(encoding="utf-8"))["table"])
        ref = ref[(ref.layer == 40) & (ref.format == "original") & (ref.bank == "s3") & ref.scorer.isin(["D9", "D9 + K", "D5"])]
        name = {"D9": "original|D9", "D9 + K": "original|D9+K", "D5": "original|D5"}
        diffs = []
        for _, r in ref.iterrows():
            if r.dataset not in old:
                continue
            m = t[(t.layer == 40) & (t.dataset == r.dataset) & (t.scorer == name[r.scorer])]
            diffs.append(abs(float(m.auroc.iloc[0]) - r.auroc))
        sanity = {"cells": len(diffs), "max_abs_diff": max(diffs)}
        print(f"sanity v detect.json: {len(diffs)} cells, max |diff| {max(diffs):.2e}", flush=True)
        if max(diffs) > 1e-6:
            sys.exit("sanity failed; nothing compared")

    df40 = df[df.layer == 40]
    res = {"model": a.model, "datasets": {"old": old, "new": new, "dropped": dropped}, "sanity": sanity, "tests": {}}
    T = res["tests"]
    if a.model == "llama":
        T["B_new6_D9K_user_stay_v_original"] = paired(df40, alp, "original|D9+K", "user_stay|D9+K", new)
        T["B_all11_D9K_user_stay_v_original"] = paired(df40, alp, "original|D9+K", "user_stay|D9+K", allds)
        T["C_new6_D9K_blend_v_original"] = paired(df40, alp, "original|D9+K", "blend|D9+K", new)
        T["C_old5_D9K_blend_v_original_report_only"] = paired(df40, alp, "original|D9+K", "blend|D9+K", old)
        for D in ("D9", "D8", "D5"):
            T[f"A_{D}_user_stay_v_original_HPKR_KRmatched"] = paired(df40, alp, f"original|{D}", f"user_stay|{D}",
                                                                      [s for s in ("HP-KR", "KR-matched") if s in allds])
        T["B_new6_D9_user_stay_v_original"] = paired(df40, alp, "original|D9", "user_stay|D9", new)
    else:
        for D in ("D9", "D5", "D8"):
            T[f"D_{D}_user_stay_v_original"] = paired(df40, alp, f"original|{D}", f"user_stay|{D}", old)
        T["D_D9K_user_stay_v_original"] = paired(df40, alp, "original|D9+K", "user_stay|D9+K", old)
        T["D_D5K_user_stay_v_original"] = paired(df40, alp, "original|D5+K", "user_stay|D5+K", old)
    for k, v in T.items():
        print(f"{k:48s} mean {v['mean']['delta']:+.4f} [{v['mean']['lo']:+.4f}, {v['mean']['hi']:+.4f}]  {v['verdict']}"
              f"  ({v['n_below']} down, {v['n_above']} up)", flush=True)

    df.to_csv(outs[0], index=False, float_format="%.6f")
    res.update(table=t.to_dict(orient="records"), inputs=inputs, stores=store_md5, n_boot=N_BOOT, seed=SEED)
    outs[1].write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    outs[2].write_text(report(res, t, allds), encoding="utf-8")
    print(report(res, t, allds))


def report(res, t, allds):
    L = [f"# Follow-up detection tests ({res['model']})", "", "Generated by `detect_followup.py`; numbers in the .json.", ""]
    if res["datasets"]["dropped"]:
        L += ["Dropped: " + "; ".join(f"{k} ({v})" for k, v in res["datasets"]["dropped"].items()), ""]
    if res["sanity"]:
        L += [f"Sanity: the original format reproduces detect.json on {res['sanity']['cells']} cells "
              f"(max |diff| {res['sanity']['max_abs_diff']:.1e}).", ""]
    L += ["## Tests (L40; Δ = second − first, paired 95% CI)", ""]
    for k, v in res["tests"].items():
        L.append(f"**{k}**: mean Δ {v['mean']['delta']:+.3f} [{v['mean']['lo']:+.3f}, {v['mean']['hi']:+.3f}] → "
                 f"**{v['verdict']}** ({v['n_below']} datasets down, {v['n_above']} up)")
        L.append("")
        L.append("| dataset | first | second | Δ [95% CI] |")
        L.append("|---|---|---|---|")
        for s, p in v["per"].items():
            L.append(f"| {s} | {p['a']:.3f} | {p['b']:.3f} | {p['delta']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] |")
        L.append("")
    show = [c for c in ("original|D9+K", "user_stay|D9+K", "blend|D9+K", "original|D9", "user_stay|D9", "blend|D9",
                        "original|D8", "user_stay|D8", "original|D5", "user_stay|D5", "K") if c in set(t.scorer)]
    L += ["## AUROC at L40, every dataset", "", "| scorer | " + " | ".join(allds) + " |", "|---|" + "---|" * len(allds)]
    t40 = t[t.layer == 40]
    for c in show:
        L.append(f"| {c} | " + " | ".join(f"{t40[(t40.scorer == c) & (t40.dataset == s)].auroc.iloc[0]:.3f}" for s in allds) + " |")
    L += ["", "## Mean AUROC over the datasets, by layer", "", "| scorer | " + " | ".join(f"L{l}" for l in LAYERS) + " |",
          "|---|" + "---|" * len(LAYERS)]
    for c in show:
        L.append(f"| {c} | " + " | ".join(f"{t[(t.layer == l) & (t.scorer == c)].auroc.mean():.3f}" for l in LAYERS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
