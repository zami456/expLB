"""The four comparisons fixed in README.md ("Comparisons, fixed before the data"): original v user_stay, per model.

Offline; reads only stored files. All banks are s23 (scores 2|3), the main bank. Writes compare.json and compare.md
next to this file.

    1  role play     judge score-3 share per model, format and role group; roles with >= 20 score-3 replies
    2  geometry      cos(PC1, assistant), PC1 variance share, behaviour-PC1 . assistant -- every layer (6_axes.py files)
    3  vector change per-role cos(v_original, v_user_stay), raw and centred on the bank mean; cos between the two
                     formats' axes and D's
    4  D's, D9 + K   cos(D5, D8), cos(D5, D9); D9 + K spread ratio sd(A.D9) / sd(A.K) and cos(D9 + K, K) (d9k.py files)

    .venv-analysis/Scripts/python.exe -I xmodel/userstay_pair/compare.py
"""
import json
import statistics
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAYERS = (20, 36, 40, 44)            # the report layers; 40 is the thesis layer for both models
BEHAVIOURS = ("deception", "sycophancy", "unified", "knowledge_denial")

RUNS = {
    "Qwen": {
        "original": {"root": HOME / "xmodel-runs/Qwen2.5-72B-Instruct", "scores": "persona/scores",
                     "vectors": "persona/vectors_s23_v2", "axes": "persona/axes_vectors_s23_v2.pt",
                     "d": "{d}/{D}_s23_v2.pt", "d9k": "d9k/D9K_s23_v2.pt"},
        "user_stay": {"root": HOME / "xmodel-runs-userstay/Qwen2.5-72B-Instruct", "scores": "persona/scores",
                      "vectors": "persona/vectors_s23", "axes": "persona/axes_vectors_s23.pt",
                      "d": "{d}/{D}_s23.pt", "d9k": "d9k/D9K_s23.pt"},
    },
    "Llama-3.3": {
        # original: the thesis kd bank, lab run-kd via the 10-06 Drive backup (xmodel/LOG.md, rclone-checked);
        # scores: the md5-checked copy of the lab's run-kd scores in qwen_roleplay/ref_llama33_kd
        "original": {"root": HOME / "xmodel-runs/Llama-3.3-70B-Instruct",
                     "scores": REPO / "xmodel/qwen_roleplay/ref_llama33_kd/scores",
                     "vectors": "persona/vectors_s23", "axes": "persona/axes_vectors_s23.pt",
                     "d": "{d}/{D}_s23.pt", "d9k": "d9k/D9K_s23.pt"},
        "user_stay": {"root": HOME / "xmodel-runs-userstay/Llama-3.3-70B-Instruct", "scores": "persona/scores",
                      "vectors": "persona/vectors_s23", "axes": "persona/axes_vectors_s23.pt",
                      "d": "{d}/{D}_s23.pt", "d9k": "d9k/D9K_s23.pt"},
    },
}
FORMATS = ("original", "user_stay")


def path(run, key, **kw):
    p = run[key]
    p = Path(p.format(**kw)) if isinstance(p, str) else p
    return p if p.is_absolute() else run["root"] / p


def cos(a, b):
    a, b = a.double(), b.double()
    return float((a @ b) / (a.norm() * b.norm()))


def groups():
    g = json.loads((REPO / "kd/personas/kd_role_groups.json").read_text(encoding="utf-8"))["groups"]
    return {name: v["roles"] for name, v in g.items()}


# ---- 1 role play -----------------------------------------------------------------------------
def role_play(run, grp):
    per_role = {}
    for f in sorted(path(run, "scores").glob("*.json")):
        s = list(json.loads(f.read_text(encoding="utf-8")).values())
        per_role[f.stem] = {"n": len(s), "s3": sum(v == 3 for v in s), "s23": sum(v in (2, 3) for v in s),
                            "s0": sum(v == 0 for v in s)}
    n = sum(r["n"] for r in per_role.values())
    out = {"roles_judged": len(per_role), "replies": n,
           "score3_share": sum(r["s3"] for r in per_role.values()) / n,
           "score23_share": sum(r["s23"] for r in per_role.values()) / n,
           "score0_share": sum(r["s0"] for r in per_role.values()) / n,
           "roles_ge20_score3": sum(r["s3"] >= 20 for r in per_role.values()),
           "lowest5": sorted(((r["s3"], k) for k, r in per_role.items()))[:5], "groups": {}}
    for name, roles in grp.items():
        rs = [per_role[r] for r in roles if r in per_role]
        if rs:
            out["groups"][name] = sum(r["s3"] for r in rs) / sum(r["n"] for r in rs)
    return out, per_role


# ---- 2 geometry ------------------------------------------------------------------------------
def geometry(ax):
    layers = list(ax["layers"])
    A = ax["axes"]["assistant"]
    allc = ax["pca"]["all"]["components"]                      # (K, k, H), sign-pinned
    allv = ax["pca"]["all"]["variance_ratio"]
    rows = {}
    for i, L in enumerate(layers):
        r = {"pc1_assistant": cos(allc[i, 0], A[i]), "pc1_var": float(allv[i, 0])}
        for b in BEHAVIOURS:
            r[f"{b}_pc1_assistant"] = cos(ax["pca"][b]["components"][i, 0], A[i])
        rows[L] = r
    return rows


# ---- 3 vector change -------------------------------------------------------------------------
def load_bank(d):
    out = {}
    for f in sorted(Path(d).glob("*.pt")):
        v = torch.load(f, weights_only=False)
        out[f.stem] = v["vector"].double()                     # (K, H)
    return out


def vector_change(orig, new):
    roles = sorted(set(orig) & set(new))
    O = torch.stack([orig[r] for r in roles])                  # (R, K, H)
    N = torch.stack([new[r] for r in roles])
    Oc, Nc = O - O.mean(0, keepdim=True), N - N.mean(0, keepdim=True)
    res = {"roles": len(roles), "layers": {}}
    for L in LAYERS:
        raw = [cos(O[j, L], N[j, L]) for j in range(len(roles))]
        cen = [cos(Oc[j, L], Nc[j, L]) for j in range(len(roles))]
        # baseline: a role against the OTHER roles, same format (original), centred
        other = [cos(Oc[j, L], Oc[k, L]) for j in range(len(roles)) for k in range(len(roles)) if j != k]
        res["layers"][L] = {"raw_median": statistics.median(raw), "raw_min": min(raw),
                            "centred_median": statistics.median(cen), "centred_min": min(cen),
                            "centred_lowest3": sorted(zip(cen, roles))[:3],
                            "baseline_other_roles_centred_median": statistics.median(other)}
    return res


def axis_change(ax_o, ax_n, run_o, run_n):
    out = {}
    for name in ("assistant",) + BEHAVIOURS:
        out[name] = {L: cos(ax_o["axes"][name][L], ax_n["axes"][name][L]) for L in LAYERS}
    for D, d in (("D8", "d8"), ("D9", "d9")):
        uo = torch.load(path(run_o, "d", d=d, D=D), weights_only=False)["unit"]
        un = torch.load(path(run_n, "d", d=d, D=D), weights_only=False)["unit"]
        out[D] = {L: cos(uo[L], un[L]) for L in LAYERS}
    return out


# ---- 4 D's and D9 + K ------------------------------------------------------------------------
def ds(run):
    u = {D: torch.load(path(run, "d", d=D.lower(), D=D), weights_only=False)["unit"] for D in ("D5", "D8", "D9")}
    k = torch.load(path(run, "d9k"), weights_only=False)
    kz = np.load(k["k"])                                       # the K file d9k.py read, keys L<layer> (unit)
    lay = list(k["layers"])
    out = {}
    for L in LAYERS:
        i = lay.index(L)
        sd9, sdk = float(k["sd_d9"][i]), float(k["sd_k"][i])
        out[L] = {"cos_D5_D8": cos(u["D5"][L], u["D8"][L]), "cos_D5_D9": cos(u["D5"][L], u["D9"][L]),
                  "spread_ratio_D9_over_K": sd9 / sdk,
                  "cos_D9K_K": cos(k["direction"][i], torch.from_numpy(kz[f"L{L}"])),
                  "cos_D9_K": float(k["cos_d9_k"][i])}
    return out


def main():
    grp = groups()
    res = {"banks": "s23", "layers": LAYERS, "role_play": {}, "geometry": {}, "vector_change": {}, "axis_change": {},
           "ds": {}}
    for model, runs in RUNS.items():
        for fmt in FORMATS:
            res["role_play"][f"{model} {fmt}"], _ = role_play(runs[fmt], grp)
            ax = torch.load(path(runs[fmt], "axes"), weights_only=False)
            res["geometry"][f"{model} {fmt}"] = geometry(ax)
            res["ds"][f"{model} {fmt}"] = ds(runs[fmt])
        ax_o = torch.load(path(runs["original"], "axes"), weights_only=False)
        ax_n = torch.load(path(runs["user_stay"], "axes"), weights_only=False)
        res["axis_change"][model] = axis_change(ax_o, ax_n, runs["original"], runs["user_stay"])
        res["vector_change"][model] = vector_change(load_bank(path(runs["original"], "vectors")),
                                                    load_bank(path(runs["user_stay"], "vectors")))
    (HERE / "compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (HERE / "compare.md").write_text(report(res), encoding="utf-8")
    print(report(res))


def report(r):
    cells = [f"{m} {f}" for m in RUNS for f in FORMATS]
    out = ["# user_stay matched pair -- the four comparisons (s23 bank)", "",
           "Generated by `compare.py`; every number below is in `compare.json`.", "",
           "## 1 Role play (judge Llama-3.1-70B, 35 judged roles x 150 replies)", "",
           "| | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    rp = r["role_play"]
    for key, lab in (("score3_share", "fully in role (score 3)"), ("score23_share", "score 2 or 3"),
                     ("score0_share", "score 0"), ("roles_ge20_score3", "roles with >= 20 score-3")):
        out.append(f"| {lab} | " + " | ".join(
            f"{rp[c][key]:.3f}" if isinstance(rp[c][key], float) else str(rp[c][key]) for c in cells) + " |")
    for g in rp[cells[0]]["groups"]:
        out.append(f"| group {g} | " + " | ".join(f"{rp[c]['groups'].get(g, float('nan')):.2f}" for c in cells) + " |")
    out += ["", "## 2 Geometry", ""]
    geo = r["geometry"]
    metrics = [("pc1_assistant", "cos(PC1, assistant)"), ("pc1_var", "PC1 share of variance")] + \
              [(f"{b}_pc1_assistant", f"{b} PC1 . assistant") for b in BEHAVIOURS]
    for L in LAYERS:
        out += [f"Layer {L}", "", "| | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
        for k, lab in metrics:
            out.append(f"| {lab} | " + " | ".join(f"{geo[c][L][k]:+.3f}" for c in cells) + " |")
        out.append("")
    out += ["Median over all 80 layers", "", "| | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for k, lab in metrics:
        out.append(f"| {lab} | " + " | ".join(
            f"{statistics.median(v[k] for v in geo[c].values()):+.3f}" for c in cells) + " |")
    # the matched-pair question: does the Qwen - Llama gap close under user_stay?  count layers
    out += ["", "Qwen - Llama gap, original v user_stay, layers where |gap| shrank (of 80):", ""]
    for k, lab in metrics:
        go = [geo["Qwen original"][L][k] - geo["Llama-3.3 original"][L][k] for L in geo["Qwen original"]]
        gn = [geo["Qwen user_stay"][L][k] - geo["Llama-3.3 user_stay"][L][k] for L in geo["Qwen original"]]
        shr = sum(abs(b) < abs(a) for a, b in zip(go, gn))
        out.append(f"- {lab}: {shr} / {len(go)}; median gap {statistics.median(go):+.3f} -> {statistics.median(gn):+.3f}")
    out += ["", "## 3 Vector change (original -> user_stay)", ""]
    for m, v in r["vector_change"].items():
        out.append(f"{m} ({v['roles']} roles)")
        out.append("")
        out.append("| layer | raw cos median (min) | centred cos median (min) | lowest 3 centred | baseline: other roles |")
        out.append("|---|---|---|---|---|")
        for L, x in v["layers"].items():
            low = ", ".join(f"{n.replace('pc_', '')} {c:.2f}" for c, n in x["centred_lowest3"])
            out.append(f"| {L} | {x['raw_median']:.3f} ({x['raw_min']:.3f}) | {x['centred_median']:.3f} "
                       f"({x['centred_min']:.3f}) | {low} | {x['baseline_other_roles_centred_median']:+.3f} |")
        out.append("")
    out += ["cos(axis original, axis user_stay):", "", "| model | axis | " + " | ".join(f"L{L}" for L in LAYERS) + " |",
            "|---|---|" + "---|" * len(LAYERS)]
    for m, d in r["axis_change"].items():
        for name, v in d.items():
            out.append(f"| {m} | {name} | " + " | ".join(f"{v[L]:.3f}" for L in LAYERS) + " |")
    out += ["", "## 4 D's and D9 + K", "", "| | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for L in LAYERS:
        for k, lab in (("cos_D5_D8", "cos(D5, D8)"), ("cos_D5_D9", "cos(D5, D9)"),
                       ("spread_ratio_D9_over_K", "sd(A.D9) / sd(A.K)"), ("cos_D9K_K", "cos(D9 + K, K)")):
            out.append(f"| L{L} {lab} | " + " | ".join(f"{r['ds'][c][L][k]:.3f}" for c in cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
