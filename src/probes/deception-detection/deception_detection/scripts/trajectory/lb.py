"""(b) Liars' Bench: probes on states vs on differences vs both, and D9 zero-shot on differences.

Trained arms use the thesis's raw probe (`persona_ood.make_probe`: StandardScaler, then logistic
regression with C = 0.1, no intercept) and its leave-one-dataset-out protocol
(`persona_ood.leave_one_out_scores`, naive pooling, over the six two-class datasets: the five Liars'
Bench sets and RepE as a training source), scored on each held-out Liars' Bench set:

    S[l]    the state h[l]                      (blocks 35-45)
    D[l]    the difference h[l+1] - h[l]        (ten consecutive pairs)
    SD[l]   both, [h[l], h[l+1] - h[l]]

Zero-shot arms: D9S[l] = h[l] @ unit(c[l]) and D9D[l] = (h[l'] - h[l]) @ unit(c[l'] - c[l]) on the
band pairs and on the 16-layer sweep's pairs; the trap arm Z[L]-Z[20], each layer's D9 score
standardised on Alpaca, for L = 36, 40, 44.

    python deception_detection/scripts/trajectory/lb.py gates
    python deception_detection/scripts/trajectory/lb.py fit         # resumable, one arm per file
    python deception_detection/scripts/trajectory/lb.py zeroshot
    python deception_detection/scripts/trajectory/lb.py analyse
"""

from __future__ import annotations

import json
import time

import fire
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from deception_detection.scripts import layer_sweep as LS
from deception_detection.scripts import persona_ood as P
from deception_detection.scripts.trajectory import common as C
from deception_detection.scripts.trajectory import spec as T

FIVE = ("convince_me__plain", "harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain",
        "insider_trading_cadenza__plain", "instruct_dishonesty__plain")
REPE = "repe_honesty__plain"
ORDER = list(FIVE) + [REPE]  # = persona_ood.loo's order (sorted two-class datasets, control excluded)
CONTROL = P.CONTROL_DATASET
HPKR = "harm_refusal_knowledge_report__plain"
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C", HPKR: "HP-KR",
         "insider_trading_cadenza__plain": "IT", "instruct_dishonesty__plain": "ID"}
TRAINED = T.LB_OUT / "trained"


def layer_dir(l: int):
    if l in T.BAND and l not in T.THESIS_BAND:
        return T.BAND_DIR / f"pooled_L{l}"
    return LS.pooled_dir(l)


MASKS = T.LB_OUT / "row_masks.npz"


def row_masks() -> dict[str, np.ndarray]:
    """The rows the band read covers: the emotion-coords full run's capped rows (up to 1,000 per class
    per set, 500 for IT, seeded) and every CG, HP-KR and RepE row; the other rows are NaN in the band
    files. Taken from block 35 the first time, then from `row_masks.npz`; gate G0d checks every band block
    has the same mask."""
    if MASKS.exists():
        return dict(np.load(MASKS))
    d = load_layer(T.BAND[0], masked=False)
    masks = {n: np.isfinite(z).all(1) for n, (z, _) in d.items()}
    T.LB_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(MASKS, **masks)
    return masks


def load_layer(l: int, names=tuple(ORDER), masked: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`persona_ood.load_pooled` for the named datasets only, in float32: {name: (z, y)}; `masked` keeps
    only the band read's rows (Alpaca, absent from the masks, keeps all its rows)."""
    keep_rows = row_masks() if masked else {}
    parts: dict[str, list] = {}
    for path in sorted(layer_dir(l).glob("*.npz")):
        raw = np.load(path, allow_pickle=False)
        name = str(raw["name"])
        target = P.MERGE_INTO.get(name, name)
        if target not in names:
            continue
        labels = raw["labels"].astype(str)
        keep = np.isin(labels, ["HONEST", "DECEPTIVE"])
        parts.setdefault(target, []).append((raw["pooled"][keep].astype(np.float32),
                                             (labels[keep] == "DECEPTIVE").astype(int)))
    missing = set(names) - set(parts)
    if missing:
        raise FileNotFoundError(f"layer {l}: {sorted(missing)} missing from {layer_dir(l)}")
    out = {}
    for n, p in parts.items():
        z, y = np.concatenate([z for z, _ in p]), np.concatenate([y for _, y in p])
        if n in keep_rows:
            z, y = z[keep_rows[n]], y[keep_rows[n]]
        if masked and not np.isfinite(z).all():
            raise ValueError(f"layer {l} {n}: non-finite values among the kept rows")
        out[n] = (z, y)
    return out


def loo(feats: dict[str, np.ndarray], ys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`persona_ood.leave_one_out_scores`'s naive branch for the five Liars' Bench sets held out (RepE
    trains but is never held out, so its fit is skipped; the five held-out scores are the same)."""
    out = {}
    for held in FIVE:
        sources = [n for n in ORDER if n != held]
        probe = P.make_probe()
        probe.fit(np.vstack([feats[n] for n in sources]), np.concatenate([ys[n] for n in sources]))
        out[held] = probe.decision_function(feats[held])
    return out


def _same_labels(a: dict, b: dict) -> None:
    for n in a:
        if not np.array_equal(a[n][1], b[n][1]):
            raise AssertionError(f"{n}: labels differ between layers")


def _alignment(za: np.ndarray, zb: np.ndarray, rng: np.random.Generator) -> float:
    """layer_average's check: the share of rows closer (centred cosine) to their own vector at the
    other layer than to a random row's."""
    def centred(z):
        c = z - z.mean(0)
        return c / np.linalg.norm(c, axis=1, keepdims=True)
    ca, cb = centred(za.astype(np.float64)), centred(zb.astype(np.float64))
    same = (ca * cb).sum(1)
    other = (ca * cb[rng.permutation(len(cb))]).sum(1)
    return float(np.mean(same > other))


# --- gates ---------------------------------------------------------------------------------------------

def gates() -> None:
    """G0a D9 at layer 40 on CG and HP-KR; G0d band rows line up and labels match; G0e the raw probe's
    leave-one-out at layer 40 reproduces results/loo_L40; G0f the loader equals persona_ood.load_pooled
    and the contrasts equal the bank's; the bank covers every layer used."""
    with threadpool_limits(T.BLAS_THREADS):
        from deception_detection.scripts import recall_denial as rd

        T.LOGS.mkdir(parents=True, exist_ok=True)
        report, ok = {}, True
        layers = sorted(set(T.BAND) | set(T.SWEEP))
        c = C.extended_contrasts(layers)  # raises if a layer is missing
        inp = rd.Inputs("llama")
        for l in (20, 40):
            ref = inp.bank_at(l).axes["extended"]
            err = float(np.abs(c[l] - ref).max() / np.abs(ref).max())
            report[f"G0f_contrast_L{l}_relerr"] = err
            ok &= err < 1e-12
        z40 = load_layer(40, names=tuple(ORDER) + (CONTROL,), masked=False)  # the thesis's full rows
        ref = P.load_pooled(LS.pooled_dir(40))
        for n in ORDER + [CONTROL]:
            same_y = np.array_equal(z40[n][1], ref[n].y)
            err = float(np.abs(z40[n][0].astype(np.float64) - ref[n].z).max())
            report[f"G0f_loader_{n}"] = {"labels_equal": bool(same_y), "max_abs": err}
            ok &= same_y and err < 1e-5
        del ref
        u = C.unit(c[40])
        for n, want in T.G0A_D9_L40.items():
            got = round(C.auroc(z40[n][1], z40[n][0] @ u), 4)
            report[f"G0a_D9_L40_{SHORT[n]}"] = {"got": got, "want": want}
            ok &= abs(got - want) < 1e-9
        start = time.time()
        scores = loo({n: z40[n][0] for n in ORDER}, {n: z40[n][1] for n in ORDER})
        report["G0e_fit_seconds"] = round(time.time() - start, 1)
        for n, want in T.G0E_LOO_L40.items():
            got = C.auroc(z40[n][1], scores[n])
            report[f"G0e_LOO_L40_{SHORT[n]}"] = {"got": round(got, 4), "want": want}
            ok &= abs(got - want) <= T.G0E_TOL
        del z40
        masks = row_masks()
        report["G0d_rows_kept"] = {SHORT.get(n, "RepE"): {"kept": int(m.sum()), "of": int(len(m))} for n, m in masks.items()}
        rng = np.random.default_rng(0)
        prev, align, same_mask = None, {}, True
        for l in T.BAND:
            full = load_layer(l, masked=False)
            for n in ORDER:
                finite = np.isfinite(full[n][0]).all(1)
                # a band-read block must miss exactly the masked-out rows; a thesis block misses none
                want = masks[n] if l not in T.THESIS_BAND else np.ones(len(finite), bool)
                same_mask &= bool(np.array_equal(finite, want))
            cur = {n: (full[n][0][masks[n]], full[n][1][masks[n]]) for n in ORDER}
            del full
            if prev is not None:
                _same_labels(prev, cur)
                for n in ORDER:
                    s = _alignment(prev[n][0], cur[n][0], rng)
                    align[f"{SHORT.get(n, 'RepE')}|{l - 1}->{l}"] = round(s, 4)
                    ok &= s >= T.ALIGNED
            prev = cur
        report["G0d_band_masks_consistent"] = same_mask
        ok &= same_mask
        report["G0d_alignment_min"] = min(align.values())
        report["G0d_alignment"] = align
        report["pass"] = bool(ok)
        (T.LOGS / "gates_lb.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "G0d_alignment"}, indent=1))
        if not ok:
            raise SystemExit("lb gates FAILED")


# --- trained arms --------------------------------------------------------------------------------------

def fit() -> None:
    """Every trained arm's held-out scores, one file per arm (skipped if present)."""
    C.check_freeze()
    TRAINED.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(T.BLAS_THREADS):
        log = open(T.LOGS / "fit_lb.log", "a", encoding="utf-8")
        prev = None
        for a, b in T.BAND_PAIRS:
            za = prev if prev is not None else load_layer(a)
            zb = load_layer(b)
            _same_labels(za, zb)
            ys = {n: za[n][1] for n in ORDER}
            if not (TRAINED / "labels.npz").exists():
                np.savez(TRAINED / "labels.npz", **ys)
            arms = ((f"S{a}", lambda: {n: za[n][0] for n in ORDER}),
                    (f"D{a}_{b}", lambda: {n: zb[n][0] - za[n][0] for n in ORDER}),
                    (f"SD{a}_{b}", lambda: {n: np.hstack([za[n][0], zb[n][0] - za[n][0]]) for n in ORDER}))
            if b == T.BAND[-1]:
                arms += ((f"S{b}", lambda: {n: zb[n][0] for n in ORDER}),)
            for arm, make in arms:
                path = TRAINED / f"{arm}.npz"
                if path.exists():
                    continue
                start = time.time()
                feats = make()
                scores = loo(feats, ys)
                np.savez(TRAINED / f"{arm}.partial.npz", **scores)
                (TRAINED / f"{arm}.partial.npz").replace(path)
                del feats
                line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {arm} {time.time() - start:.0f} s"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            prev = zb
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FIT DONE\n")
        log.close()


# --- zero-shot arms -------------------------------------------------------------------------------------

def zeroshot() -> None:
    """D9 on states and differences (band and sweep), Alpaca-standardised scores and the trap arms."""
    C.check_freeze()
    T.LB_OUT.mkdir(parents=True, exist_ok=True)
    names = tuple(FIVE) + (CONTROL,)
    layers = sorted(set(T.BAND) | set(T.SWEEP))
    c = C.extended_contrasts(layers)
    saved: dict[str, np.ndarray] = {}
    z_alp: dict[int, tuple[float, float]] = {}
    for l in layers:
        d = load_layer(l, names)
        u = C.unit(c[l])
        for n in names:
            saved[f"D9S{l}|{n}"] = d[n][0].astype(np.float64) @ u
        alp = saved[f"D9S{l}|{CONTROL}"]
        z_alp[l] = (float(alp.mean()), float(alp.std()))
        if l == layers[0]:
            for n in FIVE:
                saved[f"labels|{n}"] = d[n][1]
        print(f"states L{l}", flush=True)
    pairs = sorted(set(T.BAND_PAIRS) | set(T.SWEEP_PAIRS))
    for a, b in pairs:
        da, db = load_layer(a, names), load_layer(b, names)
        _same_labels(da, db)
        u = C.unit(c[b] - c[a])
        for n in FIVE:
            saved[f"D9D{a}_{b}|{n}"] = (db[n][0].astype(np.float64) - da[n][0].astype(np.float64)) @ u
        del da, db
        print(f"pair {a}->{b}", flush=True)

    def z(l, n):
        m, s = z_alp[l]
        return (saved[f"D9S{l}|{n}"] - m) / s

    for n in FIVE:
        for L in T.TRAP_LAYERS:
            saved[f"Z{L}m{T.TRAP_BASE}|{n}"] = z(L, n) - z(T.TRAP_BASE, n)
        for a, b in T.BAND_PAIRS:
            saved[f"ZD{a}_{b}|{n}"] = z(b, n) - z(a, n)
    np.savez(T.LB_OUT / "zeroshot.npz", **saved)
    print("zeroshot done", flush=True)


# --- analysis --------------------------------------------------------------------------------------------

class Draws:
    """Paired stratified bootstrap counts per dataset and per HP-KR slice (same draws for every arm)."""

    def __init__(self, labels: dict[str, np.ndarray], slices: dict[str, np.ndarray]):
        self.labels, self.slices = labels, slices
        self.w = {}
        for i, n in enumerate(FIVE):
            rng = np.random.default_rng([T.BOOT_SEED_LB, i])
            self.w[n] = C.stratified_counts(labels[n], T.N_BOOT_LB, rng)
        for j, (name, mask) in enumerate(sorted(slices.items())):
            rng = np.random.default_rng([T.BOOT_SEED_LB, 100 + j])
            self.w[name] = C.stratified_counts(labels[HPKR][mask], T.N_BOOT_LB, rng)

    def arm(self, scores: dict[str, np.ndarray]) -> dict:
        """Point AUROC and per-draw AUROC per dataset and slice; 'mean5' is the five-set mean."""
        point, draws = {}, {}
        for n in FIVE:
            y, s = self.labels[n], scores[n]
            point[n] = C.auroc(y, s)
            draws[n] = C.weighted_auroc(y, s, self.w[n])
        for name, mask in self.slices.items():
            y, s = self.labels[HPKR][mask], scores[HPKR][mask]
            point[name] = C.auroc(y, s)
            draws[name] = C.weighted_auroc(y, s, self.w[name])
        point["mean5"] = float(np.mean([point[n] for n in FIVE]))
        draws["mean5"] = np.mean(np.stack([draws[n] for n in FIVE]), axis=0)
        return {"point": point, "draws": draws}


def _gain(arms: dict, cand: str, comp: tuple[str, str], key: str = "mean5") -> tuple[float, np.ndarray, str]:
    """cand - max(comp) on `key`: the point gain, the per-draw gains, and the comparator with the higher
    point value (for the wording gate)."""
    a, b = (arms[c] for c in comp)
    pt = arms[cand]["point"][key] - max(a["point"][key], b["point"][key])
    dr = arms[cand]["draws"][key] - np.maximum(a["draws"][key], b["draws"][key])
    better = comp[0] if a["point"][key] >= b["point"][key] else comp[1]
    return pt, dr, better


def _rule(arms: dict, cand_fmt: str, comp_fmt: str, pairs) -> dict:
    pts, drs, gate_pt, gate_dr = [], [], [], []
    for a, b in pairs:
        cand = cand_fmt.format(a=a, b=b)
        pt, dr, better = _gain(arms, cand, (comp_fmt.format(l=a), comp_fmt.format(l=b)))
        pts.append(pt)
        drs.append(dr)
        gate_pt.append(arms[cand]["point"]["identical text"] - arms[better]["point"]["identical text"])
        gate_dr.append(arms[cand]["draws"]["identical text"] - arms[better]["draws"]["identical text"])
    per = np.mean(np.stack(drs), axis=0)
    lo, hi = np.percentile(per, [2.5, 97.5])
    gper = np.mean(np.stack(gate_dr), axis=0)
    glo, ghi = np.percentile(gper, [2.5, 97.5])
    gain = float(np.mean(pts))
    superior = gain >= T.MIN_GAIN and lo > 0
    wording_ok = ghi >= 0
    return {"gain": gain, "gain_ci": [float(lo), float(hi)], "superior": bool(superior),
            "identical_text_gap": float(np.mean(gate_pt)), "identical_text_gap_ci": [float(glo), float(ghi)],
            "wording_gate_passes": bool(wording_ok), "holds": bool(superior and wording_ok),
            "per_pair_gain": [round(p, 4) for p in pts]}


def analyse() -> None:
    C.check_freeze()
    from deception_detection.scripts import recall_denial as rd

    T.RESULTS.mkdir(parents=True, exist_ok=True)
    zs = dict(np.load(T.LB_OUT / "zeroshot.npz"))
    labels = {n: zs[f"labels|{n}"] for n in FIVE}
    tl = dict(np.load(TRAINED / "labels.npz"))
    for n in FIVE:
        if not np.array_equal(tl[n], labels[n]):
            raise AssertionError(f"{n}: trained and zero-shot labels differ")
    frame, slices = rd.Inputs("llama").hpkr()
    if len(frame) != len(labels[HPKR]) or not np.array_equal(frame["lie"].to_numpy() == 1, labels[HPKR] == 1):
        raise AssertionError("HP-KR frame does not line up with the pooled rows")
    slices = {k: v for k, v in slices.items() if k in ("identical text", "denials")}
    draws = Draws(labels, slices)
    arms: dict[str, dict] = {}
    for path in sorted(TRAINED.glob("*.npz")):
        if path.stem == "labels" or path.stem.endswith(".partial"):
            continue
        sc = dict(np.load(path))
        arms[path.stem] = draws.arm({n: sc[n] for n in FIVE})
    for key in sorted({k.split("|")[0] for k in zs if not k.startswith("labels|")}):
        arms[key] = draws.arm({n: zs[f"{key}|{n}"] for n in FIVE})
    want = [f"S{l}" for l in T.BAND] + [f"D{a}_{b}" for a, b in T.BAND_PAIRS] + [f"SD{a}_{b}" for a, b in T.BAND_PAIRS]
    missing = [w for w in want if w not in arms]
    if missing:
        raise SystemExit(f"trained arms missing: {missing}")
    rules = {
        "B1_difference_vs_state": _rule(arms, "D{a}_{b}", "S{l}", T.BAND_PAIRS),
        "B2_state_plus_difference_vs_state": _rule(arms, "SD{a}_{b}", "S{l}", T.BAND_PAIRS),
        "B3_D9_difference_vs_D9_state": _rule(arms, "D9D{a}_{b}", "D9S{l}", T.BAND_PAIRS),
    }
    # descriptive splits of B1-B3 by pass
    cross = tuple(p for p in T.BAND_PAIRS if p not in T.WITHIN_PASS_PAIRS)
    for name, (cf, sf) in {"B1": ("D{a}_{b}", "S{l}"), "B2": ("SD{a}_{b}", "S{l}"),
                           "B3": ("D9D{a}_{b}", "D9S{l}")}.items():
        rules[f"{name}_within_pass (descriptive)"] = _rule(arms, cf, sf, T.WITHIN_PASS_PAIRS)
        rules[f"{name}_cross_pass (descriptive)"] = _rule(arms, cf, sf, cross)
    trap = {}
    for L in T.TRAP_LAYERS:
        cand, comp = arms[f"Z{L}m{T.TRAP_BASE}"], arms[f"D9S{L}"]
        row = {}
        for key in (HPKR, "identical text", "denials", "mean5"):
            d = cand["draws"][key] - comp["draws"][key]
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[key] = {"gap": round(cand["point"][key] - comp["point"][key], 4), "ci": [round(lo, 4), round(hi, 4)]}
        row["as_predicted"] = bool(row[HPKR]["ci"][0] > 0 and row["identical text"]["ci"][0] <= 0)
        trap[f"Z{L}-Z{T.TRAP_BASE} vs D9S{L}"] = row
    table = []
    for name, r in arms.items():
        row = {"arm": name} | {SHORT.get(k, k): round(v, 4) for k, v in r["point"].items()}
        lo, hi = np.percentile(r["draws"]["mean5"], [2.5, 97.5])
        row["mean5_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        table.append(row)
    pd.DataFrame(table).to_csv(T.RESULTS / "lb_arms.csv", index=False)
    out = {"rules": rules, "trap": trap, "n_boot": T.N_BOOT_LB,
           "slices_n": {k: {"lies": int((labels[HPKR][m] == 1).sum()), "honest": int((labels[HPKR][m] == 0).sum())}
                        for k, m in slices.items()}}
    (T.RESULTS / "lb_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    fire.Fire({"gates": gates, "fit": fit, "zeroshot": zeroshot, "analyse": analyse})
