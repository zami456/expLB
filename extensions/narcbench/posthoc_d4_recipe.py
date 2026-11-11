"""Post hoc (2026-10-23, after the NARCBench held-out and the Verdict scores were read): D4 trained with their recipe.

D4 is a probe confined to the top 3 extended persona PCs (`posthoc_d4.py`). Here the confinement is kept and
everything else is their recipe (`score.fit_rose`, the steps of their `train_deception_probe`): their training rows
(Core development collusion runs, discussion rows, colluder vs honest), their end-of-turn token at their layers 32-37
(each layer with the top 3 persona PCs fitted at that layer), honest rows subsampled to balance, a StandardScaler and
logistic regression (C=25, l2, liblinear, max_iter=8), and per agent the mean over the layers of the standardised
projection. Nothing here is in either pre-registration and no Holm family changes. Fixed before any score of these
detectors:

- "D4 (their recipe)": as above; Core agents scored by the fold that held their domain out, the held-out tiers and
  Verdict by the all-Core fit;
- "D4 (their recipe, L40 span)": the same classifier on D4's own readout (layer-40 content mean, D4's basis), so it
  differs from D4 (Core) only in the classifier;
- NARCBench: `score._tier_block` unchanged on Transfer, Core dev and Stego redacted, plus their exact final Stego row
  for the end-of-turn detector; Verdict: `verdict_score.readouts`, Peak Suspicion over the two agents and the bootstrap
  over runs, unchanged;
- tests (unadjusted): Transfer ORIG and STRIP, and Verdict COMM and REFL, against the random band of the detector's
  space (their standardised end-of-turn space; the thesis layer-40 space for the L40 variant), and ΔAUROC against
  rose_exact; a band of random directions inside the 3-PC subspace (the detector's own scaler) is descriptive.

`build` fits both, checks that the same rows and code reproduce the frozen fit of their probe (`stage1_fit.npz`) and
that the layer-40 basis is D4's, and logs the hashes to `freeze.txt` before `narcbench` or `verdict` reads a row.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from extensions.narcbench import freeze as FZ
from extensions.narcbench import posthoc_d4 as PD
from extensions.narcbench import rose as RO
from extensions.narcbench import rows as R
from extensions.narcbench import score as SC
from extensions.narcbench import spec as S

L40 = S.HEADLINE_LAYER
LAYERS = list(S.ROSE_LLAMA_LAYERS)
RECIPE, SPAN40 = "D4 (their recipe)", "D4 (their recipe, L40 span)"
DETS = ("D9+K", "D9", "D4 (LB)", "D4 (Core)", RECIPE, SPAN40)
CORE = "released__orig__core"
OUT_NPZ = S.FROZEN / "posthoc_d4_recipe.npz"
CHECK = S.LOGS / "posthoc_d4_recipe_check.json"
MARK = "== POST HOC D4, THEIR RECIPE"
OUT_NARC = "posthoc_d4_recipe.json"
OUT_VERDICT = "posthoc_d4_recipe_verdict.json"


def basis_at(inp, layer: int) -> np.ndarray:
    """D4's basis at `layer`: the top 3 extended PCs of the 25 contrast personas, as `posthoc_d4.build` fits it."""
    from deception_detection import persona_pcs as pcs

    geom = pcs.fit_geometry(inp.bank_at(layer), k=3, deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                            honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
    return geom.components[:3].astype(np.float64)


def fit_in(X: np.ndarray, y: np.ndarray, basis: np.ndarray) -> tuple:
    """Their recipe on the rows projected onto the PCs: (mean, scale, unit w), all in the subspace."""
    return SC.fit_rose(X @ basis.T, y, hp=RO.HP())


def project(x: np.ndarray, basis: np.ndarray, p: tuple) -> float:
    m, s, w = p
    return float(((x @ basis.T - m) / s) @ w)


_SUB: dict[tuple[str, int, int], np.ndarray] = {}


def _sub_dirs(det: str, L: int, n: int) -> np.ndarray:
    if (det, L, n) not in _SUB:
        U = np.random.default_rng([S.RANDOM_SEED, 40_000 + L + (100 if det == SPAN40 else 0)]).standard_normal((n, 3))
        _SUB[(det, L, n)] = U / np.linalg.norm(U, axis=1, keepdims=True)
    return _SUB[(det, L, n)]


def build() -> None:
    from experiments.knowledge import recall_denial as rd

    inp = rd.Inputs("llama")
    assert inp.thesis, "not the thesis bank"
    dev = set(json.loads((S.ROWS / "runs_core_dev.json").read_text()))
    meta, frozen_fit = SC.cond_meta(CORE), SC.load_fit()
    out = {f"basis|L{L}": basis_at(inp, L) for L in LAYERS + [L40]}
    check = {"L40 basis is D4's": bool(np.allclose(out[f"basis|L{L40}"], np.load(PD.OUT_NPZ)["basis"], rtol=0, atol=1e-12))}
    rows = {}
    for variant, span, layers, det in (("rose_exact", "eot", LAYERS, RECIPE), ("rose_L40", "content", [L40], SPAN40)):
        for L in layers:
            X, y, dom = SC.rose_rows(CORE, meta, span, L, dev)
            # their probe refitted on these rows must be the frozen one: same rows, same code
            check[f"{variant} L{L} refit equals stage1_fit"] = bool(all(
                np.allclose(a, b) for a, b in zip(SC.fit_rose(X, y, hp=RO.HP()), frozen_fit[(variant, L)])))
            B = out[f"basis|L{L}"]
            for k, arr in zip(("mean", "scale", "w"), fit_in(X, y, B)):
                out[f"{det}|L{L}|all|{k}"] = arr
            for d in sorted(set(dom)):
                keep = np.array([x != d for x in dom])
                for k, arr in zip(("mean", "scale", "w"), fit_in(X[keep], y[keep], B)):
                    out[f"{det}|L{L}|fold|{d}|{k}"] = arr
            rows[f"{det} L{L}"] = {"rows": len(y), "colluder": int(y.sum()), "domains": len(set(dom))}
            del X
    np.savez(OUT_NPZ, **out)
    ok = all(check.values())
    CHECK.write_text(json.dumps({"pass": ok, "checks": check, "training_rows": rows}, indent=1))
    scripts = S.DD / "deception_detection" / "scripts"
    thesis = {"persona_pcs.py": S.DD / "deception_detection" / "persona_pcs.py"} | {
        m: scripts / m for m in ("persona_ood.py", "recall_denial.py", "lb_models.py")}
    lines = [f"{MARK} (declared before any score of these detectors)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"posthoc_d4_recipe.py sha256 {FZ.sha256(Path(__file__))}",
             f"posthoc_d4_recipe.npz sha256 {FZ.sha256(OUT_NPZ)}",
             f"score.py sha256 {FZ.sha256(Path(SC.__file__))} (unchanged since freeze 2)",
             f"posthoc_d4.npz sha256 {FZ.sha256(PD.OUT_NPZ)}",
             f"stage1_fit.npz sha256 {FZ.sha256(SC.FIT)}",
             "detectors: D4 (their recipe) = their rows, end-of-turn token, layers 32-37 (per-layer top-3 extended persona "
             "PCs), balanced, StandardScaler + LR(C=25, l2, liblinear, max_iter=8), mean over layers; D4 (their recipe, "
             "L40 span) = the same classifier on the L40 content mean in D4's basis",
             "tests (post hoc, unadjusted): Transfer ORIG and STRIP and Verdict COMM and REFL against the band of the "
             "detector's space; ΔAUROC vs rose_exact; the band inside the 3-PC subspace descriptive",
             f"checks: {json.dumps(check)}"]
    lines += [f"thesis module {m} sha256 {FZ.sha256(p)}" for m, p in thesis.items()]
    lines.append("uncommitted diff in those modules: " + (FZ._git("diff", "--stat", "--", *[
        str(p.relative_to(S.DD)).replace("\\", "/") for p in thesis.values()]).replace("\n", " | ") or "none"))
    lines.append("")
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(rows, indent=1))
    if not ok:
        raise SystemExit("a build check failed: " + json.dumps({k: v for k, v in check.items() if not v}))


def _check_declared() -> None:
    text = (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig")
    if MARK not in text:
        raise SystemExit("run `build` first")
    block = text.split(MARK)[-1].split("\n== ")[0]
    want = {line.split(" sha256 ")[0]: line.split(" sha256 ")[1].split()[0] for line in block.splitlines()
            if line.startswith(("posthoc_d4_recipe.py", "posthoc_d4_recipe.npz"))}
    cur = {"posthoc_d4_recipe.py": FZ.sha256(Path(__file__)), "posthoc_d4_recipe.npz": FZ.sha256(OUT_NPZ)}
    if want != cur:
        raise SystemExit(f"changed since the post hoc declaration: {want} vs {cur}")
    if not json.loads(CHECK.read_text())["pass"]:
        raise SystemExit("the build checks did not pass")


class RecipeFrozen(PD.D4Frozen):
    """The frozen thesis vectors, D4 and the two recipe detectors; a Core agent carrying its run id is scored by the
    fold that held its domain out, everything else by the all-Core fit."""

    def __init__(self):
        super().__init__()
        self.r = dict(np.load(OUT_NPZ))

    def _p(self, det: str, L: int, vec: dict) -> tuple:
        fold = f"fold|{RO._domain_from_scenario_id(vec['run'].split('__')[1])}" if "run" in vec else "all"
        return tuple(self.r[f"{det}|L{L}|{fold}|{k}"] for k in ("mean", "scale", "w"))

    def score(self, det: str, layer: int, vec: dict) -> float:
        if det == RECIPE:
            return float(np.mean([project(vec["eot"][SC.EL.index(L)], self.r[f"basis|L{L}"], self._p(det, L, vec))
                                  for L in LAYERS]))
        if det == SPAN40:
            return project(vec["content"][SC.CL.index(L40)], self.r[f"basis|L{L40}"], self._p(det, L40, vec))
        return super().score(det, layer, vec)

    def subspace_random(self, det: str, vec: dict, n: int = S.N_RANDOM) -> np.ndarray:
        """Random unit directions inside the 3-PC subspace, through the detector's own all-Core scaler."""
        layers, span, idx = (LAYERS, "eot", SC.EL) if det == RECIPE else ([L40], "content", SC.CL)
        out = np.zeros(n)
        for L in layers:
            m, s, _ = self._p(det, L, {})
            out += _sub_dirs(det, L, n) @ ((vec[span][idx.index(L)] @ self.r[f"basis|L{L}"].T - m) / s)
        return out / len(layers)


def _sub_nulls(vt: dict, fz: RecipeFrozen, runs: dict) -> dict:
    """The subspace band on one tier, built as `_tier_block` builds its random bands (ORIG, and STRIP mixed in)."""
    scen = sorted({SC.scenario(r) for r in runs if SC.mode(r) in ("collusion", "control")})
    col, ctl = [f"{s}__collusion" for s in scen], [f"{s}__control" for s in scen]
    out = {}
    for det in (RECIPE, SPAN40):
        mats = {c: {k: fz.subspace_random(det, v) for k, v in vt[c].items()} for c in ("orig", "strip")}
        po = SC._peaks(mats["orig"])
        out[det] = {"orig": SC.null_aurocs(po, po, col, ctl),
                    "strip": SC.null_aurocs(SC._peaks(SC._mixed(mats["orig"], mats["strip"], col)), po, col, ctl)}
    return out


def _band(obs: float, null: np.ndarray) -> dict:
    return dict(zip(("band97.5", "p"), SC.band_p(obs, null)))


def narcbench() -> None:
    SC.check_freeze2()
    _check_declared()
    SC.THESIS = DETS  # `detector_scores` scores these through `RecipeFrozen.score`
    fz, params = RecipeFrozen(), SC.load_fit()
    fz.z = {k: fz.z[k] for k in fz.z.files if k.startswith(f"L{L40}|")}  # every tier block reads layer 40 only
    res = {"note": "POST HOC: not pre-registered; p-values unadjusted; Peak Suspicion; D4 (their recipe) on the "
                   "end-of-turn token, layers 32-37; D4 (their recipe, L40 span) on the layer-40 content mean."}
    rt = R.load_runs("transfer")
    vt = {c: SC.agents(f"released__{c}__transfer", "transfer") for c in ("orig", "strip", "plant", "bare")}
    tb = SC._tier_block("transfer", vt, fz, params, rt, {}, with_nulls=True)
    sub = _sub_nulls(vt, fz, rt)
    del vt
    nulls, boots, dets = tb.pop("_null_arrays"), tb.pop("_boots"), tb["detectors"]
    tests = {}
    for det, space in ((RECIPE, "rose_space"), (SPAN40, "thesis_L40")):
        t = {"space": space, "orig_vs_band": _band(dets[det]["auroc_peak"], nulls[space]["orig"]),
             "strip_vs_band": _band(dets[det]["T_strip_vs_control"], nulls[space]["strip"]),
             "orig_vs_subspace_band": _band(dets[det]["auroc_peak"], sub[det]["orig"]),
             "strip_vs_subspace_band": _band(dets[det]["T_strip_vs_control"], sub[det]["strip"])}
        for ref in ("rose_exact", "D4 (Core)", "D4 (LB)"):
            t[f"delta_vs_{ref}"] = {"point": dets[det]["auroc_peak"] - dets[ref]["auroc_peak"],
                                    "ci": [float(x) for x in np.percentile(boots[det] - boots[ref], [2.5, 97.5])]}
        tests[det] = t
    res["transfer"] = {"detectors": PD._slim(tb), "tests": tests, "null_thesis_L40": tb["null_thesis_L40"],
                       "null_rose_space": tb["null_rose_space"]}
    dev = set(json.loads((S.ROWS / "runs_core_dev.json").read_text()))
    rc = {k: v for k, v in R.load_runs("core").items() if k in dev}
    vc = {c: {k: v | {"run": k[0]} for k, v in SC.agents(f"released__{c}__core", "core").items() if k[0] in dev}
          for c in ("orig", "strip", "plant", "bare", "observer")}
    cb = SC._tier_block("core", vc, fz, params, rc, {}, with_nulls=False)
    del vc
    cb.pop("_boots")
    res["core_dev"] = {"detectors": PD._slim(cb), "note": "D4 (Core) and both recipe detectors leave-domain-out; "
                                                          "their probe (rose_*) here is the all-Core fit, in-sample"}
    rs_ = R.load_runs("stego")
    vs = {c: SC.agents(f"redacted__{c}__stego", "stego") for c in ("orig", "strip", "plant", "bare")}
    sb = SC._tier_block("stego", vs, fz, params, rs_, {}, with_nulls=False)
    del vs
    sb.pop("_boots")
    ex = SC.agents("redacted__rose_exact__stego", "stego", final_only=True)
    exact = {}
    for det in ("rose_exact", RECIPE):
        o = SC.run_scores({k: SC.rose_score(params, det, v) if det == "rose_exact" else fz.score(det, L40, v)
                           for k, v in ex.items()})
        col = [r for r in o if SC.mode(r) == "collusion"]
        ctl = [r for r in o if SC.mode(r) == "control"]
        exact[det] = SC.auroc([o[r]["peak_suspicion"] for r in col], [o[r]["peak_suspicion"] for r in ctl])
    res["stego_redacted"] = {"detectors": PD._slim(sb), "their_rows_empty_final_auroc_peak": exact}
    (S.RESULTS / OUT_NARC).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for tier in ("transfer", "core_dev", "stego_redacted"):
        print(f"\n{tier}")
        for det, d in res[tier]["detectors"].items():
            dec = d.get("decomposition") or {}
            print(f"{det:>28}  AUROC {d['auroc_peak']:.3f} [{d['auroc_peak_ci'][0]:.2f}, {d['auroc_peak_ci'][1]:.2f}]  "
                  f"STRIP {d.get('T_strip_vs_control', float('nan')):.3f}  PLANT {d.get('C_plant_vs_control', float('nan')):.3f}  "
                  f"BARE {d.get('bare_auroc_peak', float('nan')):.3f}  share {dec.get('share') or float('nan'):.2f} "
                  f"{dec.get('share_ci')} {dec.get('verdict')}")
    print(json.dumps(tests, indent=1), json.dumps(exact))


def verdict() -> None:
    from extensions.narcbench import verdict_score as VS
    from extensions.narcbench import verdict_spec as V

    VS.check_freeze()
    _check_declared()
    labels = VS.load_labels(V.ROWS / "labels.json")
    fz, params = RecipeFrozen(), SC.load_fit()
    fz.z = {k: fz.z[k] for k in fz.z.files if k.startswith(f"L{L40}|")}  # read once, as `verdict_score.llama`
    scores, nulls = defaultdict(dict), defaultdict(dict)
    for name in VS.LLAMA_FILES:
        for f in sorted((V.ACTS / "llama" / name).glob("*.npz")):
            with np.load(f) as z:
                rows, content, points = z["rows"], z["content"], z["points"]
            for i, rid in enumerate(rows):
                base = int(rid) % 10_000
                for ro, vec in VS.readouts(name, content[i], points[i]).items():
                    if vec["eot"] is not None:
                        scores[(ro, RECIPE)][base] = fz.score(RECIPE, L40, vec)
                        scores[(ro, "rose_exact")][base] = SC.rose_score(params, "rose_exact", vec)
                        nulls[(ro, RECIPE, "space")][base] = SC.rose_random_matrix(params, vec)
                        nulls[(ro, RECIPE, "subspace")][base] = fz.subspace_random(RECIPE, vec)
                    if vec["content"] is not None:
                        scores[(ro, SPAN40)][base] = fz.score(SPAN40, L40, vec)
                        nulls[(ro, SPAN40, "space")][base] = fz.matrix("random", L40, vec)
                        nulls[(ro, SPAN40, "subspace")][base] = fz.subspace_random(SPAN40, vec)
    saved = json.loads((V.RESULTS / VS.OUT_LLAMA).read_text(encoding="utf-8"))
    draws = VS.run_draws()
    ep = {k: VS.pairs(v) for k, v in scores.items()}
    boots = {k: VS.boot_auroc(v, labels["shi"], draws) for k, v in ep.items()}
    res = {"note": "POST HOC: not pre-registered; unadjusted; episode = Peak Suspicion over the two agents; AUROC "
                   "against Shi's collusion (Rose's label alongside); all-Core fits.", "readouts": {}, "checks": {}}
    for (ro, det), pe in sorted(ep.items()):
        if det == "rose_exact":
            # the recomputed reference equals the frozen Verdict result
            res["checks"][f"rose_exact {ro} equals verdict_llama.json"] = bool(np.isclose(
                VS.auroc(pe, labels["shi"]), saved["readouts"][ro]["rose_exact"]["auroc_shi"]))
            continue
        e = {"auroc_shi": VS.auroc(pe, labels["shi"]), "auroc_rose": VS.auroc(pe, labels["rose"]),
             "auroc_shi_split": VS.auroc(pe, labels["shi"], "split"), "auroc_shi_ci": VS.ci(boots[(ro, det)]),
             "by_episode": [VS.auroc({k: v for k, v in pe.items() if k[1] == i}, labels["shi"])
                            for i in range(V.N_EPISODES)]}
        for kind in ("space", "subspace"):
            e[f"{kind}_band"] = _band(e["auroc_shi"], VS.null_band(VS.pair_matrix(nulls[(ro, det, kind)]), labels["shi"]))
        if (ro, "rose_exact") in boots:
            e["delta_vs_rose_exact"] = {"point": e["auroc_shi"] - saved["readouts"][ro]["rose_exact"]["auroc_shi"],
                                        "ci": VS.ci(boots[(ro, det)] - boots[(ro, "rose_exact")])}
        if ro in ("COMM", "REFL", "PRE", "BARE_COMM", "BARE_REFL"):
            e["relaxation_agent_auroc"] = VS.agent_auroc(scores[(ro, det)], labels["relaxation"])
        res["readouts"].setdefault(ro, {})[det] = e
    for ro in res["readouts"]:
        band = VS.null_band(VS.pair_matrix(nulls[(ro, RECIPE, "space")]), labels["shi"]) if (ro, RECIPE, "space") in nulls else None
        if band is not None:  # the random band of their space, recomputed, equals the frozen one
            res["checks"][f"rose_random band {ro} equals verdict_llama.json"] = bool(np.isclose(
                np.percentile(band, 97.5), saved["nulls"][ro]["rose_random"]["band97.5"]))
    res["reference"] = {ro: {d: saved["readouts"][ro][d]["auroc_shi"] for d in ("rose_exact", "D4 (Core)", "D4 (LB)", "D9+K")
                             if d in saved["readouts"][ro]} for ro in saved["readouts"]}
    (V.RESULTS / OUT_VERDICT).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["checks"], indent=1))
    for ro, d in res["readouts"].items():
        for det, e in d.items():
            print(f"{ro:>9} {det:>28}: AUROC {e['auroc_shi']:.3f} {[round(x, 2) for x in e['auroc_shi_ci']]} "
                  f"(Rose {e['auroc_rose']:.2f})  band {e['space_band']['band97.5']:.3f} p {e['space_band']['p']:.4f}  "
                  f"subspace band {e['subspace_band']['band97.5']:.3f} p {e['subspace_band']['p']:.4f}  "
                  f"Δ vs rose_exact {e.get('delta_vs_rose_exact', {}).get('point', float('nan')):+.3f} "
                  f"{[round(x, 2) for x in e.get('delta_vs_rose_exact', {}).get('ci', [])]}")


if __name__ == "__main__":
    fire.Fire({"build": build, "narcbench": narcbench, "verdict": verdict})
