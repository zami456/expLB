"""Post hoc (2026-10-21, after the held-out scores were read): D4 on NARCBench.

D4 is the thesis's probe in the persona subspace (`transfer_alignment.trained`, basis = the top 3 extended
PCs of the 25 contrast personas): `persona_ood.make_probe()` fitted on labelled rows projected onto the 3 PCs,
read back as a direction in activation space (basis.T @ coef / sigma). Nothing here is in the pre-registration
and no Holm family changes; the report marks every number post hoc. Fixed before any D4 score existed:

- layer 40, content-span mean, Peak Suspicion, the Arm B reads, conditions and bootstrap draws as `score.py`;
- headline D4 (LB): trained on Llama's five Liars' Bench sets pooled (no collusion data);
- D4 [set]: the thesis's six single-set D4s (its transfer-matrix rows), descriptive;
- D4 (Core): trained on the Core development rows their probe trains on (discussion rows, colluder vs honest),
  leave-domain-out for Core scores, the all-Core fit for the held-out tiers;
- tests (unadjusted, post hoc): Transfer ORIG and STRIP AUROC against the frozen thesis-space random band.

`build` fits the directions, checks the single-set D4s reproduce the thesis's L40 transfer matrix
(`results/persona_ood_ext_L40/matrix_ext_pc3.csv`, off-diagonal cells), and logs the file hashes to
`freeze.txt` before `score` reads any NARCBench row with them. `score.py` is reused unchanged.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from extensions.narcbench import freeze as FZ
from extensions.narcbench import rose as RO
from extensions.narcbench import rows as R
from extensions.narcbench import score as SC
from extensions.narcbench import spec as S

LAYER = S.HEADLINE_LAYER
OUT_NPZ = S.FROZEN / "posthoc_d4.npz"
MATRIX = S.DD / "results" / f"persona_ood_ext_L{LAYER}" / "matrix_ext_pc3.csv"
SINGLE = {"CG": "convince_me__plain", "HP-C": "harm_refusal_choice__plain",
          "HP-KR": "harm_refusal_knowledge_report__plain", "IT": "insider_trading_cadenza__plain",
          "ID": "instruct_dishonesty__plain", "RepE": "repe_honesty__plain"}
DETS = ("D9+K", "D9", "D4 (LB)", "D4 (Core)") + tuple(f"D4 [{s}]" for s in SINGLE)


def _fit(x: np.ndarray, y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    from experiments.persona import persona_ood as P

    probe = P.make_probe().fit(x @ basis.T, y)
    return basis.T @ (probe[-1].coef_.ravel() / probe[0].scale_)


def build() -> None:
    from deception_detection import persona_pcs as pcs
    from experiments.knowledge import recall_denial as rd
    from experiments.knowledge.k_robustness import FIVE

    inp = rd.Inputs("llama")
    assert inp.thesis, "not the thesis bank"
    geom = pcs.fit_geometry(inp.bank_at(LAYER), k=3, deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                            honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
    basis = geom.components[:3].astype(np.float64)
    pooled = inp.pooled(LAYER)
    out = {"basis": basis}
    with open(MATRIX, encoding="utf-8") as f:
        table = {r[""]: r for r in csv.DictReader(f)}
    check = {}
    for short, name in SINGLE.items():
        w = _fit(pooled[name].z.astype(np.float64), pooled[name].y.astype(int), basis)
        out[f"D4 [{short}]"] = w
        for b in SINGLE.values():
            if b == name:
                continue
            got = round(FZ._auroc(pooled[b].y.astype(int), pooled[b].z @ w), 4)
            check[f"{short}->{b}"] = {"got": got, "thesis": float(table[name][b]), "ok": abs(got - float(table[name][b])) <= 1e-4}
    out["D4 (LB)"] = _fit(np.concatenate([pooled[n].z for n in FIVE]).astype(np.float64),
                          np.concatenate([pooled[n].y.astype(int) for n in FIVE]), basis)
    del pooled
    dev = set(json.loads((S.ROWS / "runs_core_dev.json").read_text()))
    X, y, dom = SC.rose_rows("released__orig__core", SC.cond_meta("released__orig__core"), "content", LAYER, dev)
    out["D4 (Core)"] = _fit(X, y, basis)
    for d in sorted(set(dom)):
        keep = np.array([x != d for x in dom])
        out[f"D4 (Core)|fold|{d}"] = _fit(X[keep], y[keep], basis)
    np.savez(OUT_NPZ, **out)
    ok = all(v["ok"] for v in check.values())
    (S.LOGS / "posthoc_d4_check.json").write_text(json.dumps({"pass": ok, "cells": check}, indent=1))
    scripts = S.DD / "deception_detection" / "scripts"
    lines = [f"== POST HOC D4 (declared before any D4 score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"posthoc_d4.py sha256 {FZ.sha256(Path(__file__))}",
             f"posthoc_d4.npz sha256 {FZ.sha256(OUT_NPZ)}",
             f"score.py sha256 {FZ.sha256(Path(SC.__file__))} (unchanged since freeze 2)",
             f"thesis transfer matrix reproduced at L{LAYER} ({len(check)} cells): {ok}"]
    lines += [f"thesis module {m} sha256 {FZ.sha256(scripts / m)}" for m in ("persona_ood.py", "recall_denial.py", "lb_models.py")]
    lines.append("uncommitted diff in those modules: " + (FZ._git("diff", "--stat", "--", *[f"deception_detection/scripts/{m}" for m in
                 ("persona_ood.py", "recall_denial.py", "lb_models.py")]).replace("\n", " | ") or "none"))
    lines.append("")
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        raise SystemExit("D4 does not reproduce the thesis matrix: " + json.dumps({k: v for k, v in check.items() if not v["ok"]}))


class D4Frozen(SC.Frozen):
    """The frozen thesis vectors plus the post hoc D4 directions. D4 (Core) scores a Core agent with the fold
    that held its domain out when the vector carries its run id."""

    def __init__(self):
        super().__init__()
        self.d4 = dict(np.load(OUT_NPZ))

    def score(self, det: str, layer: int, vec: dict) -> float:
        if not det.startswith("D4"):
            return super().score(det, layer, vec)
        assert layer == LAYER
        key = det
        if det == "D4 (Core)" and "run" in vec:
            key = f"D4 (Core)|fold|{RO._domain_from_scenario_id(vec['run'].split('__')[1])}"
        return float(vec["content"][SC.CL.index(layer)] @ self.d4[key])


def _slim(blk: dict) -> dict:
    keep = ("auroc_peak", "auroc_peak_ci", "auroc_split", "within_run_rank", "T_strip_vs_control", "C_plant_vs_control",
            "bare_auroc_peak", "auroc_implicit_vs_control", "auroc_collusion_vs_implicit", "per_family_peak", "observer_shift_mean")
    out = {}
    for det, d in blk["detectors"].items():
        e = {k: d[k] for k in keep if k in d}
        if "decomposition" in d:
            e["decomposition"] = {k: d["decomposition"].get(k) for k in ("n", "E_total", "E_ctx", "E_txt", "share", "share_ci", "verdict")}
        out[det] = e
    return out


def score() -> None:
    SC.check_freeze2()
    if "== POST HOC D4" not in (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig"):
        raise SystemExit("run `build` first")
    if not json.loads((S.LOGS / "posthoc_d4_check.json").read_text())["pass"]:
        raise SystemExit("D4 check did not pass")
    SC.THESIS = DETS  # `detector_scores` scores these through `D4Frozen.score`
    fz, params = D4Frozen(), SC.load_fit()
    res = {"note": "POST HOC: not pre-registered; p-values unadjusted; layer 40 content mean, Peak Suspicion."}
    rt = R.load_runs("transfer")
    vt = {c: SC.agents(f"released__{c}__transfer", "transfer") for c in ("orig", "strip", "plant", "bare")}
    tb = SC._tier_block("transfer", vt, fz, params, rt, {}, with_nulls=True)
    del vt
    nulls, boots, dets = tb.pop("_null_arrays")["thesis_L40"], tb.pop("_boots"), tb["detectors"]
    tests = {}
    for det in ("D4 (LB)", "D4 (Core)"):
        tests[det] = {"orig_vs_band": dict(zip(("band97.5", "p"), SC.band_p(dets[det]["auroc_peak"], nulls["orig"]))),
                      "strip_vs_band": dict(zip(("band97.5", "p"), SC.band_p(dets[det]["T_strip_vs_control"], nulls["strip"])))}
        for ref in ("rose_exact", "D9+K"):
            delta = boots[det] - boots[ref]
            tests[det][f"delta_vs_{ref}"] = {"point": dets[det]["auroc_peak"] - dets[ref]["auroc_peak"],
                                             "ci": [float(x) for x in np.percentile(delta, [2.5, 97.5])]}
    res["transfer"] = {"detectors": _slim(tb), "tests": tests,
                       "null_thesis_L40": tb["null_thesis_L40"], "null_rose_space": tb["null_rose_space"]}
    dev = set(json.loads((S.ROWS / "runs_core_dev.json").read_text()))
    rc = {k: v for k, v in R.load_runs("core").items() if k in dev}
    vc = {c: {k: v | {"run": k[0]} for k, v in SC.agents(f"released__{c}__core", "core").items() if k[0] in dev}
          for c in ("orig", "strip", "plant", "bare", "observer")}
    cb = SC._tier_block("core", vc, fz, params, rc, {}, with_nulls=True)
    del vc
    cb.pop("_null_arrays"), cb.pop("_boots")
    res["core_dev"] = {"detectors": _slim(cb), "null_thesis_L40": cb["null_thesis_L40"],
                       "note": "D4 (Core) leave-domain-out; their probe (rose_*) here is the all-Core fit, in-sample"}
    rs_ = R.load_runs("stego")
    vs = {c: SC.agents(f"redacted__{c}__stego", "stego") for c in ("orig", "strip", "plant", "bare")}
    sb = SC._tier_block("stego", vs, fz, params, rs_, {}, with_nulls=False)
    sb.pop("_boots")
    res["stego_redacted"] = {"detectors": _slim(sb)}
    (S.RESULTS / "posthoc_d4.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for tier in ("transfer", "core_dev", "stego_redacted"):
        print(f"\n{tier}")
        for det, d in res[tier]["detectors"].items():
            dec = d.get("decomposition") or {}
            print(f"{det:>15}  AUROC {d['auroc_peak']:.3f} [{d['auroc_peak_ci'][0]:.2f}, {d['auroc_peak_ci'][1]:.2f}]  "
                  f"STRIP {d.get('T_strip_vs_control', float('nan')):.3f}  PLANT {d.get('C_plant_vs_control', float('nan')):.3f}  "
                  f"BARE {d.get('bare_auroc_peak', float('nan')):.3f}  share {dec.get('share') or float('nan'):.2f} "
                  f"{dec.get('share_ci')} {dec.get('verdict')}  implicit {d.get('auroc_implicit_vs_control', float('nan')):.2f}")
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    fire.Fire({"build": build, "score": score})
