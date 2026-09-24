#!/usr/bin/env python3
"""AUDIT, READ-ONLY. Level-2 recompute of F-05 and F-06 from the stored activations.

Runs on the lab PC (that is where run-coords/passB and kd/run/factor_probes live).
Writes NOTHING: every path is opened for reading.

F-05: difference-of-means factor probes d_knowledge / d_utterance scored on the two
      HP-KR slices at L20/L40/L44, mean pooling.
F-06: cos(d_knowledge, d_utterance) and the orthogonalised d_utterance_perp scored on
      the text-matched slice (the "0.784 -> 0.509" number, which exists only as console
      output of the lab-only loose script C:/Users/user4/expLB/ortho.py).

The AUROC, the unit() and the diff-of-means are re-implemented here rather than imported,
so the check is independent of kd/scripts/phase2_gate.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path("C:/Users/user4/expLB")
PIPELINE = REPO / "src" / "assistant_axis_ndif"
PROBES = REPO / "kd" / "run" / "factor_probes"
PASSB = REPO / "run-coords" / "passB"
sys.path.insert(0, str(PIPELINE))


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def auroc(y, s):
    """Rank AUROC with ties at 0.5 (independent implementation)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def main():
    P = _mod("P", "9_persona_probes.py")
    E = _mod("E", "eval_datasets.py")

    tm = E.hpkr_slice("text_matched")
    km = E.hpkr_slice("knowledge_matched")
    print(f"slices from eval_datasets.hpkr_slice: text_matched={len(tm)}  knowledge_matched={len(km)}")

    print(f"{'layer':>5} {'n_tm':>5} {'n_km':>5} | {'dK_tm':>7} {'dK_km':>7} {'dU_tm':>7} {'dU_km':>7} "
          f"| {'dK_full':>7} {'dU_full':>7} | {'cos':>7} | {'dUperp_tm':>9} {'dUperp_km':>9} {'dUperp_full':>11}")
    for L in (20, 40, 44):
        Xk, mk = P.load_dataset(PROBES, "kd_knowledge", L, "mean")
        Xu, mu = P.load_dataset(PROBES, "kd_utterance", L, "mean")
        yk = np.array([m["label"] for m in mk], int)
        yu = np.array([m["label"] for m in mu], int)
        dK = unit(Xk[yk == 1].mean(0) - Xk[yk == 0].mean(0))
        dU = unit(Xu[yu == 1].mean(0) - Xu[yu == 0].mean(0))
        cos = float(dK @ dU)
        dUp = unit(dU - cos * dK)

        Xh, mh = P.load_dataset(PASSB, "harm-pressure-knowledge-report", L, "mean")
        yh = np.array([m["label"] for m in mh], int)
        ids = np.array([m["row_id"] for m in mh])
        in_tm = np.array([i in tm for i in ids])
        in_km = np.array([i in km for i in ids])
        sK, sU, sP = Xh @ dK, Xh @ dU, Xh @ dUp
        print(f"{L:>5} {int(in_tm.sum()):>5} {int(in_km.sum()):>5} | "
              f"{auroc(yh[in_tm], sK[in_tm]):>7.4f} {auroc(yh[in_km], sK[in_km]):>7.4f} "
              f"{auroc(yh[in_tm], sU[in_tm]):>7.4f} {auroc(yh[in_km], sU[in_km]):>7.4f} | "
              f"{auroc(yh, sK):>7.4f} {auroc(yh, sU):>7.4f} | {cos:>7.4f} | "
              f"{auroc(yh[in_tm], sP[in_tm]):>9.4f} {auroc(yh[in_km], sP[in_km]):>9.4f} "
              f"{auroc(yh, sP):>11.4f}")
        if L == 20:
            print(f"      (kd_knowledge n={len(yk)} pos={int(yk.sum())} neg={int((yk==0).sum())}; "
                  f"kd_utterance n={len(yu)} pos={int(yu.sum())}; HP-KR store n={len(yh)} "
                  f"dec={int(yh.sum())})")


if __name__ == "__main__":
    main()
