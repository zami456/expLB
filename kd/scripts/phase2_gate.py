#!/usr/bin/env python
"""Phase 2 gate: do the two factor probes measure what they claim?

Both directions are difference-of-means over the teacher-forced corpora:

    d_knowledge = mean(model knows the item) - mean(does not know)   [wording held constant]
    d_utterance = mean(forced to deny)       - mean(forced to assert) [knowledge held constant]

Each is scored on two HP-KR slices where the right answer is known IN ADVANCE, because of the
structure documented in kd/docs/DESIGN.md section 3 (`deceptive = knows AND denies`, fourth
cell empty):

  text-matched slice (258 rows, every response denies)   -> deceptive <=> KNOWS
      d_knowledge should be HIGH; d_utterance should be ~0.5 (wording is constant here)
  knowledge-matched slice (2,001 rows, model knows all)  -> deceptive <=> DENIES
      d_utterance should be HIGH; d_knowledge should be ~0.5 (knowledge is constant here)

If d_utterance scores high on the TEXT-MATCHED slice, the forced-span construction is picking
up something other than the words that were forced, and the whole attribution is invalid.
That is what this gate is for.

Also runs Option 1: split-half reliability of d_knowledge, against random-direction pairs as
the floor. The negative class is thin (121 items), and this says whether that matters instead
of assuming it does.

    python kd/scripts/phase2_gate.py                    # all 21 layers
    python kd/scripts/phase2_gate.py --layers 40 44
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

KD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KD_ROOT.parent
PIPELINE = REPO_ROOT / "src" / "assistant_axis_ndif"
sys.path.insert(0, str(PIPELINE))

PROBES = KD_ROOT / "run" / "factor_probes"
PASSB = REPO_ROOT / "run-coords" / "passB"
OUT = KD_ROOT / "results" / "phase2_gate.json"


def _mod(name, fname):
    """Load a sibling pipeline module by path.

    The module MUST be registered in sys.modules before exec_module: eval_datasets defines a
    @dataclass, and dataclasses resolves the decorated class's own module out of sys.modules
    to check its annotations. Without the registration that lookup returns None and the
    import dies with "'NoneType' object has no attribute '__dict__'" -- a failure that looks
    nothing like its cause.
    """
    spec = importlib.util.spec_from_file_location(name, PIPELINE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties, so a constant score reads 0.5 rather than 0 or 1
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def diff_of_means(X, y):
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", nargs="*", type=int, default=[])
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_random", type=int, default=50)
    args = ap.parse_args()

    P = _mod("P", "9_persona_probes.py")
    E = _mod("E", "eval_datasets.py")

    import torch, glob
    first = sorted(glob.glob(str(PROBES / "kd_knowledge" / "chunk_*.pt")))[0]
    stored = list(torch.load(first, map_location="cpu", weights_only=False)["layers"])
    layers = args.layers or stored
    missing = [L for L in layers if L not in stored]
    if missing:
        raise SystemExit(f"layers {missing} not extracted; have {stored}")

    tm = E.hpkr_slice("text_matched")
    km = E.hpkr_slice("knowledge_matched")
    print(f"slices: text-matched {len(tm)} rows, knowledge-matched {len(km)} rows")
    print(f"pooling={args.pooling}  layers={layers}\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    hdr = (f"{'layer':>5} | {'dK on':>6} {'dK on':>6} | {'dU on':>6} {'dU on':>6} | "
           f"{'cos':>6} | {'split-half':>10} {'random':>7}")
    print(hdr)
    print(f"{'':>5} | {'t-match':>6} {'k-match':>6} | {'t-match':>6} {'k-match':>6} | "
          f"{'dK,dU':>6} | {'dK r':>10} {'floor':>7}")
    print("-" * len(hdr))

    Xhp, mhp = P.load_dataset(PASSB, "harm-pressure-knowledge-report", layers[0], args.pooling)
    ids_hp = np.array([m["row_id"] for m in mhp])
    y_hp = np.array([m["label"] for m in mhp], dtype=int)
    in_tm = np.array([i in tm for i in ids_hp])
    in_km = np.array([i in km for i in ids_hp])

    for L in layers:
        Xk, mk = P.load_dataset(PROBES, "kd_knowledge", L, args.pooling)
        Xu, mu = P.load_dataset(PROBES, "kd_utterance", L, args.pooling)
        yk = np.array([m["label"] for m in mk], dtype=int)
        yu = np.array([m["label"] for m in mu], dtype=int)
        dK = diff_of_means(Xk, yk)
        dU = diff_of_means(Xu, yu)

        Xh, mh = P.load_dataset(PASSB, "harm-pressure-knowledge-report", L, args.pooling)
        yh = np.array([m["label"] for m in mh], dtype=int)
        sK, sU = Xh @ dK, Xh @ dU

        r = {
            "layer": L,
            "dK_text_matched": auroc(yh[in_tm], sK[in_tm]),
            "dK_knowledge_matched": auroc(yh[in_km], sK[in_km]),
            "dU_text_matched": auroc(yh[in_tm], sU[in_tm]),
            "dU_knowledge_matched": auroc(yh[in_km], sU[in_km]),
            "dK_full": auroc(yh, sK), "dU_full": auroc(yh, sU),
            "cos_dK_dU": float(dK @ dU),
        }
        # Option 1: split-half reliability of d_knowledge, floored by random-direction pairs
        idx1 = np.where(yk == 1)[0]; idx0 = np.where(yk == 0)[0]
        rng.shuffle(idx1); rng.shuffle(idx0)
        halves = []
        for a, b in ((idx1[::2], idx0[::2]), (idx1[1::2], idx0[1::2])):
            halves.append(unit(Xk[a].mean(0) - Xk[b].mean(0)))
        r["split_half_cos"] = float(halves[0] @ halves[1])
        rnd = [unit(rng.standard_normal(Xk.shape[1])) for _ in range(2 * args.n_random)]
        r["random_pair_cos"] = float(np.mean([abs(rnd[2 * i] @ rnd[2 * i + 1])
                                              for i in range(args.n_random)]))
        rows.append(r)
        print(f"{L:>5} | {r['dK_text_matched']:>6.3f} {r['dK_knowledge_matched']:>6.3f} | "
              f"{r['dU_text_matched']:>6.3f} {r['dU_knowledge_matched']:>6.3f} | "
              f"{r['cos_dK_dU']:>6.3f} | {r['split_half_cos']:>10.3f} "
              f"{r['random_pair_cos']:>7.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pooling": args.pooling, "n_text_matched": len(tm),
                               "n_knowledge_matched": len(km), "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")

    best = max(rows, key=lambda r: r["dU_knowledge_matched"])
    print("\nGATE (read at the layer where d_utterance is strongest, L%d):" % best["layer"])
    print(f"  d_utterance on knowledge-matched {best['dU_knowledge_matched']:.3f}  "
          f"(want HIGH -- wording is the only thing varying)")
    print(f"  d_utterance on text-matched      {best['dU_text_matched']:.3f}  "
          f"(want ~0.5 -- every response denies)")
    print(f"  d_knowledge on text-matched      {best['dK_text_matched']:.3f}  "
          f"(want HIGH -- knowledge is the only thing varying)")
    print(f"  d_knowledge on knowledge-matched {best['dK_knowledge_matched']:.3f}  "
          f"(want ~0.5 -- the model knows every one)")
    print(f"  split-half reliability of d_knowledge {best['split_half_cos']:.3f} "
          f"vs random floor {best['random_pair_cos']:.3f}")


if __name__ == "__main__":
    main()
