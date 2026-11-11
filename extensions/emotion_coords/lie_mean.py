"""The averaged RepE lie directions, zero-shot (design: `lie_mean_spec.py`, frozen before any is scored). Liars' Bench's five
sets on `lie_lb`'s rows and draws, the ten datasets' test splits on `ms10`'s, blocks 35-45; the raw probe's transfer
re-scored on the same draws. Runs in E:/aletheia/venv from the deception-detection folder.

    python -m extensions.emotion_coords.lie_mean freeze
    python -m extensions.emotion_coords.lie_mean score
    python -m extensions.emotion_coords.lie_mean report
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import aletheia_topk as TK
from extensions.emotion_coords import config as C
from extensions.emotion_coords import lie_lb as LL
from extensions.emotion_coords import lie_lb_spec as B
from extensions.emotion_coords import lie_mean_spec as A
from extensions.emotion_coords import ms10 as M
from extensions.emotion_coords import ms10_lie as ML
from extensions.emotion_coords import ms10_spec as S
from extensions.emotion_coords import ms10_thesis as MT
from extensions.emotion_coords import probes as P


def out_dir() -> Path:
    return C.results_dir("lie_mean")


def freeze() -> str:
    path = Path(A.__file__)
    line = f"{path.name} sha256 {M._sha256(path)}"
    if A.FREEZE.exists():
        held = A.FREEZE.read_text(encoding="utf-8")
        if line not in held:
            raise RuntimeError(f"{A.FREEZE} holds another design:\n{held}")
        return held
    text = f"{line}\nfrozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    A.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {A.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    digest = M._sha256(Path(A.__file__))
    if not A.FREEZE.exists() or digest not in A.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"lie_mean_spec.py ({digest[:12]}) is not the frozen design ({A.FREEZE})")
    LL.check_freeze()  # lie_lb's, and through it ms10_lie's and ms10's
    return digest


def _directions(block: int) -> dict[str, np.ndarray]:
    from deception_detection import persona_pcs as pcs

    w = ML._directions(block)
    x_tr, y_tr, _, _ = ML._repe(block)
    dom = x_tr[y_tr == 1].mean(0) - x_tr[y_tr == 0].mean(0)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    return {"lie_mean100": unit(w.sum(0)), "lie_mean10": unit(w[:10].sum(0)), "lie_w1": w[0], "repe_dom": unit(dom),
            "D9": pcs.load_bank_at_layer(block, pcs.EXTENDED_VECTORS_DIR).axis_unit("extended")}


def score() -> None:
    """Every direction on every set at every block (and the raw probe's transfer), on the earlier draws; A1, A2."""
    digest = check_freeze()
    ms_tests = MT._tests()
    records, draws, cosines = [], {}, []
    for b in A.BLOCKS:
        dirs = _directions(b)
        cosines.append({"block": b} | {f"cos({n}, repe_dom)": float(dirs[n] @ dirs["repe_dom"])
                                        for n in ("lie_mean100", "lie_mean10", "lie_w1", "D9")})
        lb = LL._block(b)
        lb_tests = LL._tests(lb)
        ms = M._block(b)
        ms_x = {n: ms[f"x_{n}"][ms_tests[n][5]].astype(np.float64) for n in S.SETS}
        suites = {"lb": ({n: (lb[n][0].astype(np.float64), lb_tests[n]) for n in B.SETS},
                         {"all": list(B.SETS), "excl. HP-KR": [n for n in B.SETS if n != B.HPKR]}),
                  "ms10": ({n: (ms_x[n], ms_tests[n][:5]) for n in S.SETS},
                           {"deception": list(S.DECEPTION), "new deception": list(MT.NEW),
                            "sycophancy": list(S.SYCOPHANCY)})}
        for suite, (sets, regions) in suites.items():
            for name, v in dirs.items():
                cells = {}
                for t, (x, (y, pos, neg, up, un)) in sets.items():
                    s = x @ v
                    cells[t] = TK._boot_aurocs(up, un, s, pos, neg)
                    records.append({"suite": suite, "block": b, "detector": name, "train": "none", "test": t,
                                    "auroc": P.auroc(y, s)})
                for r, keys in regions.items():
                    draws[suite, b, name, r] = np.mean([cells[t] for t in keys], axis=0)
        # the raw probe's transfer: Liars' Bench from lie_lb's parts, the ten datasets from ms10's
        with np.load(LL._part(b, "raw", 0)) as z:
            probes = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
        cells = {}
        for src, (v, o) in probes.items():
            for t in B.SETS:
                if t != src:
                    y, pos, neg, up, un = lb_tests[t]
                    s = lb[t][0].astype(np.float64) @ v + o
                    cells[src, t] = TK._boot_aurocs(up, un, s, pos, neg)
                    records.append({"suite": "lb", "block": b, "detector": "raw probe", "train": src, "test": t,
                                    "auroc": P.auroc(y, s)})
        for r, keys in LL._regions(False).items():
            draws["lb", b, "raw probe", r] = np.mean([cells[k] for k in keys], axis=0)
        with np.load(M._part(b, "raw", 0)) as z:
            probes = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
        cells = {}
        for src, (v, o) in probes.items():
            for t in S.SETS:
                y, pos, neg, up, un, _ = ms_tests[t]
                s = ms_x[t] @ v + o
                cells[src, t] = TK._boot_aurocs(up, un, s, pos, neg)
                records.append({"suite": "ms10", "block": b, "detector": "raw probe", "train": src, "test": t,
                                "auroc": P.auroc(y, s)})
        for r, keys in MT._regions(False).items():
            if r in ("deception", "new deception", "sycophancy"):
                draws["ms10", b, "raw probe", r] = np.mean([cells[k] for k in keys], axis=0)
        logger.info(f"B{b}: scored")
    auroc = pd.DataFrame(records)
    auroc.to_csv(out_dir() / "auroc.csv", index=False)
    pd.DataFrame(cosines).to_csv(out_dir() / "cosines.csv", index=False)
    rows = [{"suite": s, "block": b, "detector": n, "region": r, "lo": float(np.quantile(d, 0.025)),
             "hi": float(np.quantile(d, 0.975)), "boot_mean": float(d.mean())} for (s, b, n, r), d in draws.items()]
    regions = pd.DataFrame(rows)
    point = []
    for (s, b, n, r) in draws:
        if n == "raw probe":
            keys = (LL._regions(False) if s == "lb" else MT._regions(False))[r]
            sub = auroc[(auroc["suite"] == s) & (auroc["block"] == b) & (auroc["detector"] == n)]
            sub = sub.set_index(["train", "test"])["auroc"]
            point.append(float(np.mean([sub[k] for k in keys])))
        else:
            keys = {"all": list(B.SETS), "excl. HP-KR": [x for x in B.SETS if x != B.HPKR], "deception": list(S.DECEPTION),
                    "new deception": list(MT.NEW), "sycophancy": list(S.SYCOPHANCY)}[r]
            sub = auroc[(auroc["suite"] == s) & (auroc["block"] == b) & (auroc["detector"] == n)].set_index("test")["auroc"]
            point.append(float(np.mean([sub[t] for t in keys])))
    regions["mean"] = point
    regions.to_csv(out_dir() / "regions.csv", index=False)
    deltas = []
    means = regions.set_index(["suite", "block", "detector", "region"])["mean"]
    for (s, b, n, r) in draws:
        for ref in ("lie_w1", "raw probe", "repe_dom", "D9"):
            if n != ref and (s, b, ref, r) in draws:
                diff = draws[s, b, n, r] - draws[s, b, ref, r]
                deltas.append({"suite": s, "block": b, "region": r, "detector": n, "minus": ref,
                               "delta": means[s, b, n, r] - means[s, b, ref, r],
                               "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    verdicts = {"freeze_sha256": digest}
    for key, (suite, region) in (("A1", ("lb", "all")), ("A2", ("ms10", "deception"))):
        d = deltas[(deltas["suite"] == suite) & (deltas["region"] == region) & (deltas["detector"] == "lie_mean100")
                   & (deltas["minus"] == "lie_w1")]
        n = int((d["lo"] > 0).sum())
        verdicts[key] = {"blocks_with_ci_above_0": n, "blocks_with_ci_below_0": int((d["hi"] < 0).sum()),
                         "mean_delta": float(d["delta"].mean()), "holds": n >= A.MIN_BLOCKS}
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info(json.dumps(verdicts, indent=1))


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    regions = pd.read_csv(out_dir() / "regions.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    auroc = pd.read_csv(out_dir() / "auroc.csv")
    cos = pd.read_csv(out_dir() / "cosines.csv")
    lines = ["# The averaged RepE lie directions, zero-shot (Llama-3.3-70B, blocks 35-45; post hoc question)", "",
             f"Design: `emotion_coords/lie_mean_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before any "
             "averaged direction was scored. Same rows and draws as `lie_lb` and `ms10`.", "", "## Predictions", ""]
    for key in ("A1", "A2"):
        v = verdicts[key]
        lines.append(f"- **{key}** ({A.PREDICTIONS[key]}): mean {v['mean_delta']:+.3f}; CI above 0 at "
                     f"{v['blocks_with_ci_above_0']}, below at {v['blocks_with_ci_below_0']}: "
                     f"**{'holds' if v['holds'] else 'fails'}**.")
    order = ["lie_mean100", "lie_mean10", "lie_w1", "repe_dom", "D9", "raw probe"]
    for suite, regs in (("lb", ("all", "excl. HP-KR")), ("ms10", ("deception", "new deception", "sycophancy"))):
        lines += ["", f"## {'Liars Bench' if suite == 'lb' else 'The ten datasets'}: mean over the 11 blocks", "",
                  "| detector | " + " | ".join(regs) + " |", "|---|" + "---|" * len(regs)]
        r = regions[regions["suite"] == suite]
        for n in order:
            lines.append(f"| {n} | " + " | ".join(f3(r[(r['detector'] == n) & (r['region'] == g)]['mean'].mean())
                                                  for g in regs) + " |")
    lb = auroc[(auroc["suite"] == "lb") & (auroc["train"] == "none")].groupby(["detector", "test"])["auroc"].mean()
    lines += ["", "## Liars' Bench, each set (zero-shot; mean over the blocks)", "",
              "| detector | " + " | ".join(B.SHORT[s] for s in B.SETS) + " |", "|---|" + "---|" * len(B.SETS)]
    for n in order[:-1]:
        lines.append(f"| {n} | " + " | ".join(f3(lb[n, s]) for s in B.SETS) + " |")
    lines += ["", "## Cosine with RepE's difference of means (mean over the blocks)", "",
              " | ".join(f"{c}: {cos[c].mean():.3f}" for c in cos.columns if c != "block")]
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"freeze": freeze, "score": score, "report": report})
