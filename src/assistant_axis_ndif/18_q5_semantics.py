#!/usr/bin/env python
"""Q5: does transfer fail across LABEL SEMANTICS, or merely across datasets?

The transfer matrix has an off-diagonal mean. Splitting that mean by whether source and
target share a label definition is the whole of Q5 -- but the naive split is confounded,
and this script exists to say how loudly.

**The confound.** Seven of the ten trainable datasets carry `instructed_intent`, and they
come from two shared generation pipelines. So "within-semantics" is largely
"within-provenance", and `17_validate.py` already measured how big that effect is: dataset
identity is 99.3% decodable from these activations against a label lift of +0.032, and
PC1-PC3 separate individual corpora at AUROC 1.000. A high within-semantics mean is the
expected result under EITHER hypothesis and on its own decides nothing.

**What does decide it.** `mask` and `sycophancy_mt` share `belief_inconsistency` and share
nothing else. MASK is an external benchmark of pressured statements checked against
elicited beliefs; sycophancy_mt is our own multi-turn corpus where the model retracts its
own answer under social pressure with no new evidence. Different authors, prompts, topics
and construction -- one construct. That pair is the cleanest test in the design, and it
did not exist until the sycophancy corpus was extracted.

So this reports three things, in increasing order of what they are worth:

1. the pooled within/across split, marked as confounded;
2. the 3x3 block matrix by (source semantics, target semantics);
3. the cross-provenance cells, which are the actual evidence.

A label-permutation test accompanies (1): reassign the semantics tags across datasets at
random and recompute within-minus-across. That holds the matrix and the group SIZES fixed
and asks only whether the observed grouping is special.

    python 18_q5_semantics.py --p1 run-coords/p1_L44_last_withsyc.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

# Which construct each dataset's label encodes. Mirrors eval_datasets.py; kept explicit
# here so the analysis can be read without running the loader.
SEMANTICS = {
    "instructed-deception": "instructed_intent",
    "harm-pressure-knowledge-report": "instructed_intent",
    "harm-pressure-choice": "instructed_intent",
    "insider-trading": "instructed_intent",
    "convincing-game": "instructed_intent",
    "ai_liar": "instructed_intent",
    "sandbagging_v2": "instructed_intent",
    "ai_audit": "instructed_intent",
    "roleplaying": "judge_grade",
    "mask": "belief_inconsistency",
    "sycophancy_mt": "belief_inconsistency",
}

# Where the rows came from, which is NOT the same question as what the label means. This
# is the variable that has to be held apart from semantics for Q5 to mean anything.
PROVENANCE = {
    "instructed-deception": "liars-bench",
    "harm-pressure-knowledge-report": "liars-bench",
    "harm-pressure-choice": "liars-bench",
    "insider-trading": "liars-bench",
    "convincing-game": "liars-bench",
    "ai_liar": "apollo",
    "sandbagging_v2": "apollo",
    "ai_audit": "apollo",
    "roleplaying": "apollo",
    "mask": "mask-benchmark",
    "sycophancy_mt": "ours",
}

SHORT = {"instructed_intent": "intent", "judge_grade": "judge",
         "belief_inconsistency": "belief"}
ORDER = ["instructed_intent", "judge_grade", "belief_inconsistency"]


def parse_matrix(cells: dict) -> dict:
    out = {}
    for key, v in cells.items():
        s, t = key.split("->")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[(s, t)] = float(v)
    return out


def permutation_p(mat: dict, sem: dict, draws: int, rng):
    """Hold the matrix and the group sizes fixed; reshuffle which dataset carries which tag.

    Shuffling the ASSIGNMENT rather than the cells is what makes this a test of the
    grouping. A block that looks cohesive only because it is large, or because its members
    are mutually similar for reasons unrelated to the label, gets matched by a random
    relabelling of the same sizes.
    """
    keys = sorted({k for pair in mat for k in pair})
    tags = [sem[k] for k in keys]

    def stat(assign):
        same = [v for (s, t), v in mat.items() if s != t and assign[s] == assign[t]]
        cross = [v for (s, t), v in mat.items() if s != t and assign[s] != assign[t]]
        if not same or not cross:
            return float("nan")
        return float(np.mean(same) - np.mean(cross))

    obs = stat(sem)
    hits = tot = 0
    for _ in range(draws):
        perm = list(tags)
        rng.shuffle(perm)
        d = stat(dict(zip(keys, perm)))
        if not np.isnan(d):
            tot += 1
            hits += int(d >= obs)
    return obs, ((hits + 1) / (tot + 1) if tot else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default=str(REPO_ROOT / "run-coords" / "p1_L44_last_withsyc.json"))
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    doc = json.loads(Path(args.p1).read_text(encoding="utf-8"))
    entries = doc["p1"] if isinstance(doc, dict) and "p1" in doc else doc
    rng = np.random.default_rng(args.seed)
    results = []

    for e in entries:
        mat = parse_matrix(e["matrix"])
        keys = sorted({k for pair in mat for k in pair})
        unknown = [k for k in keys if k not in SEMANTICS]
        if unknown:
            raise SystemExit(f"no semantics tag for {unknown}")
        logger.info("")
        logger.info("=" * 78)
        logger.info("%s / %s   (%d datasets, %d off-diagonal cells)",
                    e["features"], e["probe"], len(keys),
                    sum(1 for s, t in mat if s != t))

        # ---- 1. pooled split, confounded -----------------------------------------
        obs, p = permutation_p(mat, SEMANTICS, args.draws, rng)
        same = [v for (s, t), v in mat.items() if s != t and SEMANTICS[s] == SEMANTICS[t]]
        cross = [v for (s, t), v in mat.items() if s != t and SEMANTICS[s] != SEMANTICS[t]]
        logger.info("")
        logger.info("1. pooled within/across  -- CONFOUNDED with provenance, see 3")
        logger.info("   within-semantics  %.3f  (n=%d)", np.mean(same), len(same))
        logger.info("   across-semantics  %.3f  (n=%d)", np.mean(cross), len(cross))
        logger.info("   difference       %+.3f    permutation p = %.4f", obs, p)

        # ---- 2. block matrix -------------------------------------------------------
        block = defaultdict(list)
        for (s, t), v in mat.items():
            if s != t:
                block[(SEMANTICS[s], SEMANTICS[t])].append(v)
        logger.info("")
        logger.info("2. mean AUROC by block, rows = source construct, cols = target")
        logger.info("   %-10s%s", "src/tgt", "".join(f"{SHORT[c]:>14}" for c in ORDER))
        for r in ORDER:
            cells = []
            for c in ORDER:
                v = block.get((r, c))
                cells.append(f"{np.mean(v):>10.3f}({len(v):>2})" if v else f"{'--':>14}")
            logger.info("   %-10s%s", SHORT[r], "".join(cells))

        # ---- 3. the cells that are not confounded ---------------------------------
        clean_same = [((s, t), v) for (s, t), v in mat.items()
                      if s != t and SEMANTICS[s] == SEMANTICS[t]
                      and PROVENANCE[s] != PROVENANCE[t]]
        clean_cross = [((s, t), v) for (s, t), v in mat.items()
                       if s != t and SEMANTICS[s] != SEMANTICS[t]
                       and PROVENANCE[s] != PROVENANCE[t]]
        logger.info("")
        logger.info("3. cross-PROVENANCE cells -- the label definition is the only thing shared")
        for (s, t), v in sorted(clean_same, key=lambda x: -x[1]):
            logger.info("   same construct   %-32s -> %-22s %.3f", s, t, v)
        m_same = float(np.mean([v for _, v in clean_same])) if clean_same else float("nan")
        m_cross = float(np.mean([v for _, v in clean_cross])) if clean_cross else float("nan")
        logger.info("   same-construct mean  %.3f  (n=%d)", m_same, len(clean_same))
        logger.info("   diff-construct mean  %.3f  (n=%d)", m_cross, len(clean_cross))
        logger.info("   gap                 %+.3f", m_same - m_cross)

        results.append({
            "features": e["features"], "probe": e["probe"],
            "within_semantics": float(np.mean(same)), "n_within": len(same),
            "across_semantics": float(np.mean(cross)), "n_across": len(cross),
            "difference": obs, "permutation_p": p,
            "blocks": {f"{r}->{c}": {"mean": float(np.mean(v)), "n": len(v)}
                       for (r, c), v in block.items()},
            "cross_provenance_same_construct": {f"{s}->{t}": v for (s, t), v in clean_same},
            "cross_provenance_diff_construct": {f"{s}->{t}": v for (s, t), v in clean_cross},
            "cross_provenance_same_mean": m_same,
            "cross_provenance_diff_mean": m_cross,
            "cross_provenance_gap": m_same - m_cross,
        })

    logger.info("")
    logger.info("=" * 78)
    logger.info("judge_grade still has exactly one dataset (roleplaying), so its within-")
    logger.info("construct block is empty by construction. No claim about it is available.")
    path = Path(args.out) if args.out else Path(args.p1).with_name("q5_semantics.json")
    path.write_text(json.dumps({"source": str(args.p1), "semantics": SEMANTICS,
                                "provenance": PROVENANCE, "results": results}, indent=2),
                    encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
