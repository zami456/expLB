#!/usr/bin/env python
"""Q1 mechanism test: is it the personas, or the way we grouped them?

Q1 showed paper16's deception axis is far cleaner than own100's: cos(deception, assistant)
-0.37 against -0.66. Two explanations survive that result. Either the paper's personas are
better *material*, or our grouping of the Assistant-Axis role library was the problem and any
correctly-grouped set would do.

This separates them. Same HF vectors, same model, same layers -- only the grouping changes:

    ours        transgression vs scholarship. spy / saboteur / criminal / predator against
                auditor / librarian / statistician. The plan already flagged this as the
                wrong axis, and noted `gossip` sitting in the deceptive group although
                gossip DISCLOSES private information rather than concealing it.
    disclosure  the paper's criterion: a stance toward what the speaker lets others know.
                Concealment -- operates covertly, makes things disappear, complies in a way
                that misleads -- against disclosure -- surfaces, verifies, documents,
                completes the record.

If the disclosure grouping lands near paper16's -0.37, the contamination was ours to fix and
the personas were never the issue. If it stays near -0.66, the paper's personas really are
better material and no regrouping of an occupation library substitutes for them.

    python 16_regroup_hf.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

# Read off the role descriptions, by the paper's criterion rather than by occupation.
# Concealment: controls what others are allowed to know.
CONCEAL = ["spy", "smuggler", "rogue", "saboteur", "fixer", "criminal", "hacker", "genie"]
# Disclosure: surfaces, verifies, documents, completes the record.
DISCLOSE = ["auditor", "journalist", "reporter", "witness", "observer", "archivist",
            "validator", "examiner", "detective", "judge", "secretary", "jester",
            "prophet", "realist"]

# The grouping the diagnostics used, kept verbatim as the comparison arm. Note `predator`
# and `parasite`: transgressive, but they conceal nothing, which is the whole complaint.
OURS_DEC = ["spy", "saboteur", "criminal", "predator", "parasite", "rogue", "smuggler",
            "hacker", "gossip", "trickster", "fixer", "chameleon"]
OURS_HON = ["auditor", "librarian", "statistician", "scholar", "scientist", "analyst",
            "researcher", "historian", "archivist", "curator", "specialist", "generalist"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_hf_vectors():
    c = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = glob.glob(os.path.join(c, "datasets--lu-christina--assistant-axis-vectors",
                                   "snapshots", "*"))
    if not snaps:
        raise SystemExit("no cached HF vectors")
    vecs = {}
    for p in glob.glob(os.path.join(snaps[0], "llama-3.3-70b", "role_vectors", "*.pt")):
        name = os.path.basename(p)[:-3]
        obj = torch.load(p, map_location="cpu", weights_only=False)
        v = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
        vecs[name] = v.float().numpy()
    return vecs


def contrast(vecs, pos, neg, layer):
    p = [vecs[n][layer] for n in pos if n in vecs]
    q = [vecs[n][layer] for n in neg if n in vecs]
    if not p or not q:
        return None, 0, 0
    return unit(np.mean(p, axis=0) - np.mean(q, axis=0)), len(p), len(q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 40, 60])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    vecs = load_hf_vectors()
    logger.info("cached HF role vectors: %d", len(vecs))
    for label, group in (("concealment", CONCEAL), ("disclosure", DISCLOSE),
                         ("ours-deceptive", OURS_DEC), ("ours-honest", OURS_HON)):
        have = [n for n in group if n in vecs]
        logger.info("  %-16s %2d of %2d available%s", label, len(have), len(group),
                    "" if len(have) == len(group)
                    else "  missing: " + ", ".join(n for n in group if n not in vecs))

    # The Assistant Axis itself: default minus the mean of every other role, the
    # construction the Assistant-Axis paper uses.
    results = []
    logger.info("")
    logger.info("  %6s %14s %14s %12s", "layer", "ours", "disclosure", "paper16 ref")
    ref = {20: -0.364, 22: -0.337, 40: -0.371, 60: -0.374}
    for L in args.layers:
        anchor = "assistant" if "assistant" in vecs else "default"
        if anchor not in vecs:
            raise SystemExit("no assistant/default anchor vector cached")
        others = [n for n in vecs if n != anchor]
        asst = unit(vecs[anchor][L] - np.mean([vecs[n][L] for n in others], axis=0))

        d_ours, a1, b1 = contrast(vecs, OURS_DEC, OURS_HON, L)
        d_disc, a2, b2 = contrast(vecs, CONCEAL, DISCLOSE, L)
        c_ours = float(d_ours @ asst) if d_ours is not None else float("nan")
        c_disc = float(d_disc @ asst) if d_disc is not None else float("nan")
        results.append({"layer": L, "ours": c_ours, "disclosure": c_disc,
                        "n_ours": [a1, b1], "n_disclosure": [a2, b2]})
        logger.info("  %6d %14.3f %14.3f %12.3f", L, c_ours, c_disc, ref.get(L, float("nan")))

    logger.info("")
    m_ours = np.mean([abs(r["ours"]) for r in results])
    m_disc = np.mean([abs(r["disclosure"]) for r in results])
    logger.info("  mean |cos|: ours %.3f, disclosure-grouped %.3f  (paper16 %.3f)",
                m_ours, m_disc, np.mean([abs(v) for v in ref.values()]))
    if m_disc < m_ours - 0.10:
        logger.info("  -> regrouping alone removes most of it: the grouping was the problem")
    elif m_disc > m_ours - 0.03:
        logger.info("  -> regrouping does NOT help: the paper's personas are better material, "
                    "and an occupation library cannot be reorganised into them")
    else:
        logger.info("  -> partial: grouping explains some of it, persona material the rest")
    logger.info("  NOTE: the concealment side has only %d usable roles, so this cannot extend "
                "the Q9 persona-count curve much beyond paper16's 8.", len([n for n in CONCEAL if n in vecs]))

    path = Path(args.out) if args.out else REPO_ROOT / "run-coords" / "regroup_hf.json"
    path.write_text(json.dumps({"conceal": CONCEAL, "disclose": DISCLOSE,
                                "ours_dec": OURS_DEC, "ours_hon": OURS_HON,
                                "results": results}, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
