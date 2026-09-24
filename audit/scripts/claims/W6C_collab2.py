"""W6C-briefing, addendum: O-036 done LIKE-FOR-LIKE.

`audit_L40_s42.json`'s "median_over_sources" is the mean over TARGETS of the
per-target median over sources (verified in W6C_premise.py), not the median of
all 110 pairs. So the same estimator has to be applied to the collaborator's
own 6x6 transfer matrix before the briefing's sentence
  "whether 0.732 ... when the median over the same pairs is 0.722"
can be checked. This script does that.

Read-only; the collaborator's matrix is read with `git show`.
"""
import csv
import io
import subprocess
import sys

import numpy as np

REPO = "C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
REF = "origin/feat/persona-pc-ood"
BASE = "src/probes/deception-detection/results"
p = sys.stdout.write


def gitshow(path):
    o = subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (REF, path)],
                       capture_output=True)
    o.check_returncode()
    return o.stdout.decode("utf-8", "replace")


def matrix(layer, name="matrix_baseline.csv"):
    rd = list(csv.reader(io.StringIO(
        gitshow("%s/persona_ood_ext_L%d/%s" % (BASE, layer, name)))))
    cols = rd[0][1:]
    cells = {}
    for line in rd[1:]:
        for j, c in enumerate(line[1:]):
            try:
                cells[(line[0], cols[j])] = float(c)
            except ValueError:
                pass
    return cells, sorted({k[0] for k in cells}), cols


for layer in (20, 40, 44):
    cells, srcs, tgts = matrix(layer)
    p("\n---- his layer %d, trained single-source transfer matrix (%d sources x %d targets)\n"
      % (layer, len(srcs), len(tgts)))
    per_t_mean, per_t_med, per_t_or, per_t_best = [], [], [], []
    for t in tgts:
        col = [cells[(s, t)] for s in srcs if s != t and (s, t) in cells]
        if not col:
            continue
        a = np.array(col, float)
        per_t_mean.append(a.mean())
        per_t_med.append(float(np.median(a)))
        per_t_or.append(float(np.maximum(a, 1 - a).mean()))
        per_t_best.append(float(a.max()))
        p("    target %-34s n=%d  mean %.4f  median %.4f  oriented %.4f  best %.4f\n"
          % (t, len(col), a.mean(), np.median(a), np.maximum(a, 1 - a).mean(), a.max()))
    p("  HIS four baselines, same estimators as our audit_L%d_s42.json:\n" % layer)
    p("    mean over sources        %.4f    (the briefing's '0.732' at L40)\n"
      % float(np.mean(per_t_mean)))
    p("    median over sources      %.4f    (the briefing quotes 0.722 -- OUR number)\n"
      % float(np.mean(per_t_med)))
    p("    sign-oriented mean       %.4f    (the briefing quotes 0.735 -- OUR number)\n"
      % float(np.mean(per_t_or)))
    p("    best source per target   %.4f    (we quote 0.897)\n" % float(np.mean(per_t_best)))
    flat = np.array([v for (s, t), v in cells.items() if s != t], float)
    p("    (plain median of all %d off-diagonal pairs: %.4f; min %.4f;"
      " below 0.5: %d; below 0.4: %d)\n"
      % (len(flat), float(np.median(flat)), flat.min(),
         int((flat < 0.5).sum()), int((flat < 0.4).sum())))
