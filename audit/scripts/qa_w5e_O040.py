"""LEAD QA for O-040 — independent re-derivation.

The claim (docs/NOVELTY_CHECK_PROMPT.md:103-108, asserted as the project's own
contribution in docs/NOVELTY_ASSESSMENT.md:30 and used as the premise of the
pre-registration's directional prediction, docs/LAYER_PREREGISTRATION.md:93):

    "A regex on denial phrasing alone reaches 0.91 AUROC on a dataset where
     probes report ~0.87"

The batch reported the 0.91 as correct and the ~0.87 as unsourceable.  This
script re-derives the probe side from scratch with its own loader, making no use
of the batch's note, and answers three questions:

  Q1. Over EVERY stored matrix, what AUROCs do probes actually reach on
      harm-pressure-knowledge-report, split by role (diagonal / as transfer
      target / as transfer source) and by feature?
  Q2. Does ANY probe cell on HP-KR land in [0.86, 0.88]?
  Q3. What else in this project equals ~0.87 on HP-KR?

Read-only.  Prints; writes nothing.
"""

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = "harm-pressure-knowledge-report"
LO, HI = 0.86, 0.88


def matrices():
    for p in sorted(ROOT.glob("run-crossgen/**/matrices/*.json")):
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! unreadable {p}: {exc}", file=sys.stderr)


def main():
    diag, as_target, as_source = [], [], []
    in_band = []
    n_files = 0
    feat_target = collections.defaultdict(list)

    for path, d in matrices():
        ds = d.get("datasets")
        if not ds or TARGET not in ds:
            continue
        n_files += 1
        j = ds.index(TARGET)
        rel = path.relative_to(ROOT).as_posix()
        meta = f"{rel} L{d.get('layer')} {d.get('pooling')} s{d.get('seed')}"

        for res in d.get("results", []):
            feat, probe = res.get("feature"), res.get("probe")
            a = res.get("auroc")
            if not isinstance(a, list):
                continue
            # a is a square matrix: a[row_trained_on][col_tested_on]
            for i, row in enumerate(a):
                if not isinstance(row, list) or len(row) != len(ds):
                    continue
                v = row[j]
                if v is None:
                    continue
                tag = f"{feat}|{probe}"
                if i == j:
                    diag.append((v, tag, meta))
                else:
                    as_target.append((v, tag, meta, ds[i]))
                    feat_target[tag].append(v)
                if LO <= v <= HI:
                    role = "diagonal" if i == j else f"target (trained on {ds[i]})"
                    in_band.append((v, tag, role, meta))
                # HP-KR as the SOURCE of transfer
                if i == j:
                    continue
            for i, row in enumerate(a):
                if not isinstance(row, list) or len(row) != len(ds) or i != j:
                    continue
                for k, v in enumerate(row):
                    if k != j and v is not None:
                        as_source.append((v, f"{feat}|{probe}", meta, ds[k]))
                        if LO <= v <= HI:
                            in_band.append((v, f"{feat}|{probe}",
                                            f"source -> {ds[k]}", meta))

    def rng(vals, name):
        if not vals:
            print(f"  {name}: none")
            return
        xs = sorted(v[0] for v in vals)
        print(f"  {name}: n={len(xs)}  min {xs[0]:.4f}  "
              f"median {xs[len(xs)//2]:.4f}  max {xs[-1]:.4f}")

    print(f"Q1. matrices containing {TARGET}: {n_files}")
    rng(diag, "HP-KR diagonal (trained and tested on HP-KR)")
    rng(as_target, "HP-KR as transfer TARGET")
    rng(as_source, "HP-KR as transfer SOURCE")

    print("\n  per-feature, HP-KR as transfer target:")
    for tag in sorted(feat_target, key=lambda t: -len(feat_target[t]))[:14]:
        xs = sorted(feat_target[tag])
        print(f"    {tag:26} n={len(xs):5}  min {xs[0]:.4f}  "
              f"median {xs[len(xs)//2]:.4f}  max {xs[-1]:.4f}")

    print(f"\nQ2. probe cells on HP-KR inside [{LO}, {HI}]: {len(in_band)}")
    seen = collections.Counter((t, r.split(' ->')[0].split(' (')[0])
                               for _, t, r, _ in in_band)
    for (tag, role), n in seen.most_common(20):
        vals = [v for v, t, r, _ in in_band
                if t == tag and r.startswith(role)]
        print(f"    {tag:26} {role:10} n={n:4}  "
              f"[{min(vals):.4f}, {max(vals):.4f}]")
    if not in_band:
        print("    NONE")

    print("\nQ3. the diagonal is the only place a probe is strong on HP-KR;")
    print("    a claim of the form 'probes report X on HP-KR' must say which role.")


if __name__ == "__main__":
    main()
