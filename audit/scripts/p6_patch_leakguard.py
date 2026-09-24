"""Patch a COPY of 27_crossgen.py to close the HP-KR / kr-matched row-level leak.

Run on the lab PC against the copied tree, never against the original.

The leak (F-108, measured): `harm-pressure-kr-matched` is a VIEW over
`harm-pressure-knowledge-report` -- 258 rows, 100% of its row_ids are HP-KR
row_ids. The original `test_rows` scores an off-diagonal cell on 100% of the
target's rows, so for that one pair the probe is tested on rows it trained on,
in both directions. F-108 also measured that no other dataset pair shares a
single row_id, so the guard is a no-op everywhere else.

  python p6_patch_leakguard.py <path-to-copied-27_crossgen.py>
"""
import hashlib
import sys
from pathlib import Path

EXPECT_MD5 = "681126242fe4414e4a3c131793de81c6"

OLD = '''def test_rows(data, splits_seed, s, t):
    """The rows a probe trained on s is scored on in column t."""
    return splits_seed[t][1] if t == s else np.arange(len(data[t][1]))'''

NEW = '''def test_rows(data, splits_seed, s, t):
    """The rows a probe trained on s is scored on in column t.

    LEAK GUARD (audit, 2026-09-23). Off-diagonal, drop any target row whose row_id
    also appears in the SOURCE's training rows.

    Why: `harm-pressure-kr-matched` is a VIEW over `harm-pressure-knowledge-report`
    -- 258 rows, 100% of its row_ids are HP-KR row_ids (F-108, measured). Scoring
    off-diagonal cells on 100% of the target therefore makes that ONE pair
    train-on-test in both directions. F-108 also measured that no other dataset pair
    shares a single row_id, so this is a no-op for every other cell and the rest of
    the matrix is bit-identical to the unguarded run.
    """
    if t == s:
        return splits_seed[t][1]
    all_t = np.arange(len(data[t][1]))
    train_ids = {data[s][2][i]["row_id"] for i in splits_seed[s][0]}
    meta_t = data[t][2]
    keep = np.array([i for i in all_t if meta_t[i]["row_id"] not in train_ids], dtype=int)
    if len(keep) != len(all_t):
        y_t = data[t][1]
        n_pos = int(y_t[keep].sum()) if len(keep) else 0
        n_neg = int(len(keep) - n_pos)
        logger.warning("  LEAK GUARD %s -> %s: dropped %d of %d target rows "
                       "(row_id shared with source train); %d left (%d pos / %d neg)",
                       s, t, len(all_t) - len(keep), len(all_t), len(keep), n_pos, n_neg)
        if len(keep) == 0 or n_pos == 0 or n_neg == 0:
            logger.error("  LEAK GUARD %s -> %s: cell is UNSCORABLE after the guard "
                         "(single-class or empty) -- it will be reported as null", s, t)
    return keep


# Resamplers for guarded cells, keyed (source, target, n_rows). Built lazily in
# compute_cell_block because res_full[t] is sized to the FULL target.
_GUARD_BOOT = {}'''

OLD_BOOT = """                    if with_boot:
                        b = (res_split[t] if t == s else res_full[t]).boot(sc)"""

NEW_BOOT = """                    if with_boot:
                        # LEAK GUARD companion: res_full[t] indexes the FULL target, so a
                        # guarded cell (fewer rows) would run off the end of it. Build and
                        # cache a resampler matching the rows actually scored. Deterministic
                        # seed so the run reproduces.
                        if t == s:
                            _R = res_split[t]
                        elif len(rows) == len(data[t][1]):
                            _R = res_full[t]
                        else:
                            _ck = (s, t, int(len(rows)))
                            _R = _GUARD_BOOT.get(_ck)
                            if _R is None:
                                _sd = int(hashlib.md5(repr(_ck).encode()).hexdigest()[:8], 16)
                                _R = Resampler(yt, args.boot_rows, np.random.default_rng(_sd))
                                _GUARD_BOOT[_ck] = _R
                        b = _R.boot(sc)"""

OLD_IMPORT = "import argparse\nimport importlib.util"
NEW_IMPORT = "import argparse\nimport hashlib\nimport importlib.util"

OLD_DESIGN = '''DESIGN = ("train on 80% of the row dataset; diagonal = its own 20% test split; "
          "off-diagonal = 100% of the column dataset")'''

NEW_DESIGN = '''DESIGN = ("train on 80% of the row dataset; diagonal = its own 20% test split; "
          "off-diagonal = 100% of the column dataset MINUS any row whose row_id is in "
          "the source's training rows [AUDIT LEAK GUARD 2026-09-23, F-108]")'''


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    p = Path(sys.argv[1])
    src = p.read_text(encoding="utf-8")

    md5 = hashlib.md5(src.encode("utf-8")).hexdigest()
    print(f"target : {p}")
    print(f"md5    : {md5}")
    if md5 != EXPECT_MD5:
        print(f"WARNING: expected {EXPECT_MD5} -- the copy is not the version this patch was "
              f"written against. Check before trusting the result.")

    for name, old in (("test_rows", OLD), ("DESIGN", OLD_DESIGN),
                      ("bootstrap", OLD_BOOT), ("imports", OLD_IMPORT)):
        if old not in src:
            raise SystemExit(f"FAILED: could not find {name} to replace. Nothing written.")

    if "LEAK GUARD" in src:
        print("already patched -- nothing to do (idempotent)")
        return

    out = (src.replace(OLD, NEW).replace(OLD_DESIGN, NEW_DESIGN)
              .replace(OLD_BOOT, NEW_BOOT).replace(OLD_IMPORT, NEW_IMPORT))
    bak = p.with_suffix(p.suffix + ".unpatched")
    if not bak.exists():
        bak.write_text(src, encoding="utf-8")
        print(f"backup : {bak.name}")
    p.write_text(out, encoding="utf-8")
    print(f"patched: +{len(out.splitlines()) - len(src.splitlines())} lines")
    print("         test_rows now drops target rows whose row_id is in the source's train set")
    print("         DESIGN string stamped, so every output JSON records the change")


if __name__ == "__main__":
    main()
