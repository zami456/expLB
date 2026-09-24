"""Build the NEVER-RUN `knowledge_matched` control slice of HP-KR, without touching originals.

The 2x2 this completes
----------------------
harm-pressure-knowledge-report confounds two things: the model DENIES knowledge, and it
words that denial distinctively. `eval_datasets.hpkr_slice` can hold either one constant:

  text_matched       every response denies  -> deceptive <=> the model KNOWS      258 rows  (RAN)
  knowledge_matched  every row is known     -> deceptive <=> the model DENIES   2,001 rows  (NEVER RUN)

The first asks "with wording held constant, can the detector find knowledge?".
The second asks "with knowledge held constant, is the detector just reading denial?" --
which is the direct test of the register/assertion diagnosis, on 7.8x more rows.

Isolation
---------
`make_view.py` (the lab's own script) writes its view INTO run-coords/passB, i.e. into an
original store. This does not. It builds a parallel store whose other datasets are
DIRECTORY JUNCTIONS to the originals and whose one new dataset is a real folder of
HARDLINKS to HP-KR's existing chunk files. Zero extra bytes, nothing original modified.

  python audit/scripts/p6_build_km_view.py --print-only   # sizes, writes nothing
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "assistant_axis_ndif"))

NEW_KEY = "harm-pressure-kr-knowledge-matched"
SRC_KEY = "harm-pressure-knowledge-report"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    import eval_datasets as E

    tm = E.hpkr_slice("text_matched")
    km = E.hpkr_slice("knowledge_matched")
    rows = E.load_dataset(SRC_KEY)
    lab = {}
    for r in rows:
        rid = r.row_id if hasattr(r, "row_id") else r["row_id"]
        lab[rid] = r.label if hasattr(r, "label") else r["label"]

    print(f"full {SRC_KEY}: {len(rows)} rows")
    for name, s in (("text_matched", tm), ("knowledge_matched", km)):
        c = Counter(lab[i] for i in s if i in lab)
        print(f"  {name:18s} {len(s):5d} rows   deceptive={c.get(1,0):4d}  honest={c.get(0,0):4d}"
              f"   present in store={len(s & set(lab)):5d}")
    print(f"  overlap between the two slices: {len(tm & km)}")

    if args.print_only:
        print("\n--print-only: nothing written.")
        return

    print(f"\nThis script only computes the row set. The view itself is built on the lab PC,\n"
          f"where the activations live, by p6_build_km_view.ps1 -- which writes ONLY under\n"
          f"C:/Users/user4/expLB-resplit/ and never into run-coords/passB.")


if __name__ == "__main__":
    main()
