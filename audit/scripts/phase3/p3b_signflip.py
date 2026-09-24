"""P3-B: the HP-KR training row in the collaborator's 1-d matrices is sign-flipped.

For a 1-d feature the trained probe learns only a sign, so every row of the matrix is either the
zero-shot vector or 1 - zero-shot. Detect that, and report the off-diagonal mean with and without
sign correction.
"""
import csv, sys, statistics as st
sp = sys.argv[1]

def read_matrix(path):
    rows = list(csv.reader(open(path)))
    cols = rows[0][1:]
    m = {r[0]: {c: float(v) for c, v in zip(cols, r[1:])} for r in rows[1:]}
    return cols, m

def offdiag(m, cols, flip=()):
    v = []
    for r in cols:
        for c in cols:
            if r == c:
                continue
            x = m[r][c]
            v.append(1 - x if r in flip else x)
    return st.mean(v)

print("store,layer,matrix,offdiag_as_written,offdiag_sign_corrected,flipped_rows")
for store, layers in (("ext", (20, 40, 44)), ("mod", (20, 40, 44))):
    for L in layers:
        for name in ("axis1", "ext_axis1", "denial_axis1", "pc1"):
            path = f"{sp}/his_{store}_L{L}_matrix_{name}.csv"
            try:
                cols, m = read_matrix(path)
            except FileNotFoundError:
                continue
            # a row is flipped if its off-diagonal entries are closer to 1-x of the modal row
            base = {}
            for c in cols:
                vals = sorted(m[r][c] for r in cols if r != c)
                base[c] = st.median(vals)
            flip = []
            for r in cols:
                d_same = st.mean(abs(m[r][c] - base[c]) for c in cols if c != r)
                d_flip = st.mean(abs((1 - m[r][c]) - base[c]) for c in cols if c != r)
                if d_flip < d_same - 0.05:
                    flip.append(r)
            print(f"{store},{L},{name},{offdiag(m, cols):.4f},{offdiag(m, cols, flip):.4f},"
                  f"{'|'.join(flip) if flip else '-'}")
