"""RF-16 completeness: fingerprint EVERY scale_70b matrix (both poolings, all layers) against banks whose identity
is known, instead of inferring the sweep from two files.

Method: the same bank on the same activations gives identical zero-shot AUROCs. For each scale_70b matrix we compare
its zero-shot (direction, dataset) values with the matrix of the same pooling/layer in each reference store, and
report the best match. Read-only; runs on the lab PC.
"""
import json, os

B = "C:/Users/user4/expLB/run-crossgen/"
REF = [("paper16 (kd11 root, fp32)", "kd11/matrices"),
       ("paper16 (root run, fp32)", "matrices"),
       ("kd bank (36 personas)", "kd11/banks/kd/matrices"),
       ("kd_s23 bank", "kd11/banks/kd_s23/matrices"),
       ("paper16_famcentred", "banks/paper16_famcentred/matrices"),
       ("paper16_shared", "banks/paper16_shared/matrices"),
       ("own100", "banks/own100/matrices"),
       ("paper16 fp32 (audit re-run)", "scale_70b_paper16fp32/matrices")]


def zs(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for z in d.get("zeroshot", []):
        for ds, a in zip(d["datasets"], z["auroc"]):
            if a is not None:
                out[(z["direction"], ds)] = round(a, 6)
    return d.get("bank"), d.get("bank_name"), out


print(f"{'matrix':22s} {'bank field':26s} {'best match':30s} {'common':>6s} {'identical':>9s} {'max|d|':>8s}")
rows = []
for fn in sorted(os.listdir(B + "scale_70b/matrices")):
    if not fn.endswith("_s42.json"):
        continue
    tb, tn, T = zs(B + "scale_70b/matrices/" + fn)
    best = None
    for label, sub in REF:
        p = B + sub + "/" + fn
        if not os.path.exists(p):
            continue
        _, _, Z = zs(p)
        common = [k for k in T if k in Z]
        if not common:
            continue
        ident = sum(abs(T[k] - Z[k]) < 1e-9 for k in common)
        maxd = max(abs(T[k] - Z[k]) for k in common)
        cand = (-ident, maxd, label, len(common))
        if best is None or cand < best:
            best = cand
    if best is None:
        print(f"{fn:22s} {str(tb):26s} {'(no comparable reference)':30s}")
        continue
    nident, maxd, label, ncommon = best
    rows.append((fn, label))
    print(f"{fn:22s} {str(tb):26s} {label:30s} {ncommon:6d} {-nident:9d} {maxd:8.4f}")
print()
by = {}
for fn, label in rows:
    by.setdefault(label, []).append(fn)
for label, fns in sorted(by.items()):
    print(f"{label}: {len(fns)} matrices -> {', '.join(f[:-9] for f in sorted(fns))}")
