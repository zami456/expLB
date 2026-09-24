"""F-98 and F-99 - the collaborator's two arms rebuilt from OUR vectors and run on OUR eleven.

F-98: zero-shot mean AUROC over the 11 datasets for `contrast` (= axis1, 8v7) and
`contrast:collab_extended` (= ext_axis1, 12v13) from run-crossgen/kd11/banks/collab, against the
raw single-source trained baseline (off-diagonal mean, paper probe) and the pooled LODO figure
from run-premise. Also the per-dataset decomposition (drop HP-KR, drop HP-KR + sandbagging_v2)
and the provenance cross-check against the paper16 bank's own deception contrast.

F-99: the same ext - axis1 difference at every stored layer, with the share of the margin over raw.

Usage: python F-98.py [<kd11 root>] [<run-premise dir>] [<paper16 kd11-style store>]
"""
import glob, json, os, re, statistics, sys

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
premise = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/user4/expLB/run-premise"


def zs(bank, L, pool="mean"):
    p = os.path.join(root, bank, "matrices", f"{pool}_L{L}_s42.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return d


def offdiag(d, feat, probe="paper"):
    ds = d["datasets"]
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][i][j] for i in range(len(ds)) for j in range(len(ds))
                 if i != j and r["auroc"][i][j] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


def zmean(d, direction, drop=()):
    ds = d["datasets"]
    z = next((x for x in d["zeroshot"] if x["direction"] == direction), None)
    if z is None:
        return None, None
    per = {ds[i]: z["auroc"][i] for i in range(len(ds)) if z["auroc"][i] is not None}
    keep = {k: v for k, v in per.items() if k not in drop}
    return statistics.fmean(keep.values()), per


print("#" * 78)
print("F-98: two layers, our eleven datasets")
for L in (20, 40):
    d = zs("banks/collab", L)
    b = zs("", L)
    if d is None or b is None:
        print(f"  L{L}: MISSING"); continue
    print("=" * 78)
    print(f"L{L}  collab bank={d.get('bank')} name={d.get('bank_name')} k={d.get('k')}"
          f" | baseline bank={b.get('bank')} name={b.get('bank_name')}")
    raw, n = offdiag(b, "raw")
    a1, pera = zmean(d, "contrast")
    e1, pere = zmean(d, "contrast:collab_extended")
    print(f"  raw single-source trained baseline (off-diag, paper) = {raw:.4f}  (n={n} pairs)")
    print(f"  axis1      `contrast`                   zero-shot mean = {a1:.4f}   margin {a1-raw:+.4f}")
    print(f"  ext_axis1  `contrast:collab_extended`   zero-shot mean = {e1:.4f}   margin {e1-raw:+.4f}")
    print(f"  kd bank worth (ext - axis1) = {e1-a1:+.4f}   share of the margin = {100*(e1-a1)/(e1-raw):.1f}%")
    # pooled LODO
    pp = os.path.join(premise, f"premise_L{L}_mean_s42.json")
    if os.path.exists(pp):
        st = os.stat(pp)
        pj = json.load(open(pp, encoding="utf-8"))
        s = pj.get("summary", {})
        print(f"  run-premise file: {pp}  size={st.st_size}  mtime={st.st_mtime}")
        for k, v in (s.items() if isinstance(s, dict) else []):
            if isinstance(v, (int, float)):
                print(f"    summary.{k} = {v:.4f}")
            elif isinstance(v, dict):
                print(f"    summary.{k} = " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()
                                                        if isinstance(vv, (int, float))))
    else:
        print(f"  run-premise file MISSING: {pp}")
    # decomposition
    for drop in ((), ("harm-pressure-knowledge-report",),
                 ("harm-pressure-knowledge-report", "sandbagging_v2")):
        aa, _ = zmean(d, "contrast", drop)
        ee, _ = zmean(d, "contrast:collab_extended", drop)
        print(f"  drop {str(drop) if drop else '(nothing)':<60} ext - axis1 = {ee-aa:+.4f}")
    print("  per-dataset zero-shot, axis1 -> ext_axis1:")
    for k in pera:
        print(f"    {k:<34}{pera[k]:>8.4f} ->{pere[k]:>8.4f}   {pere[k]-pera[k]:+.4f}")
    # provenance: the paper16 bank's own deception contrast on the same rows
    pb = zs("", L)
    pz, _ = zmean(pb, "contrast")
    print(f"  provenance: paper16 bank's own `contrast` (s3 vectors) zero-shot mean = {pz:.4f}"
          f"  vs our axis1 from vectors_s23 = {a1:.4f}  gap = {abs(pz-a1):.4f}")

print("\n" + "#" * 78)
print("F-99: every stored layer (mean pooling)")
rows = []
for fp in sorted(glob.glob(os.path.join(root, "banks/collab", "matrices", "mean_L*_s42.json")),
                 key=lambda s: int(re.search(r"_L(\d+)_", s).group(1))):
    L = int(re.search(r"_L(\d+)_", fp).group(1))
    d = json.load(open(fp, encoding="utf-8"))
    b = zs("", L)
    if b is None:
        continue
    raw, _ = offdiag(b, "raw")
    a1, pera = zmean(d, "contrast")
    e1, pere = zmean(d, "contrast:collab_extended")
    share = 100 * (e1 - a1) / (e1 - raw) if (e1 - raw) != 0 else float("nan")
    rows.append((L, a1, e1, raw, e1 - a1, share,
                 pera.get("harm-pressure-knowledge-report"), pere.get("harm-pressure-knowledge-report")))
    print(f"  L{L:<3} axis1={a1:.4f} ext={e1:.4f} raw={raw:.4f} | ext-axis1={e1-a1:+.4f}"
          f" share={share:6.1f}%  | HP-KR axis1={pera.get('harm-pressure-knowledge-report'):.3f}"
          f" -> ext={pere.get('harm-pressure-knowledge-report'):.3f}"
          + ("   axis1 BELOW raw" if a1 < raw else ""))
if rows:
    d_ = [r[4] for r in rows]; sh = [r[5] for r in rows]
    print(f"  positive at {sum(1 for x in d_ if x > 0)} of {len(d_)} layers")
    print(f"  ext - axis1: mean {statistics.fmean(d_):+.4f}  range [{min(d_):+.4f} (L{rows[d_.index(min(d_))][0]}),"
          f" {max(d_):+.4f} (L{rows[d_.index(max(d_))][0]})]")
    print(f"  share: median {statistics.median(sh):.1f}%  mean {statistics.fmean(sh):.1f}%"
          f"  range [{min(sh):.1f}%, {max(sh):.1f}%]")
    print(f"  layers where axis1 < raw: " + ", ".join(f"L{r[0]} ({r[1]:.4f} vs {r[3]:.4f})"
                                                     for r in rows if r[1] < r[3]))
