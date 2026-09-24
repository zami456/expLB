"""P17 Agent A, task 3: cell-by-cell drift of every result CSV that changed between two pins.

Usage: python audit/scripts/p17_csv_drift.py [old_sha] [new_sha] [out_csv] [path_prefix]
Defaults: PIN_OLD -> PIN_PAPER, out audit/p17/csv_drift.csv, prefix = results dir.
Rows are aligned on a key made of the non-numeric leading columns (so long-format tables such as
zero_shot_operating.csv align on layer/arm/dataset/threshold_source); columns on header name.
"""
import csv, io, subprocess, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p17_common as C

OLD = sys.argv[1] if len(sys.argv) > 1 else C.PIN_OLD
NEW = sys.argv[2] if len(sys.argv) > 2 else C.PIN_PAPER
OUTF = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else C.ROOT / "audit" / "p17" / "csv_drift.csv"
PREFIX = sys.argv[4] if len(sys.argv) > 4 else C.RES


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def keyed(rows, header):
    # key columns: leading columns until the first column that is numeric in every row,
    # except that a column named 'layer' is always part of the key
    keycols = []
    for h in header:
        vals = [r[h] for r in rows]
        if h == "layer" or any(num(v) is None and v not in ("", "nan") for v in vals):
            keycols.append(h)
        else:
            break
    if not keycols:
        keycols = [header[0]]
    out = {}
    for i, r in enumerate(rows):
        k = tuple(r[h] for h in keycols)
        out[k if k not in out else k + (f"#dup{i}",)] = r
    return keycols, out


def main():
    r = subprocess.run(["git", "diff", "--name-status", OLD, NEW, "--", PREFIX], cwd=C.ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    files = [ln.split("\t") for ln in r.stdout.splitlines() if ln.endswith(".csv")]
    res = []
    for st, path in ((f[0], f[-1]) for f in files):
        new = C.tab(NEW, path) or []
        old = C.tab(OLD, path) if st != "A" else []
        old = old or []
        hn = list(new[0].keys()) if new else []
        ho = list(old[0].keys()) if old else []
        if not old:
            res.append(dict(file=path, status=st, n_changed_cells=0, max_abs_delta="",
                            rows_added=len(new), rows_removed=0, cols_added=len(hn), changed_rows="(new file)"))
            continue
        kc, ko = keyed(old, ho)
        _, kn = keyed(new, hn) if hn[:len(kc)] == kc else (None, {tuple(x[h] for h in kc): x for x in new})
        added = [k for k in kn if k not in ko]
        removed = [k for k in ko if k not in kn]
        ncell, mx, which = 0, 0.0, []
        for k in kn:
            if k not in ko:
                continue
            ch = []
            for h in hn:
                if h not in ho:
                    continue
                a, b = ko[k][h], kn[k][h]
                if a == b:
                    continue
                fa, fb = num(a), num(b)
                if fa is not None and fb is not None:
                    if fa == fb:
                        continue
                    d = abs(fa - fb)
                    mx = max(mx, d)
                    ch.append(f"{h}:{d:.4g}")
                else:
                    ch.append(f"{h}:str")
                ncell += 1
            if ch:
                which.append("/".join(k) + "[" + ",".join(ch[:4]) + ("..." if len(ch) > 4 else "") + "]")
        res.append(dict(file=path, status=st, n_changed_cells=ncell, max_abs_delta=f"{mx:.6g}" if ncell else "0",
                        rows_added=len(added), rows_removed=len(removed),
                        cols_added=len([h for h in hn if h not in ho]),
                        changed_rows=("; ".join(which) + (" | ADDED: " + "; ".join("/".join(a) for a in added) if added else ""))[:900]))
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTF, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    print(f"{OLD[:8]} -> {NEW[:8]}: {len(res)} CSVs changed under {PREFIX}")
    print(f"  new files: {sum(x['status']=='A' for x in res)}; modified: {sum(x['status']=='M' for x in res)}")
    mods = [x for x in res if x["status"] == "M"]
    if mods:
        print(f"  changed cells total {sum(x['n_changed_cells'] for x in mods)}; max |delta| {max(float(x['max_abs_delta'] or 0) for x in mods):.6g}")
        print(f"  rows added total {sum(x['rows_added'] for x in mods)}; cols added {sum(x['cols_added'] for x in mods)}")
    for x in mods:
        print(f"  {x['file'].split('/results/')[-1]:58s} cells {x['n_changed_cells']:3d} max {x['max_abs_delta']:>9s} +rows {x['rows_added']}")


main()
