"""Compare EVERY file that differs between git (asix HEAD) and the lab PC, by key rather than by position.

For each pair it reports whether the git version is a strict subset of the lab version with identical shared values
(the harmless case: the lab file simply grew), or whether shared values actually changed.

  summary.json      records keyed on (pooling, layer, seed, feature, probe); metrics compared
  significance.json tests keyed on (block, pooling, layer, probe, comparison); numeric fields compared
  splits.json       per (seed, dataset): test_row_ids compared
  *.jsonl           records keyed on item_id (or the first id-like field); values compared

Memory: one file at a time, released immediately. Usage: python audit/scripts/fix_git_vs_lab_all.py
"""
import csv, gc, json, subprocess, sys

METRICS = ("diag_mean", "offdiag_mean_10", "offdiag_mean_9", "tpr_diag_mean", "tpr_offdiag_mean_10", "generalization_gap_10")
SKIP_FIELDS = {"row_ci95"}


def git_json(rel):
    out = subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True).stdout
    return json.loads(out.decode("utf-8"))


def cmp_keyed(G, L, keyf, valuef, name):
    gk = {keyf(r): r for r in G}
    lk = {keyf(r): r for r in L}
    common = set(gk) & set(lk)
    diffs, worst, worstf = 0, 0.0, ""
    fields = set()
    for k in common:
        for f, a in valuef(gk[k]).items():
            b = valuef(lk[k]).get(f)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                d = abs(a - b)
                if d > 1e-9:
                    diffs += 1; fields.add(f)
                    if d > worst: worst, worstf = d, f
            elif a != b and b is not None:
                diffs += 1; fields.add(f)
    return dict(kind=name, git=len(G), lab=len(L), common=len(common), git_only=len(set(gk) - set(lk)),
                diff_values=diffs, max_delta=worst, max_field=worstf, fields=",".join(sorted(fields)[:6]))


def compare(rel):
    if rel.endswith("summary.json"):
        G, L = git_json(rel)["records"], json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))["records"]
        key = lambda r: tuple(r.get(k) for k in ("pooling", "layer", "seed", "feature", "probe"))
        val = lambda r: {m: r.get(m) for m in METRICS}
        return cmp_keyed(G, L, key, val, "summary")
    if rel.endswith("significance.json"):
        G, L = git_json(rel)["tests"], json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))["tests"]
        kf = [k for k in ("block", "pooling", "layer", "probe", "comparison", "seed", "bank") if k in (G[0] if G else {})]
        key = lambda r: tuple(str(r.get(k)) for k in kf)
        val = lambda r: {k: v for k, v in r.items() if k not in kf and k not in SKIP_FIELDS and not isinstance(v, (list, dict))}
        return cmp_keyed(G, L, key, val, "significance")
    if rel.endswith("splits.json"):
        g, l = git_json(rel)["splits"], json.load(open("audit/cache/labpc/" + rel, encoding="utf-8"))["splits"]
        common = diffs = 0
        for seed in set(g) & set(l):
            for ds in set(g[seed]) & set(l[seed]):
                common += 1
                if g[seed][ds].get("test_row_ids") != l[seed][ds].get("test_row_ids"):
                    diffs += 1
        return dict(kind="splits", git=sum(len(v) for v in g.values()), lab=sum(len(v) for v in l.values()),
                    common=common, git_only=0, diff_values=diffs, max_delta=0, max_field="test_row_ids" if diffs else "", fields="")
    if rel.endswith(".jsonl"):
        def load(lines):
            out = {}
            for ln in lines:
                if not ln.strip():
                    continue
                try: r = json.loads(ln)
                except Exception: continue
                k = r.get("item_id") or r.get("id") or r.get("row_id") or len(out)
                out[k] = r
            return out
        G = load(subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True).stdout.decode("utf-8", "replace").splitlines())
        L = load(open("audit/cache/labpc/" + rel, encoding="utf-8", errors="replace").read().splitlines())
        common = set(G) & set(L)
        diffs = sum(1 for k in common if G[k] != L[k])
        return dict(kind="jsonl", git=len(G), lab=len(L), common=len(common), git_only=len(set(G) - set(L)),
                    diff_values=diffs, max_delta=0, max_field="", fields="")
    return None


rels = [r[0] for r in csv.reader(open("audit/fix/git_vs_lab_pairs.tsv", encoding="utf-8"), delimiter="\t")
        if r and r[0].endswith((".json", ".jsonl"))]
rows = []
for rel in rels:
    try:
        res = compare(rel)
    except Exception as e:
        res = dict(kind="ERROR", git=0, lab=0, common=0, git_only=0, diff_values=-1, max_delta=0, max_field=str(e)[:60], fields="")
    res["file"] = rel
    verdict = ("SUBSET-IDENTICAL" if res["diff_values"] == 0 and res["git_only"] == 0 else
               "ERROR" if res["diff_values"] < 0 else "VALUES DIFFER")
    res["verdict"] = verdict
    rows.append(res)
    print(f"{verdict:17s} {res['kind']:13s} git={res['git']:6d} lab={res['lab']:6d} common={res['common']:6d} "
          f"gitonly={res['git_only']:4d} diffs={res['diff_values']:6d} max|d|={res['max_delta']:.4g} "
          f"[{res['max_field']}] {rel}")
    gc.collect()
with open("audit/fix/git_vs_lab_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "kind", "verdict", "git", "lab", "common", "git_only", "diff_values", "max_delta", "max_field", "fields"])
    w.writeheader()
    for r in rows: w.writerow(r)
print("\nSUBSET-IDENTICAL:", sum(r["verdict"] == "SUBSET-IDENTICAL" for r in rows),
      " VALUES DIFFER:", sum(r["verdict"] == "VALUES DIFFER" for r in rows),
      " ERROR:", sum(r["verdict"] == "ERROR" for r in rows))
